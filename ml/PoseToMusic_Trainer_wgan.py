import os
import numpy as np
import torch
from tqdm import tqdm
import pickle as pkl
import time
import gc
import argparse

import torch
import shutil
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
from torch.optim import Adam
from transformers import EncodecModel
from models import Pose2AudioTransformer, AudioCodeDiscriminator, MelSpectrogramDiscriminator
from utils import DanceToMusic
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from utils.loss_helpers import *
from DanceToMusicApp.ml.utils.training_utils import *
import torchaudio.transforms as T
from torch.profiler import profile, record_function, ProfilerActivity


def save_model(model, folder_path, loss, last_saved_model, name = ''):
    # saves the current model weights and deletes the last best model
    model_name = f"{name}_best_model_{loss:.4f}.pt"
    model_path = os.path.join(folder_path, model_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    torch.save(model.state_dict(), model_path)
    if last_saved_model:
        try:
            os.remove(os.path.join(folder_path, last_saved_model))
        except FileNotFoundError:
            print(f"Warning: Could not find {last_saved_model} to delete.")
    return model_name

def main():
    args = parse_args()

    # assign GPU or CPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model_id = args.encodec_model_id
    encodec_model = EncodecModel.from_pretrained(model_id)
    codebook_size = encodec_model.quantizer.codebook_size
    encodec_model.to(device)
    sample_rate = args.sample_rate

    data_dir = args.data_dir
    train_dataset = DanceToMusic(data_dir, encoder = encodec_model, sample_rate = sample_rate, device=device, dnb = True)
    # input_size = train_dataset.data['poses'].shape[2] * train_dataset.data['poses'].shape[3]
    # embed_size = 32
    input_size = None
    embed_size = train_dataset.data['poses'].shape[2] * train_dataset.data['poses'].shape[3]    
    target_shape = train_dataset.data['audio_codes'][0].shape

    discriminator_hidden_units = args.discriminator_hidden_units
    discriminator_num_hidden_layers = args.discriminator_num_hidden_layers
    discriminator_alpha = args.discriminator_alpha
    discriminator = AudioCodeDiscriminator(target_shape, 
                                       hidden_units = discriminator_hidden_units, 
                                       num_hidden_layers = discriminator_num_hidden_layers, 
                                       alpha = discriminator_alpha, 
                                       sigmoid_out = False)
    discriminator.to(device)


    train_ratio = args.train_ratio # Define the split ratio
    dataset_size = len(train_dataset)
    train_len = int(train_ratio * dataset_size)
    val_len = dataset_size - train_len

    # Randomly split the dataset
    random_seed = args.random_seed
    train_dataset, val_dataset = random_split(train_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(random_seed))
    
    batch_size = args.batch_size 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    
    src_pad_idx = 0
    trg_pad_idx = 0
    pose_model = Pose2AudioTransformer(codebook_size, src_pad_idx, trg_pad_idx, 
                                       device = device, 
                                       num_layers = args.pose2audio_num_layers, 
                                       heads = args.pose2audio_num_heads, 
                                       embed_size = embed_size, 
                                       dropout = args.pose2audio_dropout, 
                                       input_size = input_size)
    pose_model.to(device)
    
    # Load the starting weights if provided ex. earlier checkpoitn
    if args.starting_weights_path != None:
        weights = args.starting_weights_path
        pose_model.load_state_dict(torch.load(weights, map_location=device))

    learning_rate = args.learning_rate
    criterion_g = torch.nn.NLLLoss()
    mel_spectrogram_transform = T.MelSpectrogram(sample_rate=24000, n_fft=2048, hop_length=256, n_mels=64).to(device)

    # Prepare encodec model for training, freeze all parameters except the final conv1d layer of the decoder    
    for param in encodec_model.parameters():
        param.requires_grad = False # freeze all parameters of the encoded model
    for param in encodec_model.decoder.layers[-1].parameters(): 
        param.requires_grad = True # unfreeze just the final conv1d layer of the encodec decoder 

    # criterion = MSELoss()
    generator_params = list(pose_model.parameters()) + list(encodec_model.decoder.layers[-1].parameters())
    optimizer_g = torch.optim.Adam(generator_params, lr=learning_rate)
    # optimizer_g = torch.optim.Adam(pose_model.parameters(), lr=learning_rate)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)


    # Set up for tracking the best model
    weights_dir = args.weights_dir
    best_loss = float('inf')  # Initialize with a high value
    g_last_saved_model = ''
    d_last_saved_model = ''
    encodec_last_saved_model = ''

    # Generate a unique directory name based on current date and time
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join('./my_logs', 'run_' + current_time)
    config_dr = os.path.join(log_dir, 'config_file')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(config_dr, exist_ok=True)
    shutil.copy(args.config, config_dr)
    model_save_dir = os.path.join(log_dir,'model_saves_and_validation_samples')
    os.makedirs(model_save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    teacher_forcing_ratio = args.teacher_forcing_ratio # 50% of the time we will use teacher forcing
    val_epoch_interval = args.val_epoch_interval # 5
    num_epochs = args.num_epochs 
    CLIP_VALUE = args.clip_value # 0.01 weight clipping value for WGAN
    N_CRITIC = args.n_critic # number of times to train the discriminator per generator training step

    rolling_mel_mse_loss = []
    rolling_batch_nll_loss = []
    for epoch in range(num_epochs):
        pose_model.train()
        discriminator.train()
        encodec_model.decoder.train()
        epoch_loss = 0  # Initialize epoch_loss
        total_timesteps = 0

        total_loss_g = 0  # Total generator loss
        total_loss_d = 0  # Total discriminator loss
        total_g_steps = 0  # To track the number of steps
        total_d_steps = 0  # To track the number of steps
        epoch_total_nll_loss = 0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (audio_codes, pose, pose_mask, wav, wav_mask, _, vid_path, _) in progress_bar:    
            optimizer_d.zero_grad()
            # snapshot_filename = os.path.join(model_save_dir, f"memory_snapshot_epoch_{epoch}.pickle")
            # torch.cuda.memory._dump_snapshot(snapshot_filename)

            # Forward pass
            wav = wav.unsqueeze(1)
            target = audio_codes.to(device)
            target_for_loss = target[:, 1:, :]
            
            input_for_next_step = target[:, 0:1, :]
            
            src = pose.to(device)
            enc_mask = pose_model.make_src_mask(src)
            trg_mask = pose_model.make_trg_mask(input_for_next_step.to(device))
            
            B, N, _, _ = pose.shape
            enc_context = pose_model.encoder(src.view(B, N, -1), enc_mask)
            
            batch_nll_loss = 0
            batch_time_steps = 0

            for t in range(1, target.shape[1]):
                output_softmax, output_argmax, offset, _ = pose_model.decoder(input_for_next_step.to(device), enc_context, enc_mask, trg_mask)
                
                log_softmax_output = torch.log(output_softmax[:,-2:,:])
                log_softmax_output_reshape = log_softmax_output.view(-1, log_softmax_output.shape[2])
                reshaped_target = target[:, t, :].reshape(-1).long()
                time_step_nll_loss = criterion_g(log_softmax_output_reshape, reshaped_target)
                epoch_total_nll_loss += time_step_nll_loss.detach().item()
                batch_nll_loss += time_step_nll_loss
                
                use_teacher_forcing = np.random.random() < teacher_forcing_ratio
                if use_teacher_forcing:
                    input_for_next_step = torch.cat([input_for_next_step, target[:, t:t+1, :]], dim=1)  # Concatenate along the sequence length dimension
                else:
                    next_token = output_softmax[:,-2:,:].argmax(dim=2)
                    input_for_next_step = torch.cat([input_for_next_step, next_token.unsqueeze(1)], dim=1)
                trg_mask = pose_model.make_trg_mask(input_for_next_step.to(device))
                
                total_timesteps += 1*B # Increment the timesteps by the batch size so that we can accuratly calculate the average loss per timestep
                batch_time_steps += 1*B

            # Train Discriminator on Real Data
            real_data = audio_codes.to(device)
            real_output = discriminator(real_data)
            loss_d_real = -torch.mean(real_output)

            # Train Discriminator on Fake Data
            fake_data = input_for_next_step.detach()
            fake_output = discriminator(fake_data)
            loss_d_fake = torch.mean(fake_output)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optimizer_d.step()
            total_d_steps += 1
            total_loss_d += loss_d.detach().item()
            avg_epoch_loss_d = total_loss_d / total_d_steps

            for p in discriminator.parameters():
                p.data.clamp_(-CLIP_VALUE, CLIP_VALUE)

            writer.add_scalar('Loss_Discriminator', loss_d.item(), epoch * len(train_loader) + i)
            writer.add_scalar('Loss_Discriminator/Real', loss_d_real.item(), epoch * len(train_loader) + i)
            writer.add_scalar('Loss_Discriminator/Fake', loss_d_fake.item(), epoch * len(train_loader) + i)


            if (i + 1) % N_CRITIC == 0:
                # Calculate perceptual loss. derive it by comparing the generated vs target mel spectrgorams 
                optimizer_g.zero_grad()
                generated_wav = audioCodeToWav(input_for_next_step, encodec_model, sample_rate = 24000)
                generated_spec = mel_spectrogram_transform(generated_wav)
                target_spec = mel_spectrogram_transform(wav.squeeze(1))
                min_length = min(generated_spec.shape[-1], target_spec.shape[-1])
                mel_mse_loss = ((generated_spec[:,:,:,:min_length] - target_spec[:,:,:,:min_length])**2).mean()
                mel_mse_loss = mel_mse_loss

                avg_nll_loss = epoch_total_nll_loss / (total_timesteps)
                batch_nll_loss = batch_nll_loss / (batch_time_steps)

                fake_data = input_for_next_step
                fake_output = discriminator(fake_data)
                adversarial_loss_g = -torch.mean(fake_output)
                
                # NLL Loss on the generated audio codes and the perceptual loss from the mel spectrograms are scaled
                # strategy: scale the loss value by the average of the last 20 loss values 
                if len(rolling_mel_mse_loss) < 20:
                    rolling_mel_mse_loss.append(mel_mse_loss.detach().item())
                    rolling_batch_nll_loss.append(batch_nll_loss.detach().item())
                else:
                    rolling_mel_mse_loss.pop(0)
                    rolling_batch_nll_loss.pop(0)
                    rolling_mel_mse_loss.append(mel_mse_loss.detach().item())
                    rolling_batch_nll_loss.append(batch_nll_loss.detach().item())
                
                # take the average of a list
                scaled_mel_mse_loss = mel_mse_loss / (sum(rolling_mel_mse_loss)/len(rolling_mel_mse_loss))

                combined_loss_g = (batch_nll_loss) + adversarial_loss_g + (scaled_mel_mse_loss)
                combined_loss_g.backward()
                optimizer_g.step()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                total_g_steps += 1
                total_loss_g += combined_loss_g.detach().item()
                avg_epoch_loss_g = total_loss_g / total_g_steps

                writer.add_scalar('Loss_Generator/1_Combined_Loss', combined_loss_g.item(), epoch * len(train_loader) + i)
                writer.add_scalar('Loss_Generator/2_Batch_NLL_Loss', batch_nll_loss.item(), epoch * len(train_loader) + i)
                writer.add_scalar('Loss_Generator/3_Mel_MSE_Loss_Scaled', scaled_mel_mse_loss.item(), epoch * len(train_loader) + i)
                writer.add_scalar('Loss_Generator/4_Adversarial_Loss', adversarial_loss_g.item(), epoch * len(train_loader) + i)
                writer.add_scalar('Loss_Generator/5_Average_Epcoch_Loss', avg_epoch_loss_g, epoch * len(train_loader) + i)

                # print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Generator Loss: {avg_batch_loss_g:.4f}, Discriminator Loss: {avg_batch_loss_d:.4f}", end='')
                progress_bar.set_description(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Generator Loss: {avg_epoch_loss_g:.4f}, Discriminator Loss: {avg_epoch_loss_d:.4f}")
                print(f"\rEpoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Generator Loss: {combined_loss_g:.4f}, Discriminator Loss: {avg_epoch_loss_d:.4f}", end='')
            
            if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            gc.collect()
            # writer.add_profile(profiler=prof, global_step=epoch)  # Log to TensorBoard
        # Compute average epoch loss
        writer.add_scalar('Average Generator Loss', avg_epoch_loss_g, epoch)
        writer.add_scalar('Average Discriminator Loss', avg_epoch_loss_d, epoch)
        print(f"\nEnd of Epoch [{epoch+1}/{num_epochs}], Average Generator Loss: {avg_epoch_loss_g:.4f}, Average Discriminator Loss: {avg_epoch_loss_d:.4f}")
        # Save the best model, by checking if the new model is perfroming better than the previous best model
        
        
        if epoch % val_epoch_interval == 0:
            pose_model.eval()
            encodec_model.eval()
            val_loss = 0
            val_steps = 0

            with torch.no_grad():
                for audio_codes, pose, pose_mask, wav, wav_mask, wav_path, vid_path, _ in val_loader:
                    target = audio_codes.to(device)
                    input_for_next_step = target[:, 0:1, :]
                    
                    src = pose.to(device)
                    enc_mask = pose_model.make_src_mask(src)
                    trg_mask = pose_model.make_trg_mask(input_for_next_step.to(device))

                    B, N, _, _ = pose.shape
                    enc_context = pose_model.encoder(src.view(B, N, -1), enc_mask)
                    total_nll_loss = 0

                    for t in range(1, target.shape[1]):
                        output_softmax, output_argmax, offset, _ = pose_model.decoder(input_for_next_step.to(device), enc_context, enc_mask, trg_mask)

                        log_softmax_output = torch.log(output_softmax[:,-2:,:])
                        log_softmax_output_reshape = log_softmax_output.view(-1, log_softmax_output.shape[2])
                        reshaped_target = target[:, t, :].reshape(-1).long()
                        time_step_nll_loss = criterion_g(log_softmax_output_reshape, reshaped_target)
                        total_nll_loss += time_step_nll_loss

                        next_token = output_softmax[:,-2:,:].argmax(dim=2)
                        input_for_next_step = torch.cat([input_for_next_step, next_token.unsqueeze(1)], dim=1)
                        trg_mask = pose_model.make_trg_mask(input_for_next_step.to(device))

                    avg_nll_loss = total_nll_loss / (target.shape[1] - 1)
                    val_loss += avg_nll_loss.item()
                    val_steps += 1*B

                avg_val_loss = val_loss / val_steps
                writer.add_scalar('Validation Loss', avg_val_loss, epoch)
                print(f"\n Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

                # Save a model checkpoint and validation sample
                epoch_model_save_dir = os.path.join(model_save_dir, f"epoch_{epoch+1}")
                save_model(pose_model, epoch_model_save_dir, avg_val_loss, '', name='gen_3_sec_dnb_')
                save_model(discriminator, epoch_model_save_dir, avg_val_loss, '', name='disc_3_sec_dnb_')
                save_model(encodec_model, epoch_model_save_dir, avg_val_loss, '', name='encodec_3_sec_dnb_')
                
                save_valdiation_sample(input_for_next_step[-1], encodec_model, vid_path[-1], epoch, epoch_model_save_dir)
                gc.collect()
        else:
            print(f"\n Epoch [{epoch+1}/{num_epochs}]")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    writer.close()


if __name__ == "__main__":
    main()