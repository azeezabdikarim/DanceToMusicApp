import os
import numpy as np
import torch
from tqdm import tqdm
import pickle as pkl
import time

import torch
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
from torch.optim import Adam
from transformers import EncodecModel
from models import Pose2AudioTransformer, AudioDescriminator
from utils import DanceToMusic
from utils.loss_helpers import *
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
import torchaudio.transforms as T



# Define a function to save the model and delete the last saved model
def save_model(model, folder_path, loss, last_saved_model, name = ''):
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
    # assign GPU or CPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # device = torch.device("cpu")


    model_id = "facebook/encodec_24khz"
    encodec_model = EncodecModel.from_pretrained(model_id)
    codebook_size = encodec_model.quantizer.codebook_size
    # encodec_model.to(device)
    # processor = AutoProcessor.from_pretrained(model_id)
    
    sample_rate = 24000
    batch_size = 1 # Batch size for Nvididias GTX 3080 9.88/10GB

    # data_dir = '/home/azeez/Documents/projects/DanceToMusicApp/ml/data/samples/5sec_expando_dnb_min_training_data'
    # data_dir = "/Users/azeez/Documents/pose_estimation/DanceToMusicApp/ml/data/samples/5sec_expando_dnb_min_training_data"
    # data_dir = '/Users/azeez/Documents/pose_estimation/DanceToMusicApp/ml/data/samples/5sec_expando_test'
    data_dir = '/home/azeez/Documents/projects/DanceToMusicApp/ml/data/samples/3sec_24fps_expando_dnb_min_training_data'
    train_dataset = DanceToMusic(data_dir, encoder = encodec_model, sample_rate = sample_rate, device=device, dnb = True)
    embed_size = train_dataset.data['poses'].shape[2] * train_dataset.data['poses'].shape[3]

    target_shape = train_dataset.data['audio_codes'][0].shape
    descriminator = AudioDescriminator(target_shape, hidden_units = 40, num_hidden_layers = 3, alpha = 0.01, sigmoid_out = True)
    descriminator.to(device)


    train_ratio = 0.8 # Define the split ratio
    val_ratio = 1 - train_ratio
    dataset_size = len(train_dataset)
    train_len = int(train_ratio * dataset_size)
    val_len = dataset_size - train_len

    # Randomly split the dataset
    train_dataset, val_dataset = random_split(train_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    
    src_pad_idx = 0
    trg_pad_idx = 0
    # device = torch.device("cpu")
    pose_model = Pose2AudioTransformer(codebook_size, src_pad_idx, trg_pad_idx, device=device, num_layers=4, heads = 4, embed_size=embed_size, dropout=0.1)
    pose_model.to(device)
    
    # weights = 'DanceToMusicApp/ml/model_weights/5_sec_dnb_best_model_weights_loss_4.911053791451962.pth'
    # weights = '/home/azeez/Documents/projects/DanceToMusicApp/ml/model_weights/5_sec_dnb_best_model_weights_loss_4.911053791451962.pth'
    # pose_model.load_state_dict(torch.load(weights, map_location=device))

    learning_rate = 1e-4
    criterion_g = torch.nn.NLLLoss()
    criterion_d = torch.nn.BCELoss()
    mel_spectrogram_transform = T.MelSpectrogram(sample_rate=24000, n_fft=2048, hop_length=512, n_mels=128).to(device)


    # criterion = MSELoss()
    optimizer_g = torch.optim.Adam(pose_model.parameters(), lr=learning_rate)
    optimizer_d = torch.optim.Adam(descriminator.parameters(), lr=learning_rate)


    # Set up for tracking the best model
    # weights_dir = '/home/azeez/Documents/projects/DanceToMusicApp/ml/model_weights'
    # weights_dir = '/Users/azeez/Documents/pose_estimation/DanceToMusicApp/ml/model_weights'
    best_loss = float('inf')  # Initialize with a high value
    g_last_saved_model = ''
    d_last_saved_model = ''
    encodec_last_saved_model = ''

    weights_dir = '/home/azeez/Documents/projects/DanceToMusicApp/ml/model_weights'
    # Generate a unique directory name based on current date and time
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join('./my_logs', 'run_' + current_time)
    os.makedirs(log_dir, exist_ok=True)
    model_save_dir = os.path.join('./model_saves', 'run_' + current_time)
    os.makedirs(model_save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # accumulation_steps = 12  # Number of mini-batches for gradient accumulation

    teacher_forcing_ratio = 0.5 # 50% of the time we will use teacher forcing
    val_epoch_interval = 5
    num_epochs = 30000
    CLIP_VALUE = 0.01 # weight clipping value for WGAN
    N_CRITIC = 4 # Number of times to train the discriminator per generator training step
    for epoch in range(num_epochs):
        pose_model.train()
        descriminator.train()
        # encodec_model.decoder.train()
        epoch_loss = 0  # Initialize epoch_loss
        timesteps = 0

        total_loss_g = 0  # Total generator loss
        total_loss_d = 0  # Total discriminator loss
        total_steps = 0  # To track the number of steps
        batch_nll_loss = 0
        total_d_steps = 0
        total_g_steps = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (audio_codes, pose, pose_mask, wav, wav_mask, wav_path, sample_rate) in progress_bar:            
            

            # Forward pass
            wav = wav.unsqueeze(1)
            target = audio_codes.to(device)
            target_for_loss = target[:, 1:, :]
            
            input_for_next_step = target[:, 0:1, :]
            total_nll_loss = 0
            
            src = pose.to(device)
            enc_mask = pose_model.make_src_mask(src)
            trg_mask = pose_model.make_trg_mask(input_for_next_step.to(device))
            
            B, N, _, _ = pose.shape
            enc_context = pose_model.encoder(src.view(B, N, -1), enc_mask)
            
            for t in range(1, target.shape[1]):
                output_softmax, output_argmax, offset, _ = pose_model.decoder(input_for_next_step.to(device), enc_context, enc_mask, trg_mask)
                
                log_softmax_output = torch.log(output_softmax[:,-2:,:])
                log_softmax_output_reshape = log_softmax_output.view(-1, log_softmax_output.shape[2])
                reshaped_target = target[:, t, :].reshape(-1).long()
                time_step_nll_loss = criterion_g(log_softmax_output_reshape, reshaped_target)
                total_nll_loss += time_step_nll_loss
                batch_nll_loss += time_step_nll_loss
                
                use_teacher_forcing = np.random.random() < teacher_forcing_ratio
                if use_teacher_forcing:
                    input_for_next_step = torch.cat([input_for_next_step, target[:, t:t+1, :]], dim=1)  # Concatenate along the sequence length dimension
                else:
                    next_token = output_softmax[:,-2:,:].argmax(dim=2)
                    input_for_next_step = torch.cat([input_for_next_step, next_token.unsqueeze(1)], dim=1)
                trg_mask = pose_model.make_trg_mask(input_for_next_step.to(device))
                
                timesteps += 1

            # Calculate loss back on spectrgorams generated by the model
            # generated_wav = audioCodeToWav(input_for_next_step, encodec_model, sample_rate = 24000)
            # generated_spec = mel_spectrogram_transform(generated_wav)
            # target_spec = mel_spectrogram_transform(wav.squeeze(1))
            # # generated_spec = wav_to_mel_spectrogram(generated_wav.detach().cpu().numpy(), sr=24000, n_fft=2048, hop_length=512, n_mels=128)
            # # target_spec = wav_to_mel_spectrogram(wav[0][0].cpu().numpy(), sr=24000, n_fft=2048, hop_length=512, n_mels=128)
            # min_length = min(generated_spec.shape[-1], target_spec.shape[-1])
            # mel_mse_loss = ((generated_spec[0,:,:,:min_length] - target_spec[:,:,:min_length])**2).mean()
            # mel_mse_loss = mel_mse_loss/200000


            optimizer_d.zero_grad()
            # Train Discriminator on Real Data
            real_data = audio_codes.to(device)
            real_output = descriminator(real_data)
            loss_d_real = -torch.mean(real_output)

            # Train Discriminator on Fake Data
            fake_data = input_for_next_step.detach()
            fake_output = descriminator(fake_data)
            loss_d_fake = torch.mean(fake_output)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            log_gradients(descriminator, epoch * len(train_loader) + i, writer, tag_prefix='Discriminator_Gradients')
            optimizer_d.step()
            total_d_steps += 1
            total_loss_d += loss_d.item()
            avg_epoch_loss_d = total_loss_d / total_d_steps

            for p in descriminator.parameters():
                p.data.clamp_(-CLIP_VALUE, CLIP_VALUE)

            if (i + 1) % N_CRITIC == 0:
                optimizer_g.zero_grad()
                batch_nll_loss = batch_nll_loss / (timesteps)

                fake_data = input_for_next_step
                fake_output = descriminator(fake_data)
                adversarial_loss_g = -torch.mean(fake_output)

                combined_loss_g = batch_nll_loss + adversarial_loss_g #+ mel_mse_loss
                combined_loss_g.backward()
                optimizer_g.step()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                total_g_steps += 1
                total_loss_g += combined_loss_g.item()
                avg_epoch_loss_g = total_loss_g / total_g_steps

                writer.add_scalar('Loss/Genreal_outputerator/Batch NLL_Loss', batch_nll_loss.item(), epoch * len(train_loader) + i)
                writer.add_scalar('Loss/Generator/Average_Epcoch_Loss', avg_epoch_loss_g, epoch * len(train_loader) + i)
                # writer.add_scalar('Loss/Generator/Mel_MSE_Loss', mel_mse_loss.item(), epoch * len(train_loader) + i)
                writer.add_scalar('Loss/Generator/Adversarial_Loss', adversarial_loss_g.item(), epoch * len(train_loader) + i)
                writer.add_scalar('Loss/Discriminator', loss_d.item(), epoch * len(train_loader) + i)
                writer.add_scalar('Loss/Discriminator/Real', loss_d_real.item(), epoch * len(train_loader) + i)
                writer.add_scalar('Loss/Discriminator/Fake', loss_d_fake.item(), epoch * len(train_loader) + i)

                # print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Generator Loss: {avg_batch_loss_g:.4f}, Discriminator Loss: {avg_batch_loss_d:.4f}", end='')
                progress_bar.set_description(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Generator Loss: {avg_epoch_loss_g:.4f}, Discriminator Loss: {avg_epoch_loss_d:.4f}")
                print(f"\rEpoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Generator Loss: {avg_epoch_loss_g:.4f}, Discriminator Loss: {avg_epoch_loss_d:.4f}", end='')

                if batch_nll_loss < best_loss:
                    best_loss = batch_nll_loss
                    g_last_saved_model = save_model(pose_model, weights_dir, best_loss, g_last_saved_model, name='gen_3_sec_dnb_')
                    d_last_saved_model = save_model(descriminator, weights_dir, best_loss, d_last_saved_model, name='disc_3_sec_dnb_')
            
        # Compute average epoch loss
        avg_epoch_loss_g = total_loss_g / total_steps
        avg_epoch_loss_d = total_loss_d / total_steps
        writer.add_scalar('Epoch/Average Generator Loss', avg_epoch_loss_g, epoch)
        writer.add_scalar('Epoch/Average Discriminator Loss', avg_epoch_loss_d, epoch)
        print(f"\nEnd of Epoch [{epoch+1}/{num_epochs}], Average Generator Loss: {avg_epoch_loss_g:.4f}, Average Discriminator Loss: {avg_epoch_loss_d:.4f}")
        # Save the best model, by checking if the new model is perfroming better than the previous best model
        
        if epoch % val_epoch_interval == 0:
            pose_model.eval()
            val_loss = 0
            val_steps = 0

            with torch.no_grad():
                for audio_codes, pose, pose_mask, wav, wav_mask, _, _ in val_loader:
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
                    val_steps += 1

                avg_val_loss = val_loss / val_steps
                writer.add_scalar('Validation Loss', avg_val_loss, epoch)
                print(f"\n Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

        else:
            print(f"\n Epoch [{epoch+1}/{num_epochs}]")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    writer.close()


if __name__ == "__main__":
    main()