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
import torchaudio
import moviepy.editor as mp
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
from utils.training_utils import *
import torchaudio.transforms as T
from torch.profiler import profile, record_function, ProfilerActivity

def initialize_model_and_data(mel_spectrogram_transform, args, device):
    """
    Initialize the models and set up the device.

    Args:
    - mel_spectrogram_transform: The mel spectrogram transform, used to setup input size of the discriminator.
    - args: Command-line arguments or configuration parameters.
    - device: The device (CPU or GPU) to run the models on.
    Returns:
    - encodec_model: The initialized EncodecModel.
    - pose_model: The initialized Pose2AudioTransformer model.
    - discriminator: The initialized discriminator model.
    - train_dataset: The initialized training dataset.
    """

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
    
    d_input_size = mel_spectrogram_transform(train_dataset.data['wavs'][0]).shape
    discriminator = MelSpectrogramDiscriminator(d_input_size, discriminator_hidden_units, discriminator_num_hidden_layers, discriminator_alpha) 
    discriminator.to(device)

    pose_model = Pose2AudioTransformer(codebook_size, 
                                       args.src_pad_idx, 
                                       args.trg_pad_idx, 
                                       device = device, 
                                       num_layers = args.pose2audio_num_layers, 
                                       heads = args.pose2audio_num_heads, 
                                       embed_size = embed_size, 
                                       dropout = args.pose2audio_dropout, 
                                       input_size = input_size)
    pose_model.to(device)

    return encodec_model, pose_model, discriminator, train_dataset

def validation_step(pose_model, val_loader, criterion, device, tensorboard_writer, epoch, num_epochs, model_save_dir):
    """
    Executes validation steps for one epoch.
    Args:
    - pose_model: The Pose2AudioTransformer model.
    - val_loader: DataLoader for validation data.
    - criterion: Loss function.
    - device: Computation device.
    - tensorboard_writer: TensorBoard writer for logging.
    - epoch: Current epoch number.
    - num_epochs: Total number of epochs.
    - model_save_dir: Directory for saving models.
    Returns:
    - avg_val_loss: Average validation loss.
    - generated_audio_codes: Last batch of generated audio codes.
    - vid_paths: Video paths of last batch.
    """

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
                time_step_nll_loss = criterion(log_softmax_output_reshape, reshaped_target)
                total_nll_loss += time_step_nll_loss

                next_token = output_softmax[:,-2:,:].argmax(dim=2)
                input_for_next_step = torch.cat([input_for_next_step, next_token.unsqueeze(1)], dim=1)
                trg_mask = pose_model.make_trg_mask(input_for_next_step.to(device))

            avg_nll_loss = total_nll_loss / (target.shape[1] - 1)
            val_loss += avg_nll_loss.item()
            val_steps += 1*B

        avg_val_loss = val_loss / val_steps
        tensorboard_writer.add_scalar('Validation Loss', avg_val_loss, epoch)
        print(f"\n Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

        return avg_val_loss, input_for_next_step, vid_path
    
def train_one_epoch(pose_model, discriminator, encodec_model, mel_spectrogram_transform, train_loader, criterion_g, optimizer_g, optimizer_d, device, tensorboard_writer, epoch, args):
    """
    Trains models for one epoch.
    Args:
    - pose_model: Pose2AudioTransformer model.
    - discriminator: Discriminator model.
    - encodec_model: EncodecModel.
    - mel_spectrogram_transform: Mel spectrogram transform.
    - train_loader: DataLoader for training data.
    - criterion_g: Generator's loss function.
    - optimizer_g: Generator's optimizer.
    - optimizer_d: Discriminator's optimizer.
    - device: Computation device.
    - tensorboard_writer: TensorBoard writer.
    - epoch: Current epoch number.
    - args: Configuration parameters.
    Returns:
    - Updated pose_model, discriminator, encodec_model.
    - Dictionary of average losses.
    """
    total_timesteps = 0
    total_loss_g = 0  # Total generator loss
    total_loss_d = 0  # Total discriminator loss
    total_g_steps = 0  # To track the number of generator steps
    total_d_steps = 0  # To track the number of discriminator steps
    epoch_total_nll_loss = 0

    N_CRITIC = args.n_critic # number of times to train the discriminator per generator training step
    teacher_forcing_ratio = args.teacher_forcing_ratio # 50% of the time we will use teacher forcing
    num_epochs = args.num_epochs

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (audio_codes, pose, pose_mask, wav, wav_mask, wav_path, vid_path, _) in progress_bar:    
        optimizer_d.zero_grad()
        wav = wav.unsqueeze(1)
        target = audio_codes.to(device)        
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

        # Train Discriminator
        optimizer_d.zero_grad()

        fake_wavs = audioCodeToWav(input_for_next_step, encodec_model, sample_rate = 24000)
        fake_spec = mel_spectrogram_transform(fake_wavs).detach()
        real_spec = mel_spectrogram_transform(wav.squeeze(1))
        # # generated_spec = mel_spectrogram_transform(generated_wav)
        # # target_spec = mel_spectrogram_transform(wav.squeeze(1))
        
        # real_data = audio_codes.to(device)
        # fake_data = input_for_next_step.detach()
        real_output = discriminator(real_spec)
        fake_output = discriminator(fake_spec)
        loss_d_real = -torch.mean(real_output)
        loss_d_fake = torch.mean(fake_output)

        gradient_penalty = compute_gradient_penalty(discriminator, real_spec.data, fake_spec.data, device)
        lambda_gp = 5  # Lambda for scaling the gradient penalty (hyperparameter)
        loss_gp = lambda_gp * gradient_penalty
        loss_d = loss_d_fake + loss_d_real + loss_gp
        loss_d.backward()

        # Calculate and log the gradient norms
        total_norm_d, layer_norms_d = calculate_gradient_norm_layers(discriminator)
        tensorboard_writer.add_scalar('D_Gradient_Norms/D_Total', total_norm_d, epoch * len(train_loader) + i)
        for name, norm in layer_norms_d.items():
            tensorboard_writer.add_scalar(f'D_Gradient_Norms/D_Layer_{name}', norm, epoch * len(train_loader) + i)

        optimizer_d.step()

        total_d_steps += 1
        total_loss_d += loss_d.detach().item()
        avg_epoch_loss_d = total_loss_d / total_d_steps

        tensorboard_writer.add_scalar(f'Loss_Discriminator_lr={str(args.d_learning_rate)}', loss_d.item(), epoch * len(train_loader) + i)
        tensorboard_writer.add_scalar(f'Loss_Discriminator_lr={str(args.d_learning_rate)}/Real', loss_d_real.item(), epoch * len(train_loader) + i)
        tensorboard_writer.add_scalar(f'Loss_Discriminator_lr={str(args.d_learning_rate)}/Fake', loss_d_fake.item(), epoch * len(train_loader) + i)
        tensorboard_writer.add_scalar(f'Loss_Discriminator_lr={str(args.d_learning_rate)}/Gradient_Penalty', loss_gp.item(), epoch * len(train_loader) + i)

        if (i + 1) % N_CRITIC == 0:
            # Calculate perceptual loss. derive it by comparing the generated vs target mel spectrgorams 
            optimizer_g.zero_grad()
            generated_wav = audioCodeToWav(input_for_next_step, encodec_model, sample_rate = 24000)
            # generated_spec = mel_spectrogram_transform(generated_wav)
            # target_spec = mel_spectrogram_transform(wav.squeeze(1))
            # min_length = min(generated_spec.shape[-1], target_spec.shape[-1])
            # mel_mse_loss = ((generated_spec[:,:,:,:min_length] - target_spec[:,:,:,:min_length])**2).mean()
            # mel_mse_loss = mel_mse_loss
            wav = wav.squeeze(1)
            min_length = min(generated_wav.shape[-1], wav.shape[-1])
            mel_mse_loss = ((generated_wav[:,:,:min_length] - wav[:,:,:min_length])**2).mean()
            mel_mse_loss = 50 * mel_mse_loss

            avg_nll_loss = epoch_total_nll_loss / (total_timesteps)
            batch_nll_loss = batch_nll_loss / (batch_time_steps)

            fake_data = input_for_next_step
            fake_output = discriminator(fake_data)
            adversarial_loss_g = -torch.mean(fake_output)

            combined_loss_g = (2*batch_nll_loss) + adversarial_loss_g + mel_mse_loss
            combined_loss_g.backward()

            total_norm_g, layer_norms_g = calculate_gradient_norm_layers(pose_model)
            tensorboard_writer.add_scalar('G_Gradient_Norms/G_Total', total_norm_g, epoch * len(train_loader) + i)
            for name, norm in layer_norms_g.items():
                tensorboard_writer.add_scalar(f'G_Gradient_Norms/G_Layer_{name}', norm, epoch * len(train_loader) + i)

            optimizer_g.step()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            total_g_steps += 1
            total_loss_g += combined_loss_g.detach().item()
            avg_epoch_loss_g = total_loss_g / total_g_steps

            tensorboard_writer.add_scalar(f'Loss_Generator_lr={str(args.g_learning_rate)}/1_Combined_Loss', combined_loss_g.item(), epoch * len(train_loader) + i)
            tensorboard_writer.add_scalar(f'Loss_Generator_lr={str(args.g_learning_rate)}/2_Batch_NLL_Loss', 2*batch_nll_loss.item(), epoch * len(train_loader) + i)
            tensorboard_writer.add_scalar(f'Loss_Generator_lr={str(args.g_learning_rate)}/3_Time_MSE_Loss_Scaled', mel_mse_loss.item(), epoch * len(train_loader) + i)
            tensorboard_writer.add_scalar(f'Loss_Generator_lr={str(args.g_learning_rate)}/4_Adversarial_Loss', adversarial_loss_g.item(), epoch * len(train_loader) + i)
            tensorboard_writer.add_scalar(f'Loss_Generator_lr={str(args.g_learning_rate)}/5_Average_Epcoch_Loss', avg_epoch_loss_g, epoch * len(train_loader) + i)

            progress_bar.set_description(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Generator Loss: {avg_epoch_loss_g:.4f}, Discriminator Loss: {avg_epoch_loss_d:.4f}")
            print(f"\rEpoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Generator Loss: {combined_loss_g:.4f}, Discriminator Loss: {avg_epoch_loss_d:.4f}", end='')
        
        if torch.cuda.is_available():
                torch.cuda.empty_cache()
        gc.collect()
    return pose_model, discriminator, encodec_model, {'avg_epoch_loss_g': avg_epoch_loss_g, 'avg_epoch_loss_d': avg_epoch_loss_d}

def build_data_loaders(train_dataset, args):
    """
    Builds DataLoaders for training and validation datasets.
    Args:
    - train_dataset: The dataset to split.
    - args: Configuration parameters.
    Returns:
    - train_loader: DataLoader for training data.
    - val_loader: DataLoader for validation data.
    """
    train_ratio = args.train_ratio # Define the split ratio
    dataset_size = len(train_dataset)
    train_len = int(train_ratio * dataset_size)
    val_len = dataset_size - train_len

    # Randomly split the dataset
    batch_size = args.batch_size 
    random_seed = args.random_seed
    train_dataset, val_dataset = random_split(train_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(random_seed))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader

def generate_save_dirs(args):
    """
    Generates directories for logs and model checkpoints. Also moves the configuration file to the log directory.
    Args:
    - args: Configuration parameters.
    Returns:
    - log_dir: Directory for logging.
    - model_save_dir: Directory for model checkpoints.
    """
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join('./my_logs', 'run_' + current_time)
    config_dr = os.path.join(log_dir, 'config_file')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(config_dr, exist_ok=True)
    shutil.copy(args.config, config_dr)
    model_save_dir = os.path.join(log_dir,'model_saves_and_validation_samples')
    os.makedirs(model_save_dir, exist_ok=True)
    return log_dir, model_save_dir

def train():
    # The primary training function that sets up the model, data, and training components. It
    # loops through a specified number of epochs to train and validate the model, saving checkpoints
    # periodically. It also manages device allocation (GPU/CPU) and sets up TensorBoard for logging.

    args = parse_args()

    # assign GPU or CPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    mel_spectrogram_transform = T.MelSpectrogram(sample_rate=24000, n_fft=2048, n_mels=64).to(device)
    encodec_model, pose_model, discriminator, train_dataset = initialize_model_and_data(mel_spectrogram_transform, args, device)
    train_loader, val_loader = build_data_loaders(train_dataset, args)
    
    
    # Load the starting weights if provided ex. earlier checkpoint
    if args.starting_weights_path != None:
        weights = args.starting_weights_path
        pose_model.load_state_dict(torch.load(weights, map_location=device))

        
    # Prepare encodec model for training, first freeze all parameters, then unfreeze the final conv1d layer if args.freeze_encodec_decoder = False
    for param in encodec_model.parameters():
        param.requires_grad = False # freeze all parameters of the encoded model
    optimizer_g = torch.optim.Adam(pose_model.parameters(), lr= args.g_learning_rate)
    if not args.freeze_encodec_decoder: 
        for param in encodec_model.decoder.layers[-1].parameters(): 
            param.requires_grad = True
        generator_params = list(pose_model.parameters()) + list(encodec_model.decoder.layers[-1].parameters())
        optimizer_g = torch.optim.Adam(generator_params, lr=args.g_learning_rate)       

    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr= args.d_learning_rate)
    criterion_g = torch.nn.NLLLoss()

    log_dir, model_save_dir = generate_save_dirs(args)
    writer = SummaryWriter(log_dir=log_dir)

    val_epoch_interval = args.val_epoch_interval # 5
    num_epochs = args.num_epochs 
    for epoch in range(num_epochs):
        pose_model.train()
        discriminator.train()
        encodec_model.decoder.train()

        pose_model, discriminator, encodec_model, loss = train_one_epoch(pose_model, discriminator, encodec_model, mel_spectrogram_transform, train_loader, criterion_g, optimizer_g, optimizer_d, device, writer, epoch, args)
        avg_epoch_loss_g = loss['avg_epoch_loss_g']
        avg_epoch_loss_d = loss['avg_epoch_loss_d']
        
        # Compute average epoch loss
        writer.add_scalar('Average Generator Loss', avg_epoch_loss_g, epoch)
        writer.add_scalar('Average Discriminator Loss', avg_epoch_loss_d, epoch)
        print(f"\nEnd of Epoch [{epoch+1}/{num_epochs}], Average Generator Loss: {avg_epoch_loss_g:.4f}, Average Discriminator Loss: {avg_epoch_loss_d:.4f}")        
        
        if epoch % val_epoch_interval == 0:
            pose_model.eval()
            encodec_model.eval()
            avg_val_loss, generated_audio_codes, vid_paths = validation_step(pose_model, val_loader, criterion_g, device, writer, epoch, num_epochs, model_save_dir)

            # Save a model checkpoint and validation sample
            epoch_model_save_dir = os.path.join(model_save_dir, f"epoch_{epoch+1}")
            save_model(pose_model, epoch_model_save_dir, avg_val_loss, '', name='gen_3_sec_dnb_')
            save_model(discriminator, epoch_model_save_dir, avg_val_loss, '', name='disc_3_sec_dnb_')
            save_model(encodec_model, epoch_model_save_dir, avg_val_loss, '', name='encodec_3_sec_dnb_')
            save_valdiation_sample(generated_audio_codes[-1], encodec_model, vid_paths[-1], epoch, epoch_model_save_dir)
            gc.collect()
        else:
            print(f"\n Epoch [{epoch+1}/{num_epochs}]")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    writer.close()

if __name__ == "__main__":
    train()