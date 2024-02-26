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
from models import Pose2AudioTransformer, AudioCodeDiscriminator, MelSpectrogramDiscriminator, Dance2MusicDiffusion
from utils import DanceToMusic
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from utils.loss_helpers import *
from utils.training_utils import *
import torchaudio.transforms as T
from torch.profiler import profile, record_function, ProfilerActivity

def initialize_model_and_data(args, device):
    """
    Initialize the models and set up the device.

    Args:
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
    pose_seq_len = train_dataset.data['poses'].shape[1]
    # pose_model = Pose2AudioTransformer(codebook_size, 
    #                                    args.src_pad_idx, 
    #                                    args.trg_pad_idx, 
    #                                    device = device, 
    #                                    num_layers = args.pose2audio_num_layers, 
    #                                    heads = args.pose2audio_num_heads, 
    #                                    embed_size = embed_size, 
    #                                    dropout = args.pose2audio_dropout, 
    #                                    input_size = input_size)
    # pose_model.to(device)

    latent_len = target_shape[0]*target_shape[1]
    ld_model = Dance2MusicDiffusion(c_in = 16, c_out = 16, pose_seq_len = pose_seq_len, num_labels = codebook_size)
    ld_model.to(device)

    # return encodec_model, pose_model, ld_model, train_dataset
    return encodec_model, ld_model, train_dataset


def validation_step(ld_model, val_loader, criterion, device, tensorboard_writer, epoch, num_epochs):
    """
    Executes validation steps for one epoch.
    """
    val_loss = 0
    ld_model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient computation
        for i, (audio_codes, pose, pose_mask, wav, wav_mask, wav_path, vid_path, _) in enumerate(val_loader):
            B, N, _, _ = pose.shape
            context = pose.view(B, N, -1).to(device)
            audio_codes = audio_codes.to(device)

            t = (1 - torch.rand(pose.size(0), device=device))
            noisey_latent, mask = ld_model.add_noise(audio_codes, t)
            noisey_latent = noisey_latent.to(device)
            denoised_latent = ld_model(noisey_latent, t, context)
            t_len = min(denoised_latent.shape[2], audio_codes.shape[1])
            loss = criterion(denoised_latent[:,:,:t_len,:], audio_codes[:,:t_len,:].long())

            val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        tensorboard_writer.add_scalar('Validation Loss', avg_val_loss, epoch)
        print(f"\n Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

    return avg_val_loss, torch.argmax(denoised_latent, dim = 1), vid_path
    
def train_one_epoch(ld_model, train_loader, criterion, optimizer, device, tensorboard_writer, epoch, max_steps, args):
    total_loss = 0
    num_epochs = args.num_epochs
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (audio_codes, pose, pose_mask, wav, wav_mask, wav_path, vid_path, _) in progress_bar:
        optimizer.zero_grad()

        # src = pose.to(device)
        # enc_mask = pose_model.make_src_mask(src)

        B, N, _, _ = pose.shape
        # enc_context = pose_model.encoder(src.view(B, N, -1), enc_mask)
        context = pose.view(B, N, -1).to(device)
        audio_codes = audio_codes.to(device)

        t = (1 - torch.rand(pose.size(0), device = device))
        noisey_latent, mask = ld_model.add_noise(audio_codes, t)
        noisey_latent = noisey_latent.to(device)
        denoised_latent = ld_model(noisey_latent, t, context)
        t_len = min(denoised_latent.shape[2], audio_codes.shape[1])
        loss = criterion(denoised_latent[:,:,:t_len,:], audio_codes[:,:t_len,:].long())
        total_loss += loss.detach().item()
        loss.backward()
        optimizer.step()

        tensorboard_writer.add_scalar('Categorical Loss', loss.item(), epoch * len(train_loader) + i)
        progress_bar.set_description(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss:.4f}")
    
    avg_epoch_loss = total_loss / len(train_loader)
    return ld_model, {'avg_epoch_loss': avg_epoch_loss}

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
    device = torch.device("mps")

    encodec_model, ld_model, train_dataset = initialize_model_and_data(args, device)
    train_loader, val_loader = build_data_loaders(train_dataset, args)
    
    
    # Load the starting weights if provided ex. earlier checkpoint
    # if args.starting_weights_path != None:
    #     weights = args.starting_weights_path
    #     pose_model.load_state_dict(torch.load(weights, map_location=device))

    # mel_spectrogram_transform = T.MelSpectrogram(sample_rate=24000, n_fft=2048, n_mels=64).to(device)
        
    # Prepare encodec model for training, first freeze all parameters, then unfreeze the final conv1d layer if args.freeze_encodec_decoder = False
    for param in encodec_model.parameters():
        param.requires_grad = False # freeze all parameters of the encoded model

    # generator_params = list(pose_model.parameters()) + list(ld_model.parameters())
    generator_params = list(ld_model.parameters())
    optimizer = torch.optim.Adam(generator_params, lr=args.g_learning_rate)       
    
    criterion = torch.nn.NLLLoss()

    log_dir, model_save_dir = generate_save_dirs(args)
    writer = SummaryWriter(log_dir=log_dir)

    val_epoch_interval = args.val_epoch_interval # 5
    num_epochs = args.num_epochs
    max_steps = 100
    for epoch in range(num_epochs):
        # pose_model.train()

        ld_model, loss = train_one_epoch(ld_model, train_loader, criterion, optimizer, device, writer, epoch, max_steps, args)
        avg_epoch_loss = loss['avg_epoch_loss']
        
        # Compute average epoch loss
        writer.add_scalar('Average Epoch Loss', avg_epoch_loss, epoch)
        print(f"\nEnd of Epoch [{epoch+1}/{num_epochs}], Average Generator Loss: {avg_epoch_loss:.4f}")        
        
        if epoch % val_epoch_interval == 0:
            ld_model.eval()
            encodec_model.eval()
            avg_val_loss, generated_audio_codes, vid_paths = validation_step(ld_model, val_loader, criterion, device, writer, epoch, num_epochs)

            # Save a model checkpoint and validation sample
            epoch_model_save_dir = os.path.join(model_save_dir, f"epoch_{epoch+1}")
            save_model(ld_model, epoch_model_save_dir, avg_val_loss, '', name='ld_3_sec_dnb_')
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