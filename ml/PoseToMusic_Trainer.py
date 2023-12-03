import os
import numpy as np
import torch
from tqdm import tqdm
import pickle as pkl
import time

import librosa
import torch
import torchaudio
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
from torch.optim import Adam
from transformers import AutoProcessor, EncodecModel, EncodecFeatureExtractor
from models import Pose2AudioTransformer
from utils import DanceToMusic
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split


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


if __name__ == "__main__":

    # assign GPU or CPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    device = torch.device("cpu")


    model_id = "facebook/encodec_24khz"
    encodec_model = EncodecModel.from_pretrained(model_id)
    codebook_size = encodec_model.quantizer.codebook_size
    encodec_model.to(device)
    # processor = AutoProcessor.from_pretrained(model_id)
    
    sample_rate = 24000
    batch_size = 60

    # data_dir = '/Users/azeez/Documents/pose_estimation/DanceToMusic/data/samples/5sec_min_data'
    # data_dir = "/Users/azeez/Documents/pose_estimation/DanceToMusic/data/min_training_data"
    # data_dir = '/home/azeez/azeez_exd/misc/DanceToMusic/data/samples'
    data_dir = '/Users/azeez/Documents/pose_estimation/DanceToMusicApp/ml/data/samples/5sec_expando_dnb_min_training_data'
    train_dataset = DanceToMusic(data_dir, encoder = encodec_model, sample_rate = sample_rate, device=device, dnb = True)
    embed_size = train_dataset.data['poses'].shape[2] * train_dataset.data['poses'].shape[3]


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
    
    # weights = '/Users/azeez/Documents/pose_estimation/DanceToMusic/weights/best_model_0.0152.pt'
    # pose_model.load_state_dict(torch.load(weights, map_location=device))

    learning_rate = 1e-3
    # criterion = CrossEntropyLoss()
    criterion = torch.nn.NLLLoss()
    # criterion = MSELoss()
    optimizer = torch.optim.Adam(pose_model.parameters(), lr=learning_rate)

    # Set up for tracking the best model
    # weights_dir = '/home/azeez/azeez_exd/misc/DanceToMusic/weights'
    weights_dir = '/Users/azeez/Documents/pose_estimation/DanceToMusicApp/ml/model_weights'
    best_loss = float('inf')  # Initialize with a high value
    last_saved_model = ''

    writer = SummaryWriter(log_dir='./my_logs')

    accumulation_steps = 1  # Number of mini-batches for gradient accumulation
    samples_processed = 0  # Keep track of samples processed so far within accumulation steps
    accumulated_loss = 0  # Initialize accumulated loss

    teacher_forcing_ratio = 0.5 # 50% of the time we will use teacher forcing
    val_epoch_interval = 5
    num_epochs = 30000
    for epoch in range(num_epochs):
        pose_model.train()
        epoch_loss = 0  # Initialize epoch_loss
        timesteps = 0

        for i, (audio_codes, pose, pose_mask, wav, wav_mask, _, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
            optimizer.zero_grad()  # Clear gradients
            
            # Forward pass
            wav = wav.unsqueeze(1)
            pose = pose
            pose_mask = pose_mask
            target = audio_codes.to(device)
            target_for_loss = target[:, :, 1:]
            
            input_for_next_step = target[:, :, 0]
            outputs = []
            batch_loss = 0
            
            src = pose.to(device)
            enc_mask = pose_model.make_src_mask(src)
            trg_mask = pose_model.make_trg_mask(input_for_next_step.to(device))
            
            B, N, _, _ = pose.shape
            enc_context = pose_model.encoder(src.view(B, N, -1), enc_mask)
            
            for t in range(1, target.shape[2]):
                output_softmax, output_argmax, offset, _ = pose_model.decoder(input_for_next_step.to(device), enc_context, enc_mask, trg_mask)
                
                log_softmax_output = torch.log(output_softmax)
                log_softmax_output_reshape = log_softmax_output.view(-1, log_softmax_output.shape[2])
                reshaped_target = target[:, :, t].reshape(-1).long()
                time_step_loss = criterion(log_softmax_output_reshape, reshaped_target)
                batch_loss += time_step_loss
                
                use_teacher_forcing = np.random.random() < teacher_forcing_ratio
                if use_teacher_forcing:
                    input_for_next_step = target[:, :, t]
                else:
                    input_for_next_step = output_softmax.argmax(dim=2)
                
                outputs.append(output_softmax)
                timesteps += 1
            
            # Backpropagate the gradients
            batch_loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Update epoch loss
            epoch_loss += batch_loss.item()
            
        avg_epoch_loss = epoch_loss / (timesteps)   # Compute average epoch loss

    
        # Check if this epoch resulted in a better model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            last_saved_model = save_model(pose_model, weights_dir, best_loss, last_saved_model, name='5_sec_transformer_')

        if epoch % val_epoch_interval == 0:
            pose_model.eval()
            val_loss = 0
            with torch.no_grad():
                for audio_codes, pose, pose_mask, wav, wav_mask, _, _ in val_loader:
                    # Forward pass
                    wav = wav.unsqueeze(1)
                    target = audio_codes.to(device)
                    target_for_loss = target[:, :, 1:]

                    output_softmax, output_argmax, offset, _ = pose_model(pose.to(device), target.squeeze(1).to(device), src_mask=pose_mask.to(device))

                    output_softmax = output_softmax[:, offset:, :]
                    reshaped_output_softmax = output_softmax.reshape(-1, output_softmax.shape[2])
                    reshaped_target = target_for_loss.reshape(-1).long()
                    log_softmax_output = torch.log(output_softmax)  
                    log_softmax_output_reshape = log_softmax_output.view(-1, log_softmax_output.shape[2])  

                    computed_loss = criterion(log_softmax_output_reshape, reshaped_target)
                    val_loss += computed_loss.item()
                    
            # Compute average validation loss
            avg_val_loss = val_loss / len(val_loader)
            writer.add_scalar('Validation Loss', avg_val_loss, epoch)
            print(f"\n Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_epoch_loss:.4f}. Validation Loss: {avg_val_loss:.4f}")
        else:
            print(f"\n Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_epoch_loss:.4f}")
        writer.add_scalar('Average Loss', avg_epoch_loss, epoch)
    writer.close()
