import os
import numpy as np
import torch
from tqdm import tqdm
import pickle as pkl
import time

import librosa
import torch
import torchaudio
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from encodec.utils import convert_audio
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
from torch.optim import Adam
from transformers import EncodecModel
from models import Pose2AudioTransformer
from torch.utils.tensorboard import SummaryWriter

# load datasets of paired sequences and music
class DanceToMusic(Dataset):
    def __init__(self, directory, encoder = None, sample_rate=24000, device = torch.device("cpu")):
        self.device = device
        self.raw_data = self._load_data(directory, sample_rate)
        self.data = self._buildData(self.raw_data)
        self.encoder = encoder
        if encoder is not None:
            self.data = self._encodeAudio(encoder, self.data)

    def __len__(self):
        return len(self.data['poses'])

    def __getitem__(self, idx):
        pose = self.data['poses'][idx]
        wav = self.data['wavs'][idx]
        wav_path = self.data['wav_paths'][idx]
        sample_rate = self.data['sample_rate'][idx]
        pose_mask = self.data['pose_padding_mask'][idx]
        wav_mask = self.data['audio_padding_mask'][idx]

        if self.encoder is None:
            return [pose, pose_mask, wav, wav_mask, wav_path, sample_rate]
        
        audio_codes = self.data['audio_codes'][idx]
        return [audio_codes, pose, pose_mask, wav, wav_mask, wav_path, sample_rate]

    def _encodeAudio(self, encoder, data):
        audio_codes = []
        for i, wav in enumerate(data['wavs']):
            wav = wav.unsqueeze(0)
            encoding = encoder.encode(wav, data['audio_padding_mask'][i].unsqueeze(0))
            audio_codes.append(encoding['audio_codes'].squeeze(0))
        data['audio_codes'] = audio_codes
        return data

    def _buildData(self, raw_data):
        data = {}
        data['wav_paths'] = raw_data['wav_paths']
        data['sample_rate'] =  raw_data['sample_rate']
        
        # Handle poses
        poses = [torch.tensor(p) for p in raw_data['poses']]
        max_pos_seq_len = max(p.shape[0] for p in poses)
        
        # Handle audio
        wavs = [torch.tensor(w) for w in raw_data['wavs']]
        max_audio_len = max(w.shape[0] for w in wavs)
        
        num_samples = len(poses)
        
        # Padding for poses
        padded_poses = torch.zeros(num_samples, max_pos_seq_len, 25, 2)
        pose_padding_mask = torch.zeros(num_samples, max_pos_seq_len, dtype=torch.bool)
        
        # Padding for audio
        padded_audio = torch.zeros(num_samples, max_audio_len)
        audio_padding_mask = torch.zeros(num_samples, max_audio_len, dtype=torch.bool)

        # Fill the tensors for poses
        for i, seq in enumerate(poses):
            n = seq.size(0)
            padded_poses[i, :n, :, :] = seq
            pose_padding_mask[i, :n] = 1  # Set mask to 1 for non-padded values
        
        # Fill the tensors for audio
        for i, audio in enumerate(wavs):
            n = audio.size(0)
            padded_audio[i, :n] = audio
            audio_padding_mask[i, :n] = 1  # Set mask to 1 for non-padded values
        
        data['poses'] = padded_poses.to(self.device)
        data['pose_padding_mask'] = pose_padding_mask.to(self.device)
        data['wavs'] = padded_audio.unsqueeze(1).to(self.device)
        data['audio_padding_mask'] = audio_padding_mask.to(self.device)
        
        return data


    def _buildPoses(self, directory):
        poses = []
        for root, dirs, files in os.walk(directory):
            for f in files:
                if '.npy' in f:
                    pose = np.load(os.path.join(root, f))
                    poses.append(pose)
        return np.array(poses)

    def _load_data(self, directory, sr):
        poses = []
        wavs = []
        wav_paths = []
        sample_rate = []
        for root, dirs, files in os.walk(directory):
            for d in dirs:
                if (len(d.split('_')) == 2) and d[0] == 'I':
                    s = root.split('/')[-2]
                    pose_dir_path = os.path.join(root, d,'data')
                    wav_path = os.path.join(root, d, f"{d}.wav")

                    poses.append(self._buildPoses(pose_dir_path))
                    wav, sr = librosa.load(wav_path, sr=sr)
                    sample_rate.append(sr)
                    wav_paths.append(wav_path)
                    wavs.append(wav)

        ret = {
            "poses": poses,
            "wavs": wavs,
            "wav_paths": wav_paths,
            "sample_rate": sample_rate
        }
        return ret

class PoseWithPrecomputedMusicEncodins(Dataset):
    def __init__(self, original_dataset, encoder):
        self.original_dataset = original_dataset
        self.encoder = encoder
        self.precomputed_targets = self._precompute_targets()

    def _precompute_targets(self):
        precomputed_targets = []
        for i in range(len(self.original_dataset)):
            _, _, _, wav, wav_mask, _, _ = self.original_dataset[i]
            wav = wav.unsqueeze(0)
            encoding = self.encoder.encode(wav, wav_mask.unsqueeze(0))
            audio_codes = encoding["audio_codes"].squeeze(0)
            precomputed_targets.append(audio_codes)
        return precomputed_targets

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        audio_codes = self.precomputed_targets[idx]
        pose, pose_mask, _, _, _, _, _ = self.original_dataset[idx]
        return [audio_codes, pose, pose_mask]



if __name__ == "__main__":

    # assign GPU or CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    device = torch.device("cpu")

    
    

    model_id = "facebook/encodec_24khz"
    encodec_model = EncodecModel.from_pretrained(model_id)
    codebook_size = encodec_model.quantizer.codebook_size
    # processor = AutoProcessor.from_pretrained(model_id)
    
    sample_rate = 24000
    batch_size = 8
    data_dir = "/Users/azeez/Documents/pose_estimation/Learning2Dance/l2d_train"
    original_dataset = DanceToMusic(data_dir, encoder=encodec_model, sample_rate=sample_rate, device='cpu')
    train_dataset = PrecomputedDanceToMusic(original_dataset, encodec_model)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    src_pad_idx = 0
    trg_pad_idx = 0

    pose_model = Pose2AudioTransformer(codebook_size, src_pad_idx, trg_pad_idx, device=device, num_layers=4, heads = 5)
    pose_model.to(device)


    learning_rate = 1e-5
    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam(pose_model.parameters(), lr=learning_rate)
    
    writer = SummaryWriter(log_dir='./my_logs')

    num_epochs = 3000
    for epoch in range(num_epochs):
        pose_model.train()
        epoch_loss = 0
        for i, (audio_codes, pose, pose_mask) in tqdm(enumerate(train_loader), total=len(train_loader)):
            # Forward pass
            pose = pose.to(device)
            pose_mask = pose_mask.to(device)
            target = audio_codes.view(audio_codes.shape[0], 1, -1).to(device)
    for epoch in range(num_epochs):
        pose_model.train()
        epoch_loss = 0
        for i, (audio_codes, pose, pose_mask, wav, wav_mask, _, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
            # Forward pass
            wav = wav.unsqueeze(1)
            pose = pose
            pose_mask = pose_mask
            # encoding = encodec_model.encode(wav, wav_mask)
            # audio_codes = encoding["audio_codes"]
            target = audio_codes.view(audio_codes.shape[0], 1, -1)

            output = pose_model(pose.to(device), target.squeeze(1).to(device), src_mask=pose_mask.to(device))
            target_one_hot = F.one_hot(target, num_classes=codebook_size).squeeze(1).float()

            # Compute loss and backpropagate
            loss = criterion(output.cpu(), target_one_hot)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()  # Accumulate loss
            
        # Print epoch statistics
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"\n Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_epoch_loss:.4f}")

        writer.add_scalar('Average Loss', avg_epoch_loss, epoch)

    writer.close()
