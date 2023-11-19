import os
import numpy as np
import torch
import pickle
import time

import torch
from torch.utils.data import DataLoader, Dataset
from models import Pose2AudioTransformer
from transformers import EncodecModel
from utils import DanceToMusic
from datetime import datetime
from torch.optim import Adam

# assign GPU or CPU
# if torch.backends.mps.is_available():
    # device = torch.device("mps")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
# device = torch.device("cpu")

model_id = "facebook/encodec_24khz"
encodec_model = EncodecModel.from_pretrained(model_id)
codebook_size = encodec_model.quantizer.codebook_size
encodec_model.to(device)
sample_rate = 24000

data_dir = "DanceToMusic/data/samples/5sec_min_data"
dataset = DanceToMusic(data_dir, encoder = encodec_model, sample_rate = sample_rate, device=device)

src_pad_idx = 0
trg_pad_idx = 0
learned_weights = '/Users/azeez/Documents/pose_estimation/DanceToMusic/weights/5_sec_best_model_weights_loss_6.733452348148122.pth' 
embed_size = dataset.data['poses'].shape[2] * dataset.data['poses'].shape[3]
pose_model = Pose2AudioTransformer(codebook_size, src_pad_idx, trg_pad_idx, device=device, num_layers=2, heads = 2, embed_size=embed_size, dropout=0.1)
pose_model.load_state_dict(torch.load(learned_weights, map_location=device))
pose_model.to(device)

audio_codes, pose, pose_mask, wav, wav_mask, _, _ = dataset[0]
output = pose_model.generate(pose.unsqueeze(0).to(device), pose_mask.to(device), max_length = audio_codes.size(1), temperature=2)

print(output[0].shape, audio_codes[0].shape)