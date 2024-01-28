import os
import numpy as np
import torch
import sys

current_directory = os.getcwd()
models_dir = os.path.join(current_directory, '..')
models_dir2 = os.path.join(current_directory, 'DanceToMusicApp/webapp/processing/')
sys.path.append(models_dir)

import torch
from ml.models import Pose2AudioTransformer
from transformers import EncodecModel
from .utils import *

from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
import numpy as np
from scipy.io import wavfile

def audioCodeToWav(audio_code, encodec_model, sample_rate = 24000, device='cpu'):
    audio_code = audio_code.reshape(1,1,2,int(audio_code.size(2)))
    audio_code = audio_code.to(device)
    audio_scale = [None]
    wav = encodec_model.decode(audio_code, audio_scale)
    return wav

def generateAudio(vid_path, pose_path, max_pos_seq_len = 120):
    # assign GPU or CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    # device = torch.device("cpu")
    
    config = "processing/model/model_weights/wgan.txt"
    pm_weights = 'processing/model/model_weights/gen_3_sec_dnb__loss_1.141.pt'
    enc_weights = 'processing/model/model_weights/encodec_3_sec_dnb__loss_1.141.pt'

    args = parse_args(config)

    model_id = args.encodec_model_id
    encodec_model = EncodecModel.from_pretrained(model_id)
    encodec_model.load_state_dict(torch.load(enc_weights, map_location=device))
    encodec_model.to(device)

    codebook_size = encodec_model.quantizer.codebook_size
    sample_rate = args.sample_rate
    src_pad_idx = args.src_pad_idx
    trg_pad_idx = args.trg_pad_idx
    embed_size = 96
    pose_model = Pose2AudioTransformer(codebook_size, src_pad_idx, trg_pad_idx, 
                                       device=device, 
                                       num_layers=args.pose2audio_num_layers, 
                                       heads = args.pose2audio_num_heads, 
                                       embed_size=embed_size, 
                                       dropout = args.dropout)
    pose_model.load_state_dict(torch.load(pm_weights, map_location=device))
    pose_model.to(device)

    # Specify the path to save the output video and the temporary audio
    generated_output_video = vid_path.replace('.mp4','_generated_audio.mp4')
    temp_audio_path = vid_path.replace('.mp4','_generated_audio.wav')

    pose = np.load(pose_path)
    pose_length = min(len(pose), max_pos_seq_len)  # Ensuring we do not exceed 120 frames

    # Convert pose to a torch tensor
    pose = torch.tensor(pose[:pose_length], device=device, dtype=torch.float)

    # Create a mask for the valid positions in the pose tensor
    # The mask is True (1) where we have pose data, and False (0) elsewhere
    pose_mask = torch.arange(max_pos_seq_len, device=device) >= pose_length

    # If padding is required, pad both pose tensor and mask to the maximum sequence length
    if pose_length < max_pos_seq_len:
        padding_size = max_pos_seq_len - pose_length
        padding = torch.zeros((padding_size, *pose.shape[1:]), device=device, dtype=torch.float)
        pose = torch.cat([pose, padding], dim=0)
    
    # Pad mask (it is already the correct size, but we need to ensure it is a 2D tensor)
    pose_mask = pose_mask.unsqueeze(0).expand(1, max_pos_seq_len)  # Adds an extra dimension to make it 2D
    
    print("pose shape: ", pose.shape, "pose mask shape: ", pose_mask.shape, "max_pos_seq_len: ", max_pos_seq_len)
    output = pose_model.generate(pose.unsqueeze(0).to(device), pose_mask.to(device), max_length = 376, temperature = 1)
    wav = audioCodeToWav(output.unsqueeze(0), encodec_model, sample_rate = 24000, device=device)['audio_values']

    wav_np = wav[0].detach().cpu().numpy()
    max_val = np.max(np.abs(wav_np))
    normalized_wav = wav_np / max_val
    scaled_wav = np.int16(normalized_wav * 32767)
    wavfile.write(filename=temp_audio_path, rate=24000, data=scaled_wav.T)

    # Now create the video clip with the new audio
    video_clip = VideoFileClip(vid_path)
    new_audio_clip = AudioFileClip(temp_audio_path)

    new_audio_clip = CompositeAudioClip([new_audio_clip])
    video_clip.audio = new_audio_clip
    video_clip.write_videofile(generated_output_video)

    # Close the clips to release their resources
    video_clip.close()
    new_audio_clip.close()

    return generated_output_video