import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import librosa
from transformers import EncodecModel

def audioCodeToWav(audio_code, encodec_model, sample_rate = 24000):
    batch_size = audio_code.shape[0]
    audio_code = audio_code.reshape(batch_size,1,2,int(audio_code.shape[1]))

    # Check if the devices of audio_code and encodec_model.decoder are the same
    if encodec_model.device != audio_code.device:
        raise ValueError("The device of encodec_model.decoder and audio_code must be the same.")
    device = audio_code.device

    decoded_wavs = []
    # Iterate through each frame in audio_code and decode them individually
    for i in range(audio_code.size(0)):
        single_audio_code = audio_code[i:i+1]  # Extracting a single frame
        single_wav = encodec_model.decode(single_audio_code.int(), [None])
        decoded_wavs.append(single_wav[0])

    # Concatenate the decoded audio samples into a single tensor
    # Use torch.stack if each wav sample has more than one dimension, else use torch.cat
    if decoded_wavs[0].ndim > 1:
        combined_wav = torch.stack(decoded_wavs, dim=0)
    else:
        combined_wav = torch.cat(decoded_wavs, dim=0)

    return combined_wav.squeeze(1).to(device)

def log_gradients(model, step, writer, tag_prefix='Gradients'):
    """
    Logs the gradients of a model's parameters.
    
    Parameters:
    - model: The PyTorch model.
    - step: The current step or epoch number.
    - writer: Tensorboard SummaryWriter instance.
    - tag_prefix: The prefix for the Tensorboard tag.
    """
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            writer.add_histogram(f'{tag_prefix}/{name}', parameter.grad, step)
