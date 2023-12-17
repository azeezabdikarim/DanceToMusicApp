import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import librosa
from transformers import EncodecModel

def audioCodeToWav(audio_code, encodec_model, sample_rate = 24000, device=None):
    # if device == None:
    #     device = encodec_model.device
    audio_code = audio_code.reshape(1,1,2,int(audio_code.shape[1]))
    audio_code = audio_code.to('cpu')
    audio_scale = [None]
    wav = encodec_model.decode(audio_code.int(), audio_scale)
    return wav[0].to(device)

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
