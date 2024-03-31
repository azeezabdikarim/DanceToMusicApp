import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import librosa
import multiprocessing
from transformers import EncodecModel
from .beat_onset_detection.bod_music_processor import *

def audioCodeToWav(audio_code, encodec_model, sample_rate = 24000):
    """
    Logs the gradients of a model's parameters.
    Parameters:
    - audio_code: The audio code. A latent representation built by the encodec.encoder and decoded into a .wav by the encodec.decoder.
    - encodec_model: The model for encoding wav files into audio codes and decoding audio codes into wav files.
    - sample_rate: The sample rate of the wav file.
    Returns:
    - combined_wav: The decoded audio code as a wav file.
    """
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

def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """
    Computes the gradient penalty for WGAN-GP in order to inforce the Lipschitz constraint
    and stabilizes the training of the discriminator.

    Args:
    - D: The discriminator model.
    - real_samples: Tensor of real data samples.
    - fake_samples: Tensor of generated (fake) data samples.
    - device: The device (GPU/CPU) on which the tensors are.

    Returns:
    - gradient_penalty: The calculated gradient penalty, a scalar value that measures
      how the gradients deviate from the desired norm value of 1.
    """
    
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.size(0), 1, 1), device=device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size(), requires_grad=False, device=device)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def calculate_gradient_norms(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def calculate_gradient_norm_layers(model):
    layer_norms = {}
    total_norm = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            layer_norms[name] = param_norm.item()
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm, layer_norms


# Beat Scores Ref: https://github.com/L-YeZhu/Beats_Scores/blob/main/beats_scores.py
# def beat_detect(x, sr=24000):
# 	onsets = librosa.onset.onset_detect(x, sr=sr, wait=1, delta=0.2, pre_avg=1, post_avg=1, post_max=1, units='time')
# 	n = np.ceil( len(x) / sr)
# 	beats = [0] * int(n)
# 	for time in onsets:
# 		beats[int(np.trunc(time))] = 1
# 	return beats

# def beat_detect(batch_x, sr=24000):
#     all_beats = []
#     for x in batch_x:
#         if len(x.shape) > 1:
#             x = x.squeeze()  # Ensure x is one-dimensional
#         onsets = librosa.onset.onset_detect(y=x, sr=sr, wait=1, delta=0.2, pre_avg=1, post_avg=1, post_max=1, units='time')
#         n = np.ceil(len(x) / sr)
#         beats = [0] * int(n)
#         for time in onsets:
#             beats[int(np.trunc(time))] = 1
#         all_beats.append(beats)
#     return all_beats

# def compute_beat_scores(ref_wav, syn_wav):
#     gt_beats = beat_detect(ref_wav)
#     syn_beats = beat_detect(syn_wav)
#     score_cover, score_hit = beat_scores(gt_beats, syn_beats)
#     return score_cover, score_hit

def compute_beat_scores(model, gt_wav, gen_wav, device = 'cpu'):
    gt_stacked_specs = torch_fft_and_melscale(gt_wav.squeeze(0))
    gen_stacked_specs = torch_fft_and_melscale(gen_wav.squeeze(0))

    inference_gt = model.torch_infer(gt_stacked_specs, device, minibatch=4192)
    inference_gen = model.torch_infer(gen_stacked_specs, device, minibatch=4192)

    # Compute the MSE loss between the inferred beats and the ground truth beats
    criterion = nn.MSELoss()
    loss = criterion(inference_gen, inference_gt)

    return loss

def batch_compute_beat_scores(model, gt_wavs, gen_wavs, device = 'cpu'):
    losses = []
    model.to(device)
    gt_wavs = gt_wavs.to(device)
    gen_wavs = gen_wavs.to(device)
    for gt_wav, gen_wav in zip(gt_wavs, gen_wavs):
        loss = compute_beat_scores(model, gt_wav, gen_wav, device)
        losses.append(loss)
    return torch.mean(torch.stack(losses))


# def beat_scores(gt, syn):
# 	assert len(gt) == len(syn)
# 	total_beats = sum(gt)
# 	cover_beats = sum(syn)
# 	hit_beats = 0
# 	for i in range(len(gt)):
# 		if gt[i] == 1 and gt[i] == syn[i]:
# 			hit_beats += 1
# 	return cover_beats/total_beats, hit_beats/total_beats

# def _beat_detect_single(x, sr=24000):
#     """ Function to detect beats for a single audio sample. """
#     onsets = librosa.onset.onset_detect(y=x, sr=sr, wait=1, delta=0.2, pre_avg=1, post_avg=1, post_max=1, units='time')
#     n = np.ceil(len(x) / sr)
#     beats = [0] * int(n)
#     for time in onsets:
#         beats[int(np.trunc(time))] = 1
#     return beats

# def beat_detect(batch_x, sr=24000, num_processes=None):
#     """ Function to detect beats for a batch of audio samples. """
#     # Flatten the batch if it's multidimensional
#     batch_x = [x.squeeze() for x in batch_x]
#     # Create a pool of worker processes
#     with multiprocessing.Pool(processes=num_processes) as pool:
#         all_beats = pool.starmap(_beat_detect_single, [(x, sr) for x in batch_x])
#     return all_beats

# def compute_beat_scores(ref_wav, gen_wav):
#     gt_beats = beat_detect(ref_wav)
#     gen_beats = beat_detect(gen_wav)
#     scores = [beat_scores(gt, syn) for gt, syn in zip(gt_beats, gen_beats)]

#     batch_size = len(scores)  # Assuming scores is not empty

#     avg_cover_score = sum([score[0] for score in scores]) / batch_size
#     avg_hit_score = sum([score[1] for score in scores]) / batch_size
#     return avg_cover_score, avg_hit_score

