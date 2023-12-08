import os
import librosa
import numpy as np
import torch
import argparse
from transformers import EncodecModel

def parse_args():
    parser = argparse.ArgumentParser(description='Download youtube videos from a txt file and organize files in dataset style.')
    parser.add_argument("--data_dir", required=True,
                        help="path to txt file with url's of videos")
    parser.add_argument('--dnb', required=False, type=bool, default=False, 
                        help='Copy over drum and bass tracks rather than the original audio')

    args = parser.parse_args()
    return args

def buildEnCodecModel():
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
    encodec_model.to(device)

    return encodec_model, device

def audioPaddingMask(wavs):
    num_samples = len(wavs)
    max_audio_len = max(w.shape[0] for w in wavs)

    # Padding for audio
    padded_audio = torch.zeros(num_samples, max_audio_len)
    audio_padding_mask = torch.zeros(num_samples, max_audio_len, dtype=torch.bool)
    
    # Fill the tensors for audio
    for i, audio in enumerate(wavs):
        n = audio.size
        padded_audio[i, :n] = torch.Tensor(audio)
        audio_padding_mask[i, :n] = 1  # Set mask to 1 for non-padded values

    return padded_audio, audio_padding_mask

def saveAudioCodes(w_path, audio_code_with_start_token):
    audo_code_path = w_path.replace('.wav', '_audio_code.npy')
    np.save(audo_code_path, audio_code_with_start_token)

def preBuildEncodings(input_dir, dnb = False):
    model, device = buildEnCodecModel()
    sr = 24000

    wav_paths = []
    wavs = []
    for root, dirs, files in os.walk(input_dir):
        for d in dirs:
            if 'error' not in d and 'spleeter' not in os.path.join(root,d):
                wav_path = os.path.join(root, d, f"{d[:-7]}.wav")
                if dnb:
                    wav_path = os.path.join(root, d, f"{d[:-7]}_drum_and_bass.wav")

                wav, _ = librosa.load(wav_path, sr=sr)
                wav_paths.append(wav_path)
                wavs.append(wav)

    padded_wavs, audio_padding_masks = audioPaddingMask(wavs)
    audio_codes = []
    for i, w_path in enumerate(wav_paths):
        padded_audio = padded_wavs[i].to(device)
        padding_mask = audio_padding_masks[i].to(device)
        encoding = model.encode(padded_audio.unsqueeze(0).unsqueeze(0), padding_mask.unsqueeze(0))
        one_audio_code = encoding['audio_codes'].view(1,-1,2)
        one_audio_code = one_audio_code.squeeze(0)
        audio_code_with_start_token = torch.cat((torch.zeros(1,2).to(device), one_audio_code), dim=0)
        saveAudioCodes(w_path, audio_code_with_start_token)

def main():
    args = parse_args()
    data_dir = args.data_dir
    dnb = args.dnb

    preBuildEncodings(data_dir, dnb)

if __name__ == "__main__":
    main()