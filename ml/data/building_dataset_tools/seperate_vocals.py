import os
import sys
import soundfile as sf
from spleeter.separator import Separator
from moviepy.editor import VideoFileClip, AudioFileClip

import numpy as np
import argparse
import librosa

# python /Users/azeez/Documents/pose_estimation/DanceToMusicApp/ml/data/building_dataset_tools/seperate_vocals.py --directory /Users/azeez/Documents/pose_estimation/DanceToMusicApp/ml/data/samples/5sec_expando_test
def extract_instrumental(data_dir, sr = 24000, mp_pose=None):
    directory = data_dir

    # Initialize Spleeter
    separator = Separator('spleeter:4stems')

    for root, dirs, files in os.walk(directory):
        for d in dirs:
            if 'spleeter' not in os.path.join(root, d):
                wav_path = os.path.join(root, d, f"{d[:-7]}.wav")
                extractDrumNBass(wav_path, separator, sr=sr)

def replace_audio_in_renders(directory):
    for root, dirs, files in os.walk(directory):
        for d in dirs:
            for f in files:
                if f.endswith('drum_and_bass.wav'):
                    drum_and_bass_path = os.path.join(root, f)
                    vid_debug_path = os.path.join(root, f"{f[:-18]}_with_audio.mp4")
                    vid_normalize_path = os.path.join(root, f"{f[:-18]}_normalized_with_audio.mp4")
                    replace_audio(drum_and_bass_path, vid_debug_path)
                    replace_audio(drum_and_bass_path, vid_normalize_path)

def replace_audio(audio_path, video_path):
    # Load the audio and video
    audio = AudioFileClip(audio_path)
    video = VideoFileClip(video_path)

    # Replace the audio in the video
    video = video.set_audio(audio)

    # Remove the current video with the same name, otherwise an error will occur (the video will save but when you play it it freezes)
    os.remove(video_path)

    # Save the video
    video.write_videofile(video_path, codec="libx264", audio_codec='aac', fps=24)

def extractInstrumental2Stems(wav_path, separator, sr = 24000):
    wav, sr = librosa.load(wav_path, sr=sr)
    # temp_audio_path = 'spleeter_input/temp_input_audio.wav'
    # sf.write(temp_audio_path, wav, sr)

    # Initialize Spleeter
    separator = Separator('spleeter:2stems')
    
    # Determine output directory based on wav_path
    output_dir = os.path.dirname(wav_path)
    base_name = os.path.splitext(os.path.basename(wav_path))[0]
    spleeter_output_dir = os.path.join(output_dir, f"{base_name}_spleeter_output")

    # Use Spleeter to separate the audio file
    separator.separate_to_file(wav_path, spleeter_output_dir)

    # Paths for the separated audio
    vocals_path = os.path.join(spleeter_output_dir, 'vocals.wav')
    instrumental_path = os.path.join(spleeter_output_dir, 'accompaniment.wav')

    return vocals_path, instrumental_path

def extractDrumNBass(wav_path, separator, sr = 24000):
    # Determine output directory based on wav_path
    output_dir = os.path.dirname(wav_path)
    spleeter_output_dir = os.path.join(output_dir, "spleeter_output")
    
    # Use Spleeter to separate the audio file
    separator.separate_to_file(wav_path, spleeter_output_dir)

    # Build string to help locate the output audio files 
    dir_cont = wav_path.split('/')[-2][:-7]
    # Construct paths for the separated audio
    output_dir = f'{spleeter_output_dir}/{dir_cont}'
    vocals_path = f'{output_dir}/vocals.wav'
    drums_path = f'{output_dir}/drums.wav'
    bass_path = f'{output_dir}/bass.wav'
    other_path = f'{output_dir}/other.wav'

    # Load the separated audio back into Python
    drums, _ = librosa.load(drums_path, sr=sr, mono=True)
    bass, _ = librosa.load(bass_path, sr=sr, mono=True)

    # Combine drums and bass tracks
    drum_and_bass = drums + bass

    # Save the combined drum and bass track
    # combined_path = f'{output_dir}/drum_and_bass.wav'

    combined_path = os.path.join(*wav_path.split('/')[:-1], f"{dir_cont}_drum_and_bass.wav")
    sf.write('/'+combined_path, drum_and_bass, sr)

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--directory', type=str, help='Input video directory path')
    parser.add_argument('--sample_rate', default=24000, type=int, help='Audio Sample Rate (default is 24000)')
    args = parser.parse_args()

    # Input video directory path from command-line argument
    # directory = args.directory
    # for root, dirs, files in os.walk(directory):
    #     for d in dirs:
    #         wav_path = os.path.join(root, d, f"{d[:-7]}.wav")
    #         extractDrumNBass(wav_path)
    extract_instrumental(args.directory, sr=args.sample_rate)
    replace_audio_in_renders(args.directory)


if __name__ == "__main__":
    main()
