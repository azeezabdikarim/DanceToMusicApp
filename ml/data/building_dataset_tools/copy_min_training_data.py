import os
import argparse 
import numpy as np
import librosa  
import soundfile as sf

# python /Users/azeez/Documents/pose_estimation/DanceToMusic/data/building_tools/copy_min_training_data.py --data_dir=/Users/azeez/Documents/pose_estimation/DanceToMusic/data/samples/5sec_samples --output_dir=/Users/azeez/Documents/pose_estimation/DanceToMusic/data/samples/5sec_min_data
def parse_args():
    parser = argparse.ArgumentParser(description='Download youtube videos from a txt file and organize files in dataset style.')
    parser.add_argument("--data_dir", required=True,
                        help="path to txt file with url's of videos")
    parser.add_argument('--output_dir', required=True,
                        help='Output path to create the dataset tree structure')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    output_path = args.output_dir  # Fixed variable name to match arg parser
    data_dir = args.data_dir
    sr = 24000  # Sample rate for librosa, you can change it based on your needs

    for root, dirs, files in os.walk(data_dir):
        for d in dirs:
            if 'error' not in d and 'spleeter' not in os.path.join(root,d):
                pose_path = os.path.join(root, d, f"{root.split('/')[-1]}_{d[:-7]}_3D_landmarks.npy")
                wav_path = os.path.join(root, d, f"{d[:-7]}.wav")
                if os.path.exists(wav_path.replace('.wav', '_audio_code.npy')):
                    audio_code_path = wav_path.replace('.wav', '_audio_code.npy')
                else:
                    audio_code_path = wav_path.replace('.wav', '_drum_and_bass_audio_code.npy')

                sample_poses = np.load(pose_path)
                wav, _ = librosa.load(wav_path, sr=sr)
                audio_code = np.load(audio_code_path)

                sample_output_dir = os.path.join(output_path, d)
                pose_output_path = os.path.join(sample_output_dir, f"{sample_output_dir.split('/')[-2]}_{d[:-7]}_3D_landmarks.npy")
                wav_output_path = os.path.join(sample_output_dir, f"{d[:-7]}.wav")
                audio_code_output_path = wav_output_path.replace('.wav', '_audio_code.npy')

                # Create the output directory if it doesn't exist
                if not os.path.exists(sample_output_dir):
                    os.makedirs(sample_output_dir)

                # Save the pose and wav files
                np.save(pose_output_path, sample_poses)
                sf.write(wav_output_path, wav, sr)
                np.save(audio_code_output_path, audio_code)
                # librosa.output.write_wav(wav_output_path, wav, sr)

if __name__ == "__main__":
    main()
