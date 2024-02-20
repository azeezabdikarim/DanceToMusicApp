from download_database_split_seq_pytube import *
from standardize_fps_sr_youtube import *
from pose_extraction_media_pipe import *
from copy_min_training_data import *
from seperate_vocals import *
from build_complete_dataset import *
from build_audio_encodings import *
import csv
import os
import time


# python /Users/azeez/Documents/pose_estimation/DanceToMusicApp/ml/data/building_dataset_tools/build_dnb_dataset.py --output_path /Users/azeez/Documents/pose_estimation/DanceToMusicApp/ml/data/samples/5sec_expando_test --input_csv /Users/azeez/Documents/pose_estimation/DanceToMusicApp/ml/data/youtube_links/youtube_links_test.csv  --max_seq_len 5 --fps 24


def save_min_data_for_training(data_dir, min_output_dir, dnb = False):
    output_path = min_output_dir
    sr = 24000  # Sample rate for librosa, you can change it based on your needs

    os.makedirs(output_path, exist_ok=True)

    for root, dirs, files in os.walk(data_dir):
        for d in dirs:
            if 'error' not in d and 'spleeter' not in os.path.join(root,d):
                pose_path = os.path.join(root, d, f"{root.split('/')[-1]}_{d[:-7]}_3D_landmarks.npy")
                wav_path = os.path.join(root, d, f"{d[:-7]}_drum_and_bass.wav")

                sample_poses = np.load(pose_path)
                wav, _ = librosa.load(wav_path, sr=sr)

                sample_output_dir = os.path.join(output_path, d)
                pose_output_path = os.path.join(sample_output_dir, f"{root.split('/')[-1]}_{d[:-7]}_3D_landmarks.npy")
                if dnb:
                    wav_output_path = os.path.join(sample_output_dir, f"{d[:-7]}_drum_and_bass.wav")
                else:
                    wav_output_path = os.path.join(sample_output_dir, f"{d[:-7]}.wav")

                # Create the output directory if it doesn't exist
                os.makedirs(sample_output_dir, exist_ok=True)

                # Save the pose and wav files
                np.save(pose_output_path, sample_poses)
                sf.write(wav_output_path, wav, sr)

def main():
    start = time.time()
    # Parse common arguments for both scripts
    parser = argparse.ArgumentParser(description='Download and standardize YouTube videos.')
    parser.add_argument("--input_csv", required=True, nargs='+', help='path to txt file with URL\'s of videos')
    parser.add_argument('--output_path', required=True, help='Output path for downloaded and standardized videos')
    parser.add_argument('--max_seq_len', default=10, type=int, help='Maximum sequence length for each clip (default is 10)')
    parser.add_argument('--fps', default=24, type=int, help='Frames per second (default is 30)')
    parser.add_argument('--audio_sr', default=24000, type=int, help='Audio sample rate (default is 44100)')
    
    args = parser.parse_args()

    # Step 1: Download and clip videos
    print("Step 1: Downloading and clipping videos...")
    # download_and_clip_videos(args.input_csv, args.output_path, args.max_seq_len)

    # Step 2: Standardize outputs ex. 24 fps 24000 sr
    print("Step 2: Standardizing outputs...")
    # standardize_samples(args.output_path, args.fps, args.audio_sr)

    # Step 3: Extract poses
    print("Step 3: Extracting poses...")
    # extract_poses(args.output_path, args.fps)

    # Step 4: Extract Instrumental to build Drum and Bass dataset
    print("Step 4: Extracting instrumental...")
    extract_instrumental(args.output_path, sr=args.audio_sr)

    # Step 5: Replace audio in rendered videos with instrumental
    print("Step 5: Replacing audio in rendered videos with instrumental...")
    replace_audio_in_renders(args.output_path)

    # Step 6: Pre-Build Audio Encodings
    print("Step 6: Pre-Build Audio Encodings...")
    preBuildEncodings(args.output_path, dnb=True)
    
    # Step 7: Save Minimum Data For Training
    print("Step 7: Saving minimum data for training...")
    min_data_out_path = '/' + os.path.join(*args.output_path.split('/'))+'_min_training_data'
    save_min_data_for_training(args.output_path, min_data_out_path)

    end = time.time()
    print(f"Total time taken to build the dataset: {(end-start)/60} minutes")

if __name__ == "__main__":
    main()