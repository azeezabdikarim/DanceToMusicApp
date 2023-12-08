from download_database_split_seq_pytube import *
from standardize_fps_sr_youtube import *
from pose_extraction_media_pipe import *
from copy_min_training_data import *
from seperate_vocals import *
import csv
import os


# python /Users/azeez/Documents/pose_estimation/DanceToMusic/data/building_tools/build_complete_dataset.py --output_path /Users/azeez/Documents/pose_estimation/DanceToMusic/data/samples/5sec_expando_dataset --input_csv /Users/azeez/Documents/pose_estimation/DanceToMusic/data/youtube_links/youtube_links.csv  --max_seq_len 5 --fps 24
# python /Users/azeez/Documents/pose_estimation/DanceToMusicApp/ml/data/building_dataset_tools/build_complete_dataset.py --output_path /Users/azeez/Documents/pose_estimation/DanceToMusicApp/ml/data/samples/5sec_expando_test --input_csv /Users/azeez/Documents/pose_estimation/DanceToMusicApp/ml/data/youtube_links/youtube_links_test.csv  --max_seq_len 5 --fps 24
def download_and_clip_videos(input_csv, output_path, max_seq_len):    
    if output_path.rfind('/') != len(output_path)-1:
        output_path = output_path + '/'
    os.makedirs(output_path, exist_ok=True)

    for csv_path in input_csv:
        style = csv_path.split('/')[-1].split('.csv')[0]
        print(f"Downloading videos from style: {style}")

        with open(csv_path, mode='r') as infile:
            reader = csv.reader(infile)
            style_videos = [row for row in reader]

        for i, video_info in enumerate(tqdm(style_videos, desc="Downloading videos...")):
            video_url, start_time, end_time = video_info
            video_name = style + '_' + str(i) + '.mp4'
            download_and_clip_video(video_url, start_time, end_time, output_path, video_name, max_seq_len)

def standardize_samples(output_path, fps, audio_sr):
    videos_path = get_videos_path(output_path)
    # videos_path = [video_path + '/' + video_path.split('/')[-1] for video_path in videos_path]

    for video in tqdm(videos_path, desc="Standardizing videos..."):
        standardize_data(video, fps, audio_sr)

def extractInstrumental(data_dir, sr = 24000, mp_pose=None):
    directory = data_dir

    for root, dirs, files in os.walk(directory):
        for d in dirs:
            wav_path = os.path.join(root, d, f"{d[:-7]}.wav")
            extractDrumNBass(wav_path, sr=sr)

def extract_poses(data_dir, fps):
    # Initialize MediaPipe Pose component
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False, smooth_landmarks=True)

    # Input video directory path from command-line argument
    directory = data_dir

    for root, dirs, files in os.walk(directory):
        for d in dirs:
            vid_path = os.path.join(root, d, f"{d[:-7]}.mp4")
            saveVideoKeypoints(vid_path, pose, save_dir=os.path.join(root, d), frame_rate=fps, mp_pose=mp_pose)

def save_min_data_for_training(data_dir, min_output_dir):
    output_path = min_output_dir
    sr = 24000  # Sample rate for librosa, you can change it based on your needs

    os.makedirs(output_path, exist_ok=True)

    for root, dirs, files in os.walk(data_dir):
        for d in dirs:
            if 'error' not in d:
                pose_path = os.path.join(root, d, f"{root.split('/')[-1]}_{d[:-7]}_3D_landmarks.npy")
                wav_path = os.path.join(root, d, f"{d[:-7]}.wav")

                sample_poses = np.load(pose_path)
                wav, _ = librosa.load(wav_path, sr=sr)

                sample_output_dir = os.path.join(output_path, d)
                pose_output_path = os.path.join(sample_output_dir, f"{root.split('/')[-1]}_{d[:-7]}_3D_landmarks.npy")
                wav_output_path = os.path.join(sample_output_dir, f"{d[:-7]}.wav")

                # Create the output directory if it doesn't exist
                os.makedirs(sample_output_dir, exist_ok=True)

                # Save the pose and wav files
                np.save(pose_output_path, sample_poses)
                sf.write(wav_output_path, wav, sr)

def main():
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
    download_and_clip_videos(args.input_csv, args.output_path, args.max_seq_len)

    # Step 2: Standardize outputs
    print("Step 2: Standardizing outputs...")
    standardize_samples(args.output_path, args.fps, args.audio_sr)

    # Step 3: Extract poses
    print("Step 3: Extracting poses...")
    extract_poses(args.output_path, args.fps)

    # Step 4: Pre-Build Audio Encodings
    print("Step 4: Pre-Build Audio Encodings...")
    preBuildEncodings(args.output_path, dnb=False)

    # Step 5: Save Minimum Data For Training
    print("Step 5: Saving minimum data for training...")
    min_data_out_path = '/' + os.path.join(*args.output_path.split('/'))+'_min_training_data'
    save_min_data_for_training(args.output_path, min_data_out_path)



if __name__ == "__main__":
    main()