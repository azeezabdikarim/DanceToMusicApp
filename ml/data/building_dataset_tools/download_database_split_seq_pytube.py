from pytube import YouTube
import os
import csv
import argparse
from tqdm import tqdm
import subprocess
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import AudioFileClip, concatenate_videoclips, CompositeAudioClip
from datetime import datetime, timedelta

# Example of command to execute from youtube with a specific sequence length (time in seconds)
# python /Users/azeez/Documents/pose_estimation/Learning2Dance/youtube_dataset/scripts/download_database_split_seq_pytube.py --input_txts=/Users/azeez/Documents/pose_estimation/Learning2Dance/youtube_dataset/dataset/youtube_csv/afro_beats.csv --output_path=/Users/azeez/Documents/pose_estimation/Learning2Dance/youtube_dataset/dataset/samples/ --max_seq_len=10
# python DanceToMusic/data/building_tools/download_database_split_seq_pytube.py --input_csv=DanceToMusic/data/youtube_links/links_to_videos.csv --output_path= DanceToMusic/data/samples --max_seq_len=10

def calculate_sections(start_time, end_time, max_seq_len):
    start = datetime.strptime(start_time, "%H:%M:%S")
    end = datetime.strptime(end_time, "%H:%M:%S")
    total_seconds = int((end - start).total_seconds())
    
    sections = []
    for i in range(0, total_seconds, max_seq_len):
        new_start = (start + timedelta(seconds=i)).strftime("%H:%M:%S")
        new_end = (start + timedelta(seconds=min(i + max_seq_len, total_seconds))).strftime("%H:%M:%S")
        sections.append((new_start, new_end))
    
    return sections

def download_and_clip_video(video_url, start_time, end_time, output_path, video_name, max_seq_len):
    try:
        yt = YouTube(video_url)
        video_stream = yt.streams.filter(file_extension='mp4', only_video=True).first()
        audio_stream = yt.streams.filter(only_audio=True).first()

        if not video_stream or not audio_stream:
            raise Exception("Missing video or audio stream")

        video_file = video_stream.download(output_path=output_path, filename='temp_video')
        audio_file = audio_stream.download(output_path=output_path, filename='temp_audio')

        sections = calculate_sections(start_time, end_time, max_seq_len)
        
        for i, (s, e) in enumerate(sections):
            new_video_name = f"{video_name.split('.mp4')[0]}_{i}.mp4"
            clip_and_save_video(video_file, audio_file, s, e, output_path, new_video_name)

    except Exception as e:
        print(f"Failed to process video from {video_url}. Error: {e}")
        with open(output_path + "errors.txt", "a") as error_file:
            error_file.write(f"Failed to process video from {video_url}. Error: {e}\n")

def clip_and_save_video(video_path, audio_path, start_time, end_time, output_path, output_video_name):
    # Create a folder for each sample
    sample_folder = os.path.join(output_path, f"{output_video_name.split('.mp4')[0]}_sample/")
    os.makedirs(sample_folder, exist_ok=True)
    
    video_clip = VideoFileClip(video_path).subclip(start_time, end_time)
    audio_clip = AudioFileClip(audio_path).subclip(start_time, end_time)

    # Combine the audio and video
    final_audio = CompositeAudioClip([audio_clip])
    video_clip.audio = final_audio
    
    final_video_path = os.path.join(sample_folder, output_video_name)
    video_clip.write_videofile(final_video_path, codec="libx264")

    final_audio_path = os.path.join(sample_folder, output_video_name.split('.mp4')[0] + '.wav')
    audio_clip.write_audiofile(final_audio_path)



def parse_args():
    parser = argparse.ArgumentParser(description='Download youtube videos from a txt file and organize files in dataset style.')
    parser.add_argument("--input_csv", required=True, nargs='+',
                        help='path to txt file with url\'s of videos')
    parser.add_argument('--output_path', required=True,
                        help='Output path to create the dataset tree structure')
    parser.add_argument('--max_seq_len', required=False, type=int,
                        help='Maximum sequence length for each clip')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    max_seq_len = args.max_seq_len if args.max_seq_len else float('inf')
    
    output_path = args.output_path
    if output_path.rfind('/') != len(output_path)-1:
        output_path = output_path + '/'
    os.makedirs(args.output_path, exist_ok=True)

    for input_csv in args.input_csv:
        style = input_csv.split('/')[-1].split('.csv')[0]
        print(f"Downloading videos from style: {style}")

        with open(input_csv, mode='r') as infile:
            reader = csv.reader(infile)
            style_videos = [row for row in reader]

        for i, video_info in enumerate(tqdm(style_videos, desc="Downloading videos...")):
            video_url, start_time, end_time = video_info
            video_name = style + '_' + str(i) + '.mp4'
            download_and_clip_video(video_url, start_time, end_time, output_path, video_name, max_seq_len)

if __name__ == "__main__":
    main()
