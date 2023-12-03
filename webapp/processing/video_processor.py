import os
import subprocess
import json
import shutil
from copy import deepcopy
from .pose_extractor import *
from .generate_audio import *

def get_video_fps(video_path):
    """Get the frames per second of the video."""
    cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of json {video_path}"
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
    info = json.loads(result.stdout)
    r_frame_rate = info["streams"][0]["r_frame_rate"]
    numerator, denominator = map(int, r_frame_rate.split('/'))
    return numerator / denominator

def standardize_video(input_video, fps, audio_sr):
    """Standardize the video frame rate and audio sample rate."""
    output_video = deepcopy(input_video)
    index = output_video.rfind('I')
    output_video = output_video[:index] + 'temp.mp4'

    cmd = f"ffmpeg -y -loglevel error -i {input_video} -vf fps={fps} -ac 1 -ar {audio_sr} {output_video}"
    subprocess.run(cmd, shell=True, check=True)
    os.replace(output_video, input_video)

def trim_video(input_video, audio_sr = 24000, length=5):
    """Trim the video to the specified length in seconds and extract audio."""
    # Generate the output video filename by replacing the extension
    trimmed_output = input_video.replace('.mp4', '_trimmed.mp4')
    # Generate the audio filename by replacing the video extension with .wav
    audio_output = input_video.replace('.mp4', '_trimmed.wav')

    # Command to trim the video
    trim_cmd = f"ffmpeg -y -loglevel error -i {input_video} -t {length} -c copy {trimmed_output}"
    subprocess.run(trim_cmd, shell=True, check=True)

    # Command to extract audio from the trimmed video and save as .wav
    extract_audio_cmd = f"ffmpeg -y -loglevel error -i {trimmed_output} -ac 1 -ar {audio_sr} -vn {audio_output}"
    subprocess.run(extract_audio_cmd, shell=True, check=True)

    # Return both the trimmed video path and the audio path
    return trimmed_output

def h264_aac_codec(input_video):
    output_video = input_video + ".temp.mp4"
    cmd = f"ffmpeg -i {input_video} -vcodec h264 -acodec aac {output_video}"
    
    subprocess.run(cmd, shell=True, check=True)

    # Replace the original file with the processed one
    shutil.move(output_video, input_video)

    return input_video

def process_video(video_path, fps=24, audio_sr=24000, length=5):
    """Process a video to standardize its frame rate, audio sample rate, and length."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video path {video_path} does not exist.")

    # Standardize video fps and audio sample rate
    standardize_video(video_path, fps, audio_sr)

    # Trim the video to the first 5 seconds
    trimmed_video_path = trim_video(video_path, audio_sr, length)

    keypoint_render_path, pose_save_path = extractPoses(trimmed_video_path)

    video_and_gen_audio = generateAudio(keypoint_render_path, pose_save_path)

    video_and_gen_audio = h264_aac_codec(video_and_gen_audio)

    return video_and_gen_audio

# The following allows the script to be run standalone for debugging or direct command line usage
if __name__ == '__main__':
    import argparse

    def parse_args():
        """ Parse input arguments """
        parser = argparse.ArgumentParser(description='Standardize dataset video fps and audio sample rate.')
        parser.add_argument('--video_path', help='Path to the video file to process.', required=True)
        parser.add_argument('--sample_rate', default=24000, type=int, help='Audio sample rate.')
        parser.add_argument('--fps', default=24, type=int, help='Video fps.')
        return parser.parse_args()

    args = parse_args()
    process_video(args.video_path, args.fps, args.sample_rate)
