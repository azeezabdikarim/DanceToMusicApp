from copy import deepcopy
from tqdm import tqdm
import subprocess
import json
import argparse
import os

# Command to executte from the commnad line at sr=24000 and 24fps
# python /Users/azeez/Documents/pose_estimation/Learning2Dance/youtube_dataset/scripts/standardize_fps_sr_youtube.py --dataset_path /Users/azeez/Documents/pose_estimation/Learning2Dance/youtube_dataset/dataset/samples --sample_rate 24000 --fps 24

def get_videos_path(dataset_path):
    videos_path = []
    for style in os.listdir(dataset_path):

        vid_path = os.path.join(dataset_path, style,style[:-7] + '.mp4') # the slicing removes the tag 'sample' from the directory's name
        if os.path.exists(vid_path):
            videos_path.append(vid_path)
    return videos_path

# def standardize_data(input_video, fps, audio_sr):
#     output_video = deepcopy(input_video)
#     index = output_video.rfind('I')
#     output_video = output_video[:index] + 'temp.mp4'

#     cmd = "ffmpeg -y -loglevel error -i " + input_video + " -r " + str(fps) + " -ac 1 -ar " + str(audio_sr) + " " + output_video
#     os.system(cmd)
#     os.system('rm ' + input_video)
#     os.system('mv ' + output_video + ' ' + input_video) 

def get_video_fps(video_path):
    cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of json {video_path}"
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
    info = json.loads(result.stdout)
    r_frame_rate = info["streams"][0]["r_frame_rate"]
    numerator, denominator = map(int, r_frame_rate.split('/'))
    return numerator / denominator

def standardize_data(input_video, fps, audio_sr):
    output_video = deepcopy(input_video)
    index = output_video.rfind('I')
    output_video = output_video[:index] + 'temp.mp4'

    cmd = f"ffmpeg -y -loglevel error -i {input_video} -vf fps={fps} -ac 1 -ar {audio_sr} {output_video}"
    os.system(cmd)
    os.system(f'rm {input_video}')
    os.system(f'mv {output_video} {input_video}')

def parse_args():
    """ Parse input arguments """
    parser = argparse.ArgumentParser(description='Standardize dataset video fps and audio sample rate.')

    parser.add_argument('--dataset_path', default="", help='Path to dataset.')
    parser.add_argument('--sample_rate', default=24000, type=int, help='Audio sample rate.')
    parser.add_argument('--fps', default=24, type=int, help='Video fps.')

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    videos_path = get_videos_path(args.dataset_path)
    # videos_path = [video_path + '/' + video_path.split('/')[-1] for video_path in videos_path]

    for video in tqdm(videos_path, desc="Standardizing videos..."):
        standardize_data(video, args.fps, args.sample_rate)

if __name__ == '__main__':
    main()