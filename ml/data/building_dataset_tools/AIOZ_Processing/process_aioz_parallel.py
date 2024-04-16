import os
import argparse
import pickle
import numpy as np
from scipy.io import wavfile
import cv2
import torchaudio
import vedo
import trimesh
import time
from smplx import SMPL
import soundfile as sf

# import open3d as o3d
import torch
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FFMpegWriter
import matplotlib.animation as animation
from moviepy.editor import VideoFileClip, AudioFileClip
from transformers import EncodecModel
from spleeter.separator import Separator
import multiprocessing
import librosa
import gc

import matplotlib.pyplot as plt

# python /Users/azeez/Documents/pose_estimation/DanceToMusicApp/ml/data/building_dataset_tools/AIOZ_Processing/process_aioz_parallel.py --data_dir /Users/azeez/Documents/pose_estimation/DanceToMusicApp/ml/data/aioz/gdance --output_dir /Users/azeez/Documents/pose_estimation/DanceToMusicApp/ml/data/aioz/gdance_3sec_dnb --fps 30 --max_length 3 --mono True --smpl_path /Users/azeez/Documents/pose_estimation/DanceToMusicApp/ml/data/building_dataset_tools/AIOZ_Processing/models/smpl --compute_audio_codes True --dnb True


def load_data(data_dir, sample_name):
    # Load motion data
    motion_path = os.path.join(data_dir, "motions_smpl", f"{sample_name}.pkl")
    with open(motion_path, "rb") as f:
        data = pickle.load(f)

    smpl_poses = data['smpl_poses']
    root_trans = data['root_trans']
    smpl_betas = data['smpl_betas']
    num_frames = smpl_poses.shape[1]

    # Load music data
    music_path = os.path.join(data_dir, "musics", f"{sample_name}.wav")
    audio, sample_rate = torchaudio.load(music_path)

    return smpl_poses, root_trans, smpl_betas, num_frames, audio, sample_rate

def trim_data(smpl_poses, root_trans, smpl_betas, audio, fps, max_length, mono=False):
    num_frames = smpl_poses.shape[1]
    sample_rate = audio.shape[-1] // (num_frames / fps)

    if num_frames >= max_length * fps:
        # Split into segments of max_length
        num_segments = num_frames // (max_length * fps)
        segments = []
        for i in range(num_segments):
            start = i * max_length * fps
            end = min((i + 1) * max_length * fps, num_frames)

            segment_poses = smpl_poses[:, start:end]
            segment_trans = root_trans[:, start:end]
            segment_betas = smpl_betas[:, start:end]

            start_sample = int(start * sample_rate // fps)
            end_sample = int(end * sample_rate // fps)
            segment_audio = audio[:, start_sample:end_sample]

            if mono:
                segment_audio = segment_audio.mean(dim=0, keepdim=True)

            segments.append((segment_poses, segment_trans, segment_betas, segment_audio))

    return segments

def save_segment(output_dir, poses, trans, betas, audio, fps):
    os.makedirs(output_dir, exist_ok=True)

    # Save video
    height, width = 720, 1280
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vid_path = os.path.join(output_dir, "debug.mp4")
    video = cv2.VideoWriter(vid_path, fourcc, fps, (width, height))
    for j in range(poses.shape[1]):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        video.write(frame)
    video.release()

    # Save audio
    audio_data = audio.cpu().numpy()
    audio_data = np.clip(audio_data, -1, 1)  # Clip audio data to [-1, 1] range
    audio_data = (audio_data * 32767).astype(np.int16)  # Scale and convert to 16-bit integers
    # wavfile.write(os.path.join(output_dir, "audio.wav"), audio.shape[-1] // poses.shape[1], audio_data.astype(np.float32))
    # torchaudio.save(os.path.join(output_dir, "audio.pt"), audio, sample_rate)
    audio_path = os.path.join(output_dir, "audio.wav")
    torchaudio.save(audio_path, audio, 44100)
    # Save poses
    pose_path = os.path.join(output_dir, "poses.pkl")
    with open(pose_path, "wb") as f:
        pickle.dump({"smpl_poses": poses, "root_trans": trans, "smpl_betas": betas}, f)

    return vid_path, audio_path, pose_path


def generate_mesh(poses, trans, betas, smpl_path, output_dir, fps):
    smpl = SMPL(smpl_path, batch_size=1)
    N, T = poses.shape[:2]

    # Prepare the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def animate(frame):
        ax.clear()
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

        # Process each subject in the batch
        for n in range(N):
            global_orient = poses[n, frame, :3].reshape(1, 3)  # Global orientation for the subject
            body_pose = poses[n, frame, 3:].reshape(1, -1)    # Body pose for the subject
            smpl_trans = trans[n, frame].reshape(1, 3)        # Translation for the subject

            # Get vertices
            vertices = smpl(global_orient=torch.tensor(global_orient, dtype=torch.float32),
                            body_pose=torch.tensor(body_pose, dtype=torch.float32),
                            transl=torch.tensor(smpl_trans, dtype=torch.float32),
                            betas=torch.tensor(betas[n], dtype=torch.float32)).vertices[0].detach().numpy()

            ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=1)

    # Create the animation
    ani = FFMpegWriter(fps=fps)
    animation = matplotlib.animation.FuncAnimation(fig, animate, frames=T)

    # Save the animation
    output_path = os.path.join(output_dir, 'mesh_animation.mp4')
    animation.save(output_path, writer=ani)

    plt.close()

def generate_mesh_vedo(poses, trans, betas, smpl_path, output_dir, fps, sequence_name):
    smpl = SMPL(model_path=smpl_path, batch_size=1)
    N, T = poses.shape[:2]

    # Setup for vedo visualization
    plotter = vedo.Plotter(offscreen=True)

    for i in range(T):
        for n in range(N):
            # Reshape pose and translation data
            global_orient = poses[n, i, :3].reshape(1, 3)
            body_pose = poses[n, i, 3:].reshape(1, -1)
            smpl_trans = trans[n, i].reshape(1, 3)

            # Forward pass to get vertices
            vertices = smpl(global_orient=torch.tensor(global_orient, dtype=torch.float32),
                            body_pose=torch.tensor(body_pose, dtype=torch.float32),
                            transl=torch.tensor(smpl_trans, dtype=torch.float32),
                            betas=torch.tensor(betas[n], dtype=torch.float32)).vertices.detach().numpy()[0]
            
            # Create trimesh and vedo mesh objects
            trimesh_obj = trimesh.Trimesh(vertices, smpl.faces)
            vedo_mesh = vedo.Mesh(trimesh_obj)

            # Render and save each frame
            plotter.add(vedo_mesh)
            frame_file = os.path.join(output_dir, f"{sequence_name}_frame_{i:04d}.png")
            plotter.screenshot(filename=frame_file)
            plotter.clear()

    plotter.close()

def visualize_and_save_smpl(pose, trans, betas, smpl_model_path, output_video_path, fps=30):
    num_subjects, num_frames = pose.shape[0], pose.shape[1]
    smpl = SMPL(model_path=smpl_model_path, batch_size=num_subjects)

    # Reshape betas to remove the frame dimension
    betas = betas[:, 0, :]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.clear()
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 2])

        for n in range(num_subjects):
            global_orient = torch.tensor(pose[n, frame, :3].reshape(1, 3), dtype=torch.float32)
            body_pose = torch.tensor(pose[n, frame, 3:].reshape(1, -1), dtype=torch.float32)
            transl = torch.tensor(trans[n, frame].reshape(1, 3), dtype=torch.float32)
            betas_tensor = torch.tensor(betas[n].reshape(1, -1), dtype=torch.float32)

            joints = smpl(global_orient=global_orient, body_pose=body_pose, transl=transl, betas=betas_tensor).joints.detach().numpy().squeeze()

            ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], marker='o', s=5)

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=1000/fps)
    ani.save(os.path.join(output_video_path, "debug_render.mp4"), writer='ffmpeg', fps=fps)
    plt.close()


def visualize_and_save_smpl2D(pose, trans, betas, smpl_model_path, output_video_path, audio_path, fps=30, max_retries=3):
    num_subjects, num_frames = pose.shape[0], pose.shape[1]
    smpl = SMPL(model_path=smpl_model_path, batch_size=num_subjects)
    betas = betas[:, 0, :]

    joint_connections = [(19, 21), (19, 17), (16, 18), (18, 20), (1,4), (2,5), (5,8), (4,7),
                         (0,2), (0,1), (0,3), (17,14), (16,13), (3,6),(6,9), (14,9), (13,9)]
    # create a set of all value sin joint conidtions
    joint_connections = set(joint_connections)

    fig, ax = plt.subplots()
    joint_sequence = [[] for _ in range(num_subjects)]  # List for each subject

    def update(frame):
        ax.clear()
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])

        for n in range(num_subjects):
            global_orient = torch.tensor(pose[n, frame, :3].reshape(1, 3), dtype=torch.float32)
            body_pose = torch.tensor(pose[n, frame, 3:].reshape(1, -1), dtype=torch.float32)
            transl = torch.tensor(trans[n, frame].reshape(1, 3), dtype=torch.float32)
            betas_tensor = torch.tensor(betas[n].reshape(1, -1), dtype=torch.float32)

            joints = smpl(global_orient=global_orient, body_pose=body_pose, transl=transl, betas=betas_tensor).joints.detach().numpy().squeeze()
            joint_sequence[n].append(joints)  # Append joints for each subject

            # Project to 2D by using only X and Y coordinates
            ax.scatter(joints[:, 0], joints[:, 1], marker='o', s=5)

            # Annotate each joint with its index
            # for i, (x, y) in enumerate(joints[:, :2]):
                
            #     if i not in joint_connections and i // 2 != 0 or i == 0:
            #         ax.text(x, y, str(i), color="red", fontsize=8)

            # Draw lines between connected joints
            for start, end in joint_connections:
                ax.plot([joints[start, 0], joints[end, 0]],
                        [joints[start, 1], joints[end, 1]], 'b')

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=1000/fps)
    video_file_path = os.path.join(output_video_path, "debug_render_no_audio.mp4")
    ani.save(video_file_path, writer='ffmpeg', fps=fps)
    
    # audio_path = video_file_path.replace("debug_render_no_audio.mp4", "audio.wav")

    # Add audio to the video using moviepy
    video_clip = VideoFileClip(video_file_path)
    audio_clip = AudioFileClip(audio_path)
    video_clip_with_audio = video_clip.set_audio(audio_clip)
    final_video_path = video_file_path.replace("debug_render_no_audio.mp4", "debug_render.mp4")
    retry_count = 0
    while retry_count < max_retries:
        try:
            video_clip_with_audio.write_videofile(final_video_path, codec='libx264', audio_codec='aac')
            break
        except Exception as e:
            print(f"Error occurred while writing video file: {e}")
            print(f"Retrying ({retry_count + 1}/{max_retries})...")
            retry_count += 1
            time.sleep(1)  # Wait for a short duration before retrying

    if retry_count == max_retries:
        print("Max retries reached. Skipping video file writing.")

    os.remove(video_file_path)
    plt.close()

    # save the joints as a numpy file for later use
    joints_file_path = final_video_path.replace("debug_render.mp4", "joints.npy")
    joint_sequence = np.array(joint_sequence)
    np.save(joints_file_path, joint_sequence)

def save_audio_code(audio_path, encodec_model, sr = 24000):
    wav, _ = librosa.load(audio_path, sr=sr)
    wav = torch.tensor(wav)
    encoding = encodec_model.encode(wav.unsqueeze(0).unsqueeze(0), torch.ones(1, wav.shape[0], dtype=torch.bool).unsqueeze(0))
    one_audio_code = encoding['audio_codes'].view(1,-1,2)
    one_audio_code = one_audio_code.squeeze(0)

    audo_code_path = audio_path.replace('.wav', 'audio_code.npy')
    np.save(audo_code_path, one_audio_code)

def extractDrumNBass(wav_path, sr = 24000):
    # Initialize Spleeter
    separator = Separator('spleeter:4stems', multiprocess=False)

    # Determine output directory based on wav_path
    output_dir = os.path.dirname(wav_path)
    dir_cont = wav_path.split('/')[-2][:-7]
    spleeter_output_dir = os.path.join(output_dir, "spleeter_output", dir_cont)

    waveform, _ = librosa.load(wav_path, sr=sr, mono=True)
    separated = separator.separate(np.expand_dims(waveform, axis=1), audio_descriptor=wav_path)
    separator.join()

    os.makedirs(spleeter_output_dir, exist_ok=True)
    # Manually save each separated stem
    for stem, data in separated.items():
        stem_path = os.path.join(spleeter_output_dir, f"{stem}.wav")
        sf.write(stem_path, data, 24000)

    # Build string to help locate the output audio files 
    # Construct paths for the separated audio
    vocals_path = f'{spleeter_output_dir}/vocals.wav'
    drums_path = f'{spleeter_output_dir}/drums.wav'
    bass_path = f'{spleeter_output_dir}/bass.wav'
    other_path = f'{spleeter_output_dir}/other.wav'

    # Load the separated audio back into Python
    drums, _ = librosa.load(drums_path, sr=sr, mono=True)
    bass, _ = librosa.load(bass_path, sr=sr, mono=True)

    # Combine drums and bass tracks
    drum_and_bass = drums + bass

    # Save the combined drum and bass track
    # combined_path = f'{output_dir}/drum_and_bass.wav'

    combined_path = os.path.join(*wav_path.split('/')[:-1], f"{dir_cont}_drum_and_bass.wav")
    sf.write('/'+combined_path, drum_and_bass, sr)
    print(f"Saved dnb track to {combined_path}")
    del separator
    gc.collect()

def process_sample(data_dir, output_dir, sample_name, fps=30, max_length=10, mono=False, smpl_path="smpl/SMPL_MALE.pkl", compute_audio_codes=True, dnb = False):
    smpl_poses, root_trans, smpl_betas, num_frames, audio, sample_rate = load_data(data_dir, sample_name)
    segments = trim_data(smpl_poses, root_trans, smpl_betas, audio, fps, max_length, mono)

    if compute_audio_codes:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        model_id = "facebook/encodec_24khz"
        encodec_model = EncodecModel.from_pretrained(model_id)
        encodec_model.to(device)

    for i, (segment_poses, segment_trans, segment_betas, segment_audio) in enumerate(segments):
        output_segment_dir = os.path.join(output_dir, f"{sample_name}_{i}")
        if not os.path.exists(output_segment_dir):
            vid_path, audio_path, pose_path = save_segment(output_segment_dir, segment_poses, segment_trans, segment_betas, segment_audio, fps)
            if dnb:
                dnb_path = extractDrumNBass(audio_path)
                visualize_and_save_smpl2D(segment_poses, segment_trans, segment_betas, smpl_path, output_segment_dir, dnb_path)
                if compute_audio_codes:
                    save_audio_code(dnb_path, encodec_model)
            else:
                visualize_and_save_smpl2D(segment_poses, segment_trans, segment_betas, smpl_path, output_segment_dir, audio_path)
                if compute_audio_codes:
                    save_audio_code(audio_path, encodec_model)


def process_sample_wrapper(args):
    return process_sample(*args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--max_length", type=int, default=10, help="Maximum length of each sample in seconds")
    parser.add_argument("--mono", type=bool, default=True, help="Convert audio to mono")
    parser.add_argument("--dnb", type=bool, default=False, help="Convert audio to drum and bass track")
    parser.add_argument("--smpl_path", type=str, default="smpl/SMPL_MALE.pkl", help="Path to the SMPL model file")
    parser.add_argument("--compute_audio_codes", type=bool, default=True, help="Compute audio codes for the audio samples")
    args = parser.parse_args()

    with open(os.path.join(args.data_dir, "train_split_sequence_names.txt"), "r") as f:
        train_sequence_names = [line.strip() for line in f]

    with open(os.path.join(args.data_dir, "test_split_sequence_names.txt"), "r") as f:
        test_sequence_names = [line.strip() for line in f]

    with open(os.path.join(args.data_dir, "val_split_sequence_names.txt"), "r") as f:
        val_sequence_names = [line.strip() for line in f]

    # Create a multiprocessing pool with the number of processes equal to the number of CPU cores
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    # Prepare the arguments for each process_sample call
    train_args = [(args.data_dir, os.path.join(args.output_dir, "train"), sequence_name, args.fps, args.max_length, args.mono, args.smpl_path, args.compute_audio_codes, args.dnb) for sequence_name in train_sequence_names]
    val_args = [(args.data_dir, os.path.join(args.output_dir, "val"), sequence_name, args.fps, args.max_length, args.mono, args.smpl_path, args.compute_audio_codes, args.dnb) for sequence_name in val_sequence_names]
    test_args = [(args.data_dir, os.path.join(args.output_dir, "test"), sequence_name, args.fps, args.max_length, args.mono, args.smpl_path, args.compute_audio_codes, args.dnb) for sequence_name in test_sequence_names]

    # Process the samples in parallel
    pool.map(process_sample_wrapper, train_args)
    pool.map(process_sample_wrapper, val_args)
    pool.map(process_sample_wrapper, test_args)

    # Close the pool
    pool.close()
    pool.join()