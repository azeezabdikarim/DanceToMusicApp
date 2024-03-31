import os
import argparse
import pickle
import numpy as np
from scipy.io import wavfile
import cv2
import torchaudio
import vedo
import trimesh
from smplx import SMPL
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

import matplotlib.pyplot as plt


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
    # else:
    #     segment_poses = smpl_poses
    #     segment_trans = root_trans
    #     segment_betas = smpl_betas
    #     segment_audio = audio

    #     if mono:
    #         segment_audio = segment_audio.mean(dim=0, keepdim=True)

    #     segments = [(segment_poses, segment_trans, segment_betas, segment_audio)]

    return segments

def save_segment(output_dir, poses, trans, betas, audio, fps):
    os.makedirs(output_dir, exist_ok=True)

    # Save video
    height, width = 720, 1280
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(os.path.join(output_dir, "debug.mp4"), fourcc, fps, (width, height))
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
    torchaudio.save(os.path.join(output_dir, "audio.wav"), audio, 44100)
    # Save poses
    with open(os.path.join(output_dir, "poses.pkl"), "wb") as f:
        pickle.dump({"smpl_poses": poses, "root_trans": trans, "smpl_betas": betas}, f)


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


def visualize_and_save_smpl2D(pose, trans, betas, smpl_model_path, output_video_path, fps=30):
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
    
    audio_path = video_file_path.replace("debug_render_no_audio.mp4", "audio.wav")

    # Add audio to the video using moviepy
    video_clip = VideoFileClip(video_file_path)
    audio_clip = AudioFileClip(audio_path)
    video_clip_with_audio = video_clip.set_audio(audio_clip)
    final_video_path = video_file_path.replace("debug_render_no_audio.mp4", "debug_render.mp4")
    video_clip_with_audio.write_videofile(final_video_path, codec='libx264', audio_codec='aac')
    os.remove(video_file_path)
    plt.close()

    # save the joints as a numpy file for later use
    joints_file_path = final_video_path.replace("debug_render.mp4", "joints.npy")
    joint_sequence = np.array(joint_sequence)
    np.save(joints_file_path, joint_sequence)

def process_sample(data_dir, output_dir, sample_name, fps=30, max_length=10, mono=False, smpl_path="smpl/SMPL_MALE.pkl"):
    smpl_poses, root_trans, smpl_betas, num_frames, audio, sample_rate = load_data(data_dir, sample_name)
    segments = trim_data(smpl_poses, root_trans, smpl_betas, audio, fps, max_length, mono)

    for i, (segment_poses, segment_trans, segment_betas, segment_audio) in enumerate(segments):
        output_segment_dir = os.path.join(output_dir, f"{sample_name}_{i}")
        print('in loop')
        save_segment(output_segment_dir, segment_poses, segment_trans, segment_betas, segment_audio, fps)
        # generate_mesh_vedo(segment_poses, segment_trans, segment_betas, smpl_path, output_segment_dir, fps, f"{sample_name}_{i}")
        # generate_mesh(segment_poses, segment_trans, segment_betas, smpl_path, output_segment_dir, fps)
        # plot_poses_to_video(segment_poses, segment_trans, output_segment_dir, fps)
        visualize_and_save_smpl2D(segment_poses, segment_trans, segment_betas, smpl_path, output_segment_dir)


if __name__ == "__main__":
    print('main')
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--max_length", type=int, default=10, help="Maximum length of each sample in seconds")
    parser.add_argument("--mono", type=bool, default=True, help="Convert audio to mono")
    parser.add_argument("--smpl_path", type=str, default="smpl/SMPL_MALE.pkl", help="Path to the SMPL model file")
    args = parser.parse_args()
    print("started")

    with open(os.path.join(args.data_dir, "test_split_sequence_names.txt"), "r") as f:
        sequence_names = [line.strip() for line in f]

    for sequence_name in sequence_names:
        process_sample(args.data_dir, args.output_dir, sequence_name, args.fps, args.max_length, args.mono, args.smpl_path)