import cv2
import mediapipe as mp
import numpy as np
import os
import argparse
from moviepy.editor import VideoFileClip, AudioFileClip

# python /Users/azeez/Documents/pose_estimation/Learning2Dance/youtube_dataset/scripts/pose_extraction_media_pipe.py --directory /Users/azeez/Documents/pose_estimation/Learning2Dance/youtube_dataset/dataset/samples

def saveVideoKeypoints(video_path, pose_model, save_dir, frame_rate = 24, mp_pose=None):
    # Initialize MediaPipe Drawing component
    mp_drawing = mp.solutions.drawing_utils
    # Extract video name for saving landmarks and rendered video
    video_name = os.path.basename(video_path).split('.')[0]

    # Initialize VideoCapture
    cap = cv2.VideoCapture(video_path)

    # Get the original frame rate of the video
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    # Calculate the number of frames to skip to achieve 24 FPS
    skip_frames = int(np.ceil(original_fps / frame_rate))
    frame_count = 0  # Counter for frames

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Initialize VideoWriter for saving the rendered video
    color_vid_save_path = os.path.join(save_dir,f"{video_path.split('/')[-3]}_{video_name}_3D_render.mp4")
    normalized_vid_save_path = os.path.join(save_dir,f"{video_path.split('/')[-3]}_{video_name}_3D_normalized_render.mp4")
    out = cv2.VideoWriter(color_vid_save_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame_width, frame_height))
    out_normalized = cv2.VideoWriter(normalized_vid_save_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame_width, frame_height))

    # Initialize a list to store landmarks
    landmarks_list = []
    # Main loop to read frames from the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Finished reading the video.")
            break
        
        # Only process frames when the frame_count is a multiple of skip_frames
        if (frame_count % skip_frames == 0):
            # Convert the BGR frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Create a white background frame for the normlaized render
            white_frame = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255

    # Draw normalized keypoints on the white frame

            # Perform pose detection
            results = pose_model.process(rgb_frame)

            # Draw the pose annotations on the frame
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                mp_drawing.draw_landmarks(white_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # Extract and store the 3D landmarks
                landmarks = results.pose_landmarks.landmark
                
                # Convert landmarks to a NumPy array and append to list. drop point 0, the nose so that the array is even in length
                frame_landmarks = np.array([(lmk.x, lmk.y, lmk.z) for lmk in landmarks[1:]])
                landmarks_list.append(frame_landmarks)

            # Save the rendered frame to the output video
            out.write(frame)
            out_normalized.write(white_frame)
        frame_count += 1 

    # Release the capture and close windows
    cap.release()
    out.release()
    out_normalized.release()
    
    # Add audio to the saved videos
    audio_path = os.path.join(save_dir,f"{video_name}.wav") 
    audio = AudioFileClip(audio_path)
    
    # orig_video = VideoFileClip(video_path)
    # orig_video = orig_video.set_audio(audio)
    # orig_video.write_videofile(video_path, codec="libx264", audio_codec='aac')

    video = VideoFileClip(color_vid_save_path)
    video = video.set_audio(audio)
    video.write_videofile(os.path.join(save_dir, f"{video_name}_with_audio.mp4"), codec="libx264", audio_codec='aac')

    video_normalized = VideoFileClip(normalized_vid_save_path)
    video_normalized = video_normalized.set_audio(audio)
    video_normalized.write_videofile(os.path.join(save_dir, f"{video_name}_normalized_with_audio.mp4"), codec="libx264", audio_codec='aac')


    # Convert list to a NumPy array of shape [num_frames, num_keypoints, 3]
    final_landmarks_array = np.stack(landmarks_list, axis=0)
    pose_save_path = os.path.join(save_dir, f"{video_path.split('/')[-3]}_{video_name}_3D_landmarks.npy")
    np.save(pose_save_path, final_landmarks_array)

    os.remove(normalized_vid_save_path)
    os.remove(color_vid_save_path)

    print(f"Saved rendered video as {video_name}_render.mp4")
    print(f"Saved landmarks as {video_name}_landmarks.npy. The video was {len(landmarks_list)} frames long.)")

if __name__ == "__main__":
    # Initialize MediaPipe Pose component
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False, smooth_landmarks=True)

    # Argument parser
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--directory', type=str, help='Input video directory path')
    args = parser.parse_args()

    # Input video directory path from command-line argument
    directory = args.directory

    for root, dirs, files in os.walk(directory):
        for d in dirs:
            vid_path = os.path.join(root, d, f"{d[:-7]}.mp4")
            saveVideoKeypoints(vid_path, pose, save_dir=os.path.join(root, d), mp_pose=mp_pose)