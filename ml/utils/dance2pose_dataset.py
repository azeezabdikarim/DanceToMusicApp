import os
import numpy as np
import torch
from tqdm import tqdm
import pickle as pkl
import librosa
import torchaudio
from torch.utils.data import DataLoader, Dataset


class DanceToMusic(Dataset):
    def __init__(self, directory, encoder = None, sample_rate=24000, device = torch.device("cpu"), num_samples = None, dnb=False, clean_poses = False):
        self.device = device
        self.clean_poses = clean_poses
        self.raw_data = self._load_data(directory, sample_rate, num_samples, dnb)
        self.data = self._buildData(self.raw_data)
        if 'audio_codes' not in self.data.keys():
            self.encoder = encoder
            if encoder is not None:
                self.data = self._encodeAudio(encoder, self.data)

    def __len__(self):
        return len(self.data['poses'])

    def __getitem__(self, idx):
        pose = self.data['poses'][idx]
        wav = self.data['wavs'][idx]
        wav_path = self.data['wav_paths'][idx]
        sample_rate = self.data['sample_rate'][idx]
        pose_mask = self.data['pose_padding_mask'][idx]
        wav_mask = self.data['audio_padding_mask'][idx]
        vid_path = self.data['video_paths'][idx]

        if 'audio_codes' not in self.data.keys():
            return [pose, pose_mask, wav, wav_mask, wav_path, vid_path, sample_rate]
        
        audio_codes = self.data['audio_codes'][idx]
        return [audio_codes, pose, pose_mask, wav, wav_mask, wav_path, vid_path, sample_rate]

    def _load_data(self, directory, sr, num_samples, dnb):
        poses = []
        wavs = []
        wav_paths = []
        sample_rate = []
        count_loaded_sample = 0
        audio_codes = []
        video_paths = []
        for root, dirs, files in os.walk(directory):
            for d in dirs:
                if 'error' not in d and 'spleeter' not in os.path.join(root,d):
                    pose_path = os.path.join(root, d, f"{root.split('/')[-1]}_{d[:-7]}_3D_landmarks.npy")
                    pose_arr = np.load(pose_path) 
                    if len(pose_arr) > 0:
                        if dnb:
                            wav_path = os.path.join(root, d, f"{d[:-7]}_drum_and_bass.wav")
                        else:
                            wav_path = os.path.join(root, d, f"{d[:-7]}.wav")
                        
                        vid_path = os.path.join(root, d, f"{d[:-7]}.mp4")
                        if not os.path.exists(vid_path):
                            vid_path = ''
                        video_paths.append(vid_path)

                        audio_code_path = wav_path.replace('.wav', '_audio_code.npy')
                        if os.path.exists(audio_code_path):
                            audio_code = np.load(audio_code_path)
                            audio_codes.append(audio_code)

                        # poses.append(self._buildPoses(pose_dir_path))
                        poses.append(pose_arr)
                        wav, sr = librosa.load(wav_path, sr=sr)
                        sample_rate.append(sr)
                        wav_paths.append(wav_path)
                        wavs.append(wav)
                        count_loaded_sample += 1
                        if count_loaded_sample == num_samples:
                            break

        ret = {
            "poses": poses,
            "wavs": wavs,
            "wav_paths": wav_paths,
            "sample_rate": sample_rate,
            "video_paths":video_paths
        }
        if len(audio_codes) > 0:
            ret['audio_codes'] = audio_codes
        return ret

    # precomuting the encoding of the adio and adding a start token at the beginning of each audio code sequence
    def _encodeAudio(self, encoder, data):
        audio_codes = []
        for i, wav in enumerate(data['wavs']):
            wav = wav.unsqueeze(0).to(self.device)
            audio_padding_mask = data['audio_padding_mask'].to(self.device)
            encoding = encoder.encode(wav, audio_padding_mask[i].unsqueeze(0))
            # one_audio_code = encoding['audio_codes'].view(1,1,-1)
            one_audio_code = encoding['audio_codes'].view(1,-1,2)
            one_audio_code = one_audio_code.squeeze(0)
            # # Adding a start token at the beginning of each audio code sequence
            audio_code_with_start = torch.cat((torch.zeros(1,2).to(self.device), one_audio_code), dim=0)
            audio_codes.append(audio_code_with_start.to(self.device))
        data['audio_codes'] = audio_codes
        return data

    def _buildData(self, raw_data, movement_threshold=0.1, keypoints_threshold=4, frame_threshold = 0.2):
        # Handle poses
        poses = [torch.tensor(p) for p in raw_data['poses']]
        max_pos_seq_len = max(p.shape[0] for p in poses)
        
        # Handle audio
        wavs = [torch.tensor(w) for w in raw_data['wavs']]
        max_audio_len = max(w.shape[0] for w in wavs)
        
        num_samples = len(poses)
        
        # Padding for poses
        padded_poses = torch.zeros(num_samples, max_pos_seq_len, poses[0].shape[1], poses[0].shape[2])
        pose_padding_mask = torch.zeros(num_samples, max_pos_seq_len, dtype=torch.bool)
        
        # Padding for audio
        padded_audio = torch.zeros(num_samples, max_audio_len)
        audio_padding_mask = torch.zeros(num_samples, max_audio_len, dtype=torch.bool)

        # Fill the tensors for poses
        for i, seq in enumerate(poses):
            n = seq.size(0)
            padded_poses[i, :n, :, :] = seq
            pose_padding_mask[i, :n] = 1  # Set mask to 1 for non-padded values
        
        # Fill the tensors for audio
        for i, audio in enumerate(wavs):
            n = audio.size(0)
            padded_audio[i, :n] = audio
            audio_padding_mask[i, :n] = 1  # Set mask to 1 for non-padded values

        if self.clean_poses:
            select_indexes = []
            for i, p in enumerate(padded_poses):
                frame_pose_errors = self._keypoint_stability_check(p, movement_threshold, keypoints_threshold)
                if len(frame_pose_errors)/len(p) < frame_threshold:
                    select_indexes.append(i)
            data = {
                'wav_paths': [raw_data['wav_paths'][i] for i in select_indexes],
                'sample_rate': raw_data['sample_rate'],  # Assuming this is the same for all samples
                'video_paths': [raw_data['video_paths'][i] for i in select_indexes],
                'audio_codes': [raw_data['audio_codes'][i] for i in select_indexes] if 'audio_codes' in raw_data else None,
                'poses': padded_poses[select_indexes].to(self.device),
                'pose_padding_mask': pose_padding_mask[select_indexes].to(self.device),
                'wavs': padded_audio[select_indexes].unsqueeze(1).to(self.device),
                'audio_padding_mask': audio_padding_mask[select_indexes].to(self.device)
            }
        else:
            data = {
                'wav_paths': raw_data['wav_paths'],
                'sample_rate': raw_data['sample_rate'],  # Assuming this is the same for all samples
                'video_paths': raw_data['video_paths'],
                'poses': padded_poses.to(self.device),
                'pose_padding_mask': pose_padding_mask.to(self.device),
                'wavs': padded_audio.unsqueeze(1).to(self.device),
                'audio_padding_mask': audio_padding_mask.to(self.device)
            }
            if 'audio_codes' in raw_data.keys():
                data['audio_codes'] = raw_data['audio_codes']
        
        return data


    def _buildPoses(self, directory):
        poses = []
        for root, dirs, files in os.walk(directory):
            for f in files:
                if '.npy' in f:
                    pose = np.load(os.path.join(root, f))
                    poses.append(pose)
        return np.array(poses)
    
    def _calculate_keypoint_movement(self, kp1, kp2):
        return np.linalg.norm(kp1 - kp2, axis=1)

    def _keypoint_stability_check(self, data, movement_threshold, keypoints_threshold):
        # data is a numpy array of shape [n, 32, 3]
        error_frames = []
        for i in range(1, len(data)):
            # Calculate movement for each keypoint from frame i-1 to i
            movement = self._calculate_keypoint_movement(data[i-1, :, :2], data[i, :, :2])
            
            # Identify occluded keypoints (assuming they are marked as NaN)
            occluded = np.isnan(data[i-1, :, :2]) | np.isnan(data[i, :, :2])
            
            # Ignore occluded keypoints
            movement[occluded] = 0
            
            # Count keypoints with movement greater than the threshold
            keypoints_exceeding_threshold = np.sum(movement > movement_threshold)
            
            # Check if the count exceeds the threshold
            if keypoints_exceeding_threshold >= keypoints_threshold:
                error_frames.append(i)

        return np.array(error_frames)

        