import os
import numpy as np
import torch
from tqdm import tqdm
import pickle as pkl
import librosa
import torchaudio
from torch.utils.data import DataLoader, Dataset


class DanceToMusic_SMPL(Dataset):
    def __init__(self, directory, encoder = None, sample_rate=24000, device = torch.device("cpu"), num_samples = None, dnb=False):
        self.device = device
        self.raw_data = self._load_data(directory, sample_rate, num_samples, dnb)
        self.data = self._buildData(self.raw_data)
        if 'audio_codes' not in self.data.keys():
            self.encoder = encoder
            if encoder is not None:
                print("Encoding audio...")
                self.data = self._encodeAudio(encoder, self.data)

    def __len__(self):
        return len(self.data['joints'])

    def __getitem__(self, idx):
        # pose = self.data['poses'][idx]
        joints = self.data['joints'][idx]
        joint_mask = self.data['joint_padding_mask'][idx]
        wav = self.data['wavs'][idx]
        wav_path = self.data['wav_paths'][idx]
        sample_rate = self.data['sample_rate'][idx]
        # pose_mask = self.data['pose_padding_mask'][idx]
        wav_mask = self.data['audio_padding_mask'][idx]
        vid_path = self.data['video_paths'][idx]

        if 'audio_codes' not in self.data.keys():
            return [joints, joint_mask, wav, wav_mask, wav_path, vid_path, sample_rate]
        
        audio_codes = self.data['audio_codes'][idx]
        return [audio_codes, joints, joint_mask, wav, wav_mask, wav_path, vid_path, sample_rate]

    def _load_data(self, directory, sr, num_samples, dnb):
        poses = []
        joints = []
        wavs = []
        wav_paths = []
        sample_rate = []
        count_loaded_sample = 0
        audio_codes = []
        video_paths = []
        for root, dirs, files in os.walk(directory):
            for d in dirs:
                if 'error' not in d and 'spleeter' not in os.path.join(root,d):
                    joint_path = os.path.join(root, d, "joints.npy")
                    joint_arr = np.load(joint_path) 

                    pose_path = os.path.join(root, d, "poses.pkl")
                    motion = pkl.load(open(pose_path, "rb"))

                    if len(joint_path) > 0:
                        if dnb:
                            wav_path = os.path.join(root, d, f"audio_drum_and_bass.wav")
                        else:
                            wav_path = os.path.join(root, d, "audio.wav")
                        
                        vid_path = os.path.join(root, d, f"debug_render.mp4")
                        video_paths.append(vid_path)

                        audio_code_path = os.path.join(root, d, f"audioaudio_code.npy")
                        if os.path.exists(audio_code_path):
                            audio_code = np.load(audio_code_path)
                            audio_codes.append(audio_code)

                        # poses.append(self._buildPoses(pose_dir_path))
                        joints.append(joint_arr)
                        poses.append(motion)
                        wav, sr = librosa.load(wav_path, sr=sr)
                        sample_rate.append(sr)
                        wav_paths.append(wav_path)
                        wavs.append(wav)
                        count_loaded_sample += 1
                        if count_loaded_sample == num_samples:
                            break

        ret = {
            "poses": poses,
            "joints": joints,
            "wavs": wavs,
            "wav_paths": wav_paths,
            "sample_rate": sample_rate,
            "video_paths":video_paths
        }
        if len(audio_codes) > 0:
            min_len = min([len(x) for x in audio_codes])
            audio_codes = [x[:min_len] for x in audio_codes]
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
        # poses = [torch.tensor(p) for p in raw_data['poses']]
        # max_pos_seq_len = max(p.shape[0] for p in poses)

        joints = [torch.tensor(j) for j in raw_data['joints']]
        max_joint_seq_len = max(j.shape[1] for j in joints)
        
        # Handle audio
        wavs = [torch.tensor(w) for w in raw_data['wavs']]
        max_audio_len = max(w.shape[0] for w in wavs)
        
        num_samples = len(wavs)
        max_num_subjects = max(j.shape[0] for j in joints)
        
        # Padding for poses
        # padded_poses = torch.zeros(num_samples, max_pos_seq_len, poses[0].shape[1], poses[0].shape[2])
        # pose_padding_mask = torch.zeros(num_samples, max_pos_seq_len, dtype=torch.bool)
        
        # Fill the tensors for poses
        # for i, seq in enumerate(poses):
        #     n = seq.size(0)
        #     padded_poses[i, :n, :, :] = seq
        #     pose_padding_mask[i, :n] = 1  # Set mask to 1 for non-padded values

        # Padding for joints
        padded_joints = torch.zeros(num_samples, max_num_subjects, max_joint_seq_len, joints[0].size(2), joints[0].size(3))
        joint_padding_mask = torch.zeros(num_samples, max_num_subjects, max_joint_seq_len, dtype=torch.bool)

        # Fill the tensors for joints
        for i, seq in enumerate(joints):
            num_subjects = seq.size(0)
            n = seq.size(1)
            padded_joints[i, :num_subjects, :n, :, :] = seq
            joint_padding_mask[i, :num_subjects, :n] = 1

        # Padding for audio
        padded_audio = torch.zeros(num_samples, max_audio_len)
        audio_padding_mask = torch.zeros(num_samples, max_audio_len, dtype=torch.bool)

        # Fill the tensors for audio
        for i, audio in enumerate(wavs):
            n = audio.size(0)
            padded_audio[i, :n] = audio
            audio_padding_mask[i, :n] = 1  # Set mask to 1 for non-padded values

        data = {
            'wav_paths': raw_data['wav_paths'],
            'sample_rate': raw_data['sample_rate'],  # Assuming this is the same for all samples
            'video_paths': raw_data['video_paths'],
            # 'poses': padded_poses.to(self.device),
            # 'pose_padding_mask': pose_padding_mask.to(self.device),
            'joints': padded_joints.to(self.device),
            'joint_padding_mask': joint_padding_mask.to(self.device),
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

        