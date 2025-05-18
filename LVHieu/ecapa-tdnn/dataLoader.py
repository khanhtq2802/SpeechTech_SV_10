
import os
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
import random

class trainDataset(Dataset):
    def __init__(self, data_list_path, data_path, max_length=16000*5, noise_level=0.005, add_noise=True):
        self.data_path = data_path
        self.max_length = max_length
        self.data_list = []
        self.noise_level = noise_level
        self.add_noise = add_noise

        with open(data_list_path, 'r') as f:
            lines = f.readlines()
        
        self.speakers = sorted(list(set([line.strip().split('	')[0] for line in lines])))
        self.spk2id = {spk: idx for idx, spk in enumerate(self.speakers)}

        for line in lines:
            spk, utt = line.strip().split('	')
            full_path = os.path.join(data_path, spk, utt)
            if os.path.exists(full_path):
                self.data_list.append([os.path.join(spk, utt), self.spk2id[spk]])
            else:
                print(f"[Warning] Missing file: {full_path}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        utt, label = self.data_list[index]
        audio_path = os.path.join(self.data_path, utt)

        try:
            audio, _ = torchaudio.load(audio_path)
            audio = audio.squeeze(0)  # assume mono
        except Exception as e:
            print(f"[Error] Failed to load {audio_path}: {e}")
            return self.__getitem__((index + 1) % len(self.data_list))

        # Pad or trim to max_length
        if audio.shape[0] < self.max_length:
            pad_len = self.max_length - audio.shape[0]
            audio = F.pad(audio, (0, pad_len))
        else:
            audio = audio[:self.max_length]

        # Thêm white noise augmentation nếu bật
        if self.add_noise:
            noise = torch.randn_like(audio) * self.noise_level
            audio = audio + noise
            # Clamp để tránh vượt ngoài [-1,1]
            audio = torch.clamp(audio, -1.0, 1.0)

        return audio, label


