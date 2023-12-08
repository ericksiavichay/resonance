import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
import numpy as np

# Enable multi-threading
# pd.options.mode.chained_assignment = "pyarrow"


class ESC50Loader(Dataset):
    def __init__(self, path, subset="train", validation_split=0.2, random_state=42):
        """
        Args:
            path (string): Path to the folder containing the ESC-50 dataset
        """

        self.path = path
        self.meta_data = pd.read_csv(path + "meta/esc50.csv")
        self.meta_data["category"] = self.meta_data["category"].str.replace("_", " ")
        self.audio_files = os.listdir(path + "audio")

        # Split data
        np.random.shuffle(self.audio_files, random_state=random_state)
        total_samples = len(self.audio_files)
        split_idx = int(np.floor(validation_split * total_samples))
        if subset == "training":
            self.audio_files = self.audio_files[split_idx:]
        elif subset == "validation":
            self.audio_files = self.audio_files[:split_idx]

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_name = os.path.join(self.path + "audio/", self.audio_files[idx])
        waveform, sample_rate = torchaudio.load(audio_name)
        waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)
        text_label = self.meta_data["category"][idx]

        return waveform, sample_rate, text_label
