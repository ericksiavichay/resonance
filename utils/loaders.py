import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset


class ESC50Loader(Dataset):
    def __init__(self, path):
        """
        Args:
            path (string): Path to the folder containing the ESC-50 dataset
        """

        self.path = path
        self.meta_data = pd.read_csv(path + "meta/esc50.csv")
        self.meta_data["category"] = self.meta_data["category"].str.replace("_", " ")
        self.audio_files = os.listdir(path + "audio")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_name = os.path.join(self.path + "audio/", self.audio_files[idx])
        waveform, sample_rate = torchaudio.load(audio_name)
        waveform = waveform.squeeze(0)
        text_label = self.meta_data["category"][idx]

        return waveform, sample_rate, text_label
