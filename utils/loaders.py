import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
import numpy as np
from ast import literal_eval

# Enable multi-threading
# pd.options.mode.chained_assignment = "pyarrow"


# class MusicCapsLoader(Dataset):
#     def __init__(self, path="./musiccaps-public.csv"):
#         """
#         Args:
#             path (string): Path to the folder containing the MusicCaps dataset
#         """

#         self.path = path


class ESC50Loader(Dataset):
    def __init__(self, path):
        """
        Args:
            path (string): Path to the folder containing the ESC-50 dataset
        """

        self.path = path
        self.meta_data = pd.read_csv(path + "meta/esc50.csv")
        self.meta_data["category"] = self.meta_data["category"].str.replace("_", " ")

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_name = os.path.join(self.path + "audio/", self.meta_data["filename"][idx])
        waveform, sample_rate = torchaudio.load(audio_name)
        waveform = waveform.mean(dim=0, keepdim=True)  # takes care of stereo sound
        waveform = torch.squeeze(waveform, dim=0)
        text_label = self.meta_data["category"][idx]
        audio_id = self.meta_data["filename"][idx].split(".")[0]

        return waveform, sample_rate, text_label, audio_id


class MusicCapsLoader(Dataset):
    def __init__(self, path):
        """
        Args:
            path (string): Path to the folder containing the MusicCaps dataset
        """

        self.path = path
        self.meta_data = pd.read_csv("utils/musiccaps-public.csv")

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_name = os.path.join(self.path, f"{idx}.wav")
        waveform, sample_rate = torchaudio.load(audio_name)
        waveform = waveform.mean(dim=0, keepdim=True)  # takes care of stereo sound
        waveform = torch.squeeze(waveform, dim=0)
        aspect_list = literal_eval(self.meta_data["aspect_list"][idx])
        aspect_str = ", ".join(aspect_list)
        caption = self.meta_data["caption"][idx]

        return waveform, sample_rate, aspect_str, caption
