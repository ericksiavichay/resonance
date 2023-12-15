"""
Class for data augmentation
"""

import random
import torchaudio.transforms as T
import librosa
import torch
import numpy as np


class AudioAugmentations:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def time_stretch(self, audio):
        stretch_rates = [1 / 0.75, 1 / 0.5, 1 / 0.25, 0.75, 0.5, 0.25]
        rate = random.choice(stretch_rates)
        waveform_numpy = librosa.effects.time_stretch(y=audio.numpy(), rate=rate)
        return torch.from_numpy(waveform_numpy)

    def pitch_shift(self, audio):
        n_steps = 0
        while n_steps == 0:
            n_steps = random.randint(-3, 3)

        waveform_numpy = librosa.effects.pitch_shift(
            y=audio.numpy(), sr=self.sample_rate, n_steps=n_steps
        )

        return torch.from_numpy(waveform_numpy)

    def add_background_noise(self, audio, weight=0.25):
        noise = torch.randn_like(audio)
        return audio + weight * noise

    def time_shift(self, audio, shift_max=9):
        shift = random.randint(1, shift_max)
        waveform_numpy = np.roll(audio, shift * self.sample_rate, axis=0)
        return torch.from_numpy(waveform_numpy)

    # def dynamic_range_compression(self, audio):
    #     compressor = T.DynamicRangeCompression()
    #     return compressor(audio)

    # def random_crop(self, audio, crop_length):
    #     start = random.randint(0, max(0, audio.shape[1] - crop_length))
    #     return audio[:, start : start + crop_length]

    def volume_perturb(self, audio):
        factor = torch.FloatTensor(1).uniform_(0.25, 1.75)
        perturbed_audio = torch.clamp(audio * factor, -1.0, 1.0)
        return perturbed_audio

    # def frequency_mask(self, audio, freq_mask_param):
    #     freq_mask = T.FrequencyMasking(freq_mask_param)
    #     return freq_mask(audio)

    # def time_mask(self, audio, time_mask_param):
    #     time_mask = T.TimeMasking(time_mask_param)
    #     return time_mask(audio)

    def random_transforms(self, audio):
        transforms = [
            self.time_stretch,
            self.pitch_shift,
            self.add_background_noise,
            # self.time_shift,
            # self.dynamic_range_compression,
            # self.random_crop,
            self.volume_perturb,
            # self.frequency_mask,
            # self.time_mask,
        ]
        aug_fn_1, aug_fn_2 = random.sample(transforms, 2)
        return aug_fn_1(audio), aug_fn_2(audio)
