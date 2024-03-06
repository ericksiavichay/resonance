"""
Main file for training the VAE model
"""

import wandb
import torch
import torch.nn as nn
import torchaudio.transforms as T
import torch.nn.functional as F
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import librosa

from utils import loaders
import models.config as config
from models.vae import SimpleVAE

# HYPERPARAMETERS
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 10
BETA = 1.0
VAL_SPLIT = 0.3


def pad_batch_to_divisible_by_8(tensor):
    """
    Pads each spectrogram in the batch on the top and/or right to make its height and width divisible by 8.
    Assumes the tensor is in the shape (N, 1, H, W).
    """
    # Get the current height and width
    _, _, H, W = tensor.shape

    # Calculate the padding needed for height and width
    H_pad = (8 - H % 8) % 8
    W_pad = (8 - W % 8) % 8

    # Pad the tensor. The padding format is (left, right, top, bottom)
    padded_tensor = F.pad(tensor, (0, W_pad, H_pad, 0), mode="constant", value=0)

    return padded_tensor


def split_data(data, val_split=0.2, seed=42):
    val_size = int(len(data) * val_split)
    train_size = len(data) - val_size
    return random_split(
        data, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
    )


def get_mel_spectrogram(
    waveform,
    sample_rate,
    f_min=config.fmin,
    f_max=config.fmax,
    n_mels=config.mel_bins,
    n_fft=config.window_size,
    hop_length=config.hop_size,
    win_length=config.window_size,
    window_fn=torch.hann_window,
    center=True,
    pad_mode="reflect",
):
    mel_specgram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        f_max=f_max,
        f_min=f_min,
        n_mels=n_mels,
        hop_length=hop_length,
        win_length=win_length,
        window_fn=window_fn,
        center=center,
        pad_mode=pad_mode,
    )

    specs = mel_specgram(waveform)
    db_transform = T.AmplitudeToDB(top_db=80)
    specs = db_transform(specs)

    # make sure shape is (B, 1, F, T)
    specs = specs.unsqueeze(1)
    # specs = pad_batch_to_divisible_by_8(specs)

    return specs


class VAELoss(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()

        self.beta = beta

    def forward(self, x, x_hat, mean, log_var):
        """
        x: (B, C, H, W)
        x_hat: (B, C, H, W)
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(x_hat, x, reduction="mean")
        # KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())

        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss


if __name__ == "__main__":
    # Load the dataset
    print("Loading data...")
    esc50 = loaders.ESC50Loader("./ESC-50-master/ESC-50-master/")
    esc50_train, esc50_val = split_data(esc50, val_split=VAL_SPLIT)
    esc50_train_loader = torch.utils.data.DataLoader(
        esc50_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    esc50_val_loader = torch.utils.data.DataLoader(
        esc50_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    wandb.login()
    wandb.init(
        project="resonance",
        group="vae",
        config={
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "dataset": "ESC-50",
            "beta": BETA,
            "val_split": VAL_SPLIT,
        },
    )

    print("Initializing VAE...")
    vae = SimpleVAE().to("cuda")
    loss_fn = VAELoss(beta=BETA).to("cuda")
    optimizer = torch.optim.Adam(
        [{"params": vae.parameters()}],
        lr=LEARNING_RATE,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, verbose=True
    )

    # Training
    print("Training...")

    for epoch in range(EPOCHS):
        vae.train()
        avg_train_loss = 0
        avg_train_reconstruction_loss = 0
        avg_train_kl_loss = 0
        for i, (x, sample_rate, text_label, audio_id) in enumerate(esc50_train_loader):
            # conver to batch of mel-spectrograms
            print(f"Epoch {epoch} | Batch {i+1}/{len(esc50_train_loader)}")
            spectrograms = get_mel_spectrogram(x, sample_rate[0]).to("cuda")

            optimizer.zero_grad()
            z, mean, log_var = vae.encode(spectrograms)
            x_hat = vae.decode(z)

            loss, reconstruction_loss, kl_loss = loss_fn(
                spectrograms, x_hat, mean, log_var
            )
            loss.backward()
            optimizer.step()

            avg_train_loss += loss.item()
            avg_train_reconstruction_loss += reconstruction_loss.item()
            avg_train_kl_loss += kl_loss.item()

        avg_train_loss /= len(esc50_train_loader)
        avg_train_reconstruction_loss /= len(esc50_train_loader)
        avg_train_kl_loss /= len(esc50_train_loader)

        print(
            f"Epoch {epoch} | Train Loss: {avg_train_loss} | Train Reconstruction Loss: {avg_train_reconstruction_loss} | Train KL Loss: {avg_train_kl_loss}"
        )

        # Validation
        vae.eval()
        avg_val_loss = 0
        avg_val_reconstruction_loss = 0
        avg_val_kl_loss = 0
        with torch.no_grad():
            for i, (x, sample_rate, text_label, audio_id) in enumerate(
                esc50_val_loader
            ):
                print(f"Epoch {epoch} | Batch {i+1}/{len(esc50_val_loader)}")
                spectrograms = get_mel_spectrogram(x, sample_rate[0]).to("cuda")

                z, mean, log_var = vae.encode(spectrograms)
                x_hat = vae.decode(z)

                loss, reconstruction_loss, kl_loss = loss_fn(
                    spectrograms, x_hat, mean, log_var
                )
                avg_val_loss += loss.item()
                avg_val_reconstruction_loss += reconstruction_loss.item()
                avg_val_kl_loss += kl_loss.item()

                # log the first image from the first val batch per epoch
                if i == 0:
                    f_max = config.fmax
                    # Assuming `original` and `reconstructed` are your dB Mel spectrograms
                    # Plot the original spectrogram
                    fig1, ax1 = plt.subplots(figsize=(10, 4))
                    librosa.display.specshow(
                        spectrograms[0].detach().cpu().squeeze(0).numpy(),
                        x_axis="time",
                        y_axis="mel",
                        sr=sample_rate,
                        fmax=f_max,
                        ax=ax1,
                    )
                    plt.colorbar(format="%+2.0f dB", ax=ax1)
                    plt.title("Original Spectrogram")

                    # Plot the reconstructed spectrogram
                    fig2, ax2 = plt.subplots(figsize=(10, 4))
                    librosa.display.specshow(
                        x_hat[0].detach().cpu().numpy(),
                        x_axis="time",
                        y_axis="mel",
                        sr=sample_rate,
                        fmax=f_max,
                        ax=ax2,
                    )
                    plt.colorbar(format="%+2.0f dB", ax=ax2)
                    plt.title("Reconstructed Spectrogram")

                    wandb.log(
                        {
                            "input": wandb.Image(
                                fig1,
                                caption="Input dB Mel-Spectrogram",
                            ),
                            "reconstruction": wandb.Image(
                                fig2,
                                caption="Reconstructed dB Mel-Spectrogram",
                            ),
                        },
                        step=epoch,
                    )

                    # Close the plots to free up memory
                    plt.close(fig1)
                    plt.close(fig2)

        avg_val_loss /= len(esc50_val_loader)
        avg_val_reconstruction_loss /= len(esc50_val_loader)
        avg_val_kl_loss /= len(esc50_val_loader)

        print(
            f"Epoch {epoch} | Val Loss: {avg_val_loss} | Val Reconstruction Loss: {avg_val_reconstruction_loss} | Val KL Loss: {avg_val_kl_loss}"
        )

        wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss}, step=epoch)
        wandb.log(
            {
                "train_reconstruction_loss": avg_train_reconstruction_loss,
                "val_reconstruction_loss": avg_val_reconstruction_loss,
            },
            step=epoch,
        )
        wandb.log(
            {"train_kl_loss": avg_train_kl_loss, "val_kl_loss": avg_val_kl_loss},
            step=epoch,
        )

        scheduler.step(avg_val_loss)
