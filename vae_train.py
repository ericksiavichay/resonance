"""
Main file for training the VAE model
"""

import torch
import torch.nn as nn
import torchaudio.transforms as T
import torch.functional as F

from utils import loaders
import models.config as config
from models.vae import Encoder, Decoder

# HYPERPARAMETERS
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 10
BETA = 1.0
VAL_SPLIT = 0.3


def split_data(data, val_split=0.2):
    val_size = int(len(data) * val_split)
    train_size = len(data) - val_size
    return random_split(data, [train_size, val_size])


def get_spectrogram(
    waveform,
    n_fft=config.window_size,
    hop_length=config.hop_size,
    win_length=config.window_size,
    window=torch.hann_window,
    center=True,
    pad_mode="reflect",
):
    specgram = T.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )

    return specgram(waveform)


class VAELoss(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()

        self.beta = beta

    def forward(self, x, x_hat, mean, log_var):
        """
        x: (B, C, H, W)
        x_hat: (B, C, H, W)
        mean: (B, C, H, W)
        log_var: (B, C, H, W)
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
    esc50_train, esc50_val = split_data(esc50, val_split=val_split)
    esc50_train_loader = torch.utils.data.DataLoader(
        esc50_train, batch_size=batch_size, shuffle=True, num_workers=0
    )
    esc50_val_loader = torch.utils.data.DataLoader(
        esc50_val, batch_size=batch_size, shuffle=True, num_workers=0
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
    encoder = Encoder().to("cuda")
    decoder = Decoder().to("cuda")
    loss_fn = VAELoss(beta=BETA).to("cuda")
    optimizer = torch.optim.Adam(
        [{"params": encoder.parameters()}, {"params": decoder.parameters()}],
        lr=LEARNING_RATE,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, verbose=True
    )

    # Training
    print("Training...")

    for epoch in range(EPOCHS):
        encoder.train()
        decoder.train()
        avg_train_loss = 0
        avg_train_reconstruction_loss = 0
        avg_train_kl_loss = 0
        for i, (x, _, _, _) in enumerate(esc50_train_loader):
            # conver to batch of mel-spectrograms
            print(f"Epoch {epoch} | Batch {i+1}/{len(esc50_train_loader)}")
            spectrograms = get_spectrogram(x).to("cuda")
            H, W = spectrograms.shape[-2:]
            random_noise = torch.rand(spectrograms.shape[0], H // 8, W // 8).to("cuda")

            optimizer.zero_grad()
            z, mean, log_var = encoder(spectrograms, noise=random_noise)
            x_hat = decoder(z)

            loss, reconstruction_loss, kl_loss = loss_fn(x, x_hat, mean, log_var)
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
        encoder.eval()
        decoder.eval()
        avg_val_loss = 0
        avg_val_reconstruction_loss = 0
        avg_val_kl_loss = 0
        with torch.no_grad():
            for i, (x, _, _, _) in enumerate(esc50_val_loader):
                print(f"Epoch {epoch} | Batch {i+1}/{len(esc50_val_loader)}")
                spectrograms = get_spectrogram(x).to("cuda")
                H, W = spectrograms.shape[-2:]
                random_noise = torch.rand(spectrograms.shape[0], H // 8, W // 8).to(
                    "cuda"
                )

                z, mean, log_var = encoder(spectrograms, noise=random_noise)
                x_hat = decoder(z)

                loss, reconstruction_loss, kl_loss = loss_fn(x, x_hat, mean, log_var)
                avg_val_loss += loss.item()
                avg_val_reconstruction_loss += reconstruction_loss.item()
                avg_val_kl_loss += kl_loss.item()

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
