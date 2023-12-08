import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from models import clap
from utils import loaders
from torch.utils.data import random_split

import wandb


def split_data(data, val_split=0.2):
    val_size = int(len(data) * val_split)
    train_size = len(data) - val_size
    return random_split(data, [train_size, val_size])


if __name__ == "__main__":
    # Hyperparameters
    batch_size = 20
    learning_rate = 0.0001
    epochs = 100
    frozen = True
    val_split = 0.2

    wandb.login()
    wandb.init(
        # set the wandb project where this run will be logged
        project="resonance",
        # track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "architecture": "a:HTSAT, t:BAAI/bge-large-en-v1.5",
            "dataset": "ESC-50",
            "epochs": epochs,
            "text_encoder_base_frozen": frozen,
            "batch_size": batch_size,
            "val_split": val_split,
        },
    )
    print("Loading data...")
    esc50 = loaders.ESC50Loader("./ESC-50-master/ESC-50-master/")
    esc50_train, esc50_val = split_data(esc50, val_split=val_split)
    esc50_train_loader = torch.utils.data.DataLoader(
        esc50_train, batch_size=batch_size, shuffle=True, num_workers=0
    )
    esc50_val_loader = torch.utils.data.DataLoader(
        esc50_val, batch_size=batch_size * 3, shuffle=True, num_workers=0
    )

    print("Initializing model...")
    model = clap.CLAP(freeze_base=frozen)
    model = model.to("cuda")
    loss_fn = clap.ContrastiveLoss().to("cuda")
    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": loss_fn.t}], lr=learning_rate
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, verbose=True
    )

    # Train
    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print("Training...")
    for epoch in range(epochs, 1):
        # Training mode
        model.train()
        for batch_index, batch in enumerate(esc50_train_loader, 1):
            waveforms, sample_rates, text_labels = batch
            waveforms = waveforms.numpy()

            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=use_amp
            ):
                audio_embeddings, text_embeddings = model(waveforms, text_labels)
                loss = loss_fn(audio_embeddings, text_embeddings)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # loss.backward()
            # optimizer.step()
            optimizer.zero_grad()
            print(
                f"Epoch: {epoch} | Train Batch: {batch_index}/{len(esc50_train_loader)} | Train Loss: {loss.item():.5f} | temperature: {loss_fn.t.item():.5f}"
            )
            wandb.log({"loss": loss.item()})

        # Evaluation mode
        model.eval()
        for batch_index, batch in enumerate(esc50_val_loader, 1):
            waveforms, sample_rates, text_labels = batch
            waveforms = waveforms.numpy()

            with torch.no_grad():
                audio_embeddings, text_embeddings = model(waveforms, text_labels)
                val_loss = loss_fn(audio_embeddings, text_embeddings)

            print(
                f"Epoch: {epoch} | Val Batch: {batch_index}/{len(esc50_val_loader)} | Val Loss: {val_loss.item():.5f} | temperature: {loss_fn.t.item():.5f}"
            )
            wandb.log({"val_loss": val_loss.item()})

        torch.save(model.state_dict(), f"./model_frozen_{frozen}_epoch_{epoch}.h5")
        wandb.save(f"./model_frozen_{frozen}_epoch_{epoch}.h5")

        scheduler.step(loss)
