import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models import clap
from utils import loaders

import wandb


if __name__ == "__main__":
    # Hyperparameters
    batch_size = 20
    learning_rate = 0.0001
    epochs = 100

    wandb.login()
    wandb.init(
        # set the wandb project where this run will be logged
        project="resonance",
        # track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "architecture": "CLAP",
            "dataset": "ESC-50",
            "epochs": epochs,
        },
    )
    print("Loading data...")
    esc50_loader = loaders.ESC50Loader("./ESC-50-master/ESC-50-master/")
    esc50_loader = torch.utils.data.DataLoader(
        esc50_loader, batch_size=batch_size, shuffle=True, num_workers=-1
    )

    print("Initializing model...")
    model = clap.CLAP(freeze_base=True)
    model = model.to("cuda")
    loss_fn = clap.ContrastiveLoss().to("cuda")
    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": loss_fn.t}], lr=learning_rate
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, verbose=True
    )

    # Train
    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print("Training...")
    for epoch in range(epochs):
        for batch_index, batch in enumerate(esc50_loader, 1):
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
                f"Epoch: {epoch} | Batch: {batch_index}/{len(esc50_loader)} | Loss: {loss.item():.5f} | temperature: {loss_fn.t.item():.5f}"
            )
        wandb.log({"loss": loss.item()}, step=epoch)

        scheduler.step(loss)
