import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from models import clap
from utils import loaders
from utils.visualization import generate_umap
from torch.utils.data import random_split

import wandb


def split_data(data, val_split=0.2):
    val_size = int(len(data) * val_split)
    train_size = len(data) - val_size
    return random_split(data, [train_size, val_size])


if __name__ == "__main__":
    # Hyperparameters
    batch_size = 126
    learning_rate = 0.0001
    epochs = 20
    frozen = True
    val_split = 0.3
    use_amp = False
    init_temp = 0.5

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
            "use_amp": use_amp,
            "init_temp": init_temp,
        },
    )
    print("Loading data...")
    esc50 = loaders.ESC50Loader("./ESC-50-master/ESC-50-master/")
    esc50_train, esc50_val = split_data(esc50, val_split=val_split)
    esc50_train_loader = torch.utils.data.DataLoader(
        esc50_train, batch_size=batch_size, shuffle=True, num_workers=0
    )
    esc50_val_loader = torch.utils.data.DataLoader(
        esc50_val, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print("Initializing model...")
    model = clap.CLAP(freeze_base=frozen)
    model = model.to("cuda")
    loss_fn = clap.ContrastiveLoss(init_temp=init_temp).to("cuda")
    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": loss_fn.t}], lr=learning_rate
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, verbose=True
    )

    # Train
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print("Training...")
    for epoch in range(1, epochs + 1):
        # Training mode
        model.train()
        train_loss = 0.0
        for batch_index, batch in enumerate(esc50_train_loader, 1):
            waveforms, sample_rates, text_labels, _ = batch
            waveforms = waveforms.to("cuda")

            # if waveforms is a single vector, increase dims
            if len(waveforms.shape) == 1:
                waveforms = waveforms.unsqueeze(0)

            optimizer.zero_grad()
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
            train_loss += loss.item()
            print(
                f"Train | Epoch: {epoch} | Batch: {batch_index}/{len(esc50_train_loader)} | temperature: {loss_fn.t.item():.5f}"
            )

        avg_train_loss = train_loss / len(esc50_train_loader)
        print(f"Train | Epoch: {epoch} | Train Loss: {avg_train_loss:.5f}")

        # Evaluation mode
        model.eval()
        avg_val_loss = 0.0
        for batch_index, batch in enumerate(esc50_val_loader, 1):
            waveforms, sample_rates, text_labels, audio_ids = batch
            waveforms = waveforms.to("cuda")

            # if waveforms is a single vector, increase dims
            if len(waveforms.shape) == 1:
                waveforms = waveforms.unsqueeze(0)

            with torch.no_grad():
                audio_embeddings, text_embeddings = model(waveforms, text_labels)
                val_loss = loss_fn(audio_embeddings, text_embeddings)

            avg_val_loss += val_loss.item()

            print(
                f"Val | Epoch: {epoch} | Batch: {batch_index}/{len(esc50_val_loader)} | temperature: {loss_fn.t.item():.5f}"
            )

            if batch_index == 1:
                with torch.no_grad():
                    actual_text_embeddings = model.text_encoder.embed(text_labels)

                audio_embeddings = F.normalize(audio_embeddings, p=2, dim=1)
                text_embeddings = F.normalize(actual_text_embeddings, p=2, dim=1)
                # Generate UMAP visualization
                fig = generate_umap(
                    audio_embeddings.cpu().numpy(),
                    text_embeddings.cpu().numpy(),
                    text_labels,
                    audio_ids,
                )

                # Log the figure to wandb
                wandb.log(
                    {"embeddings_plot": wandb.Html(fig.to_html(full_html=False))},
                    step=epoch,
                )

        avg_val_loss = avg_val_loss / len(esc50_val_loader)
        print(f"Val | Epoch: {epoch} | Val Loss: {avg_val_loss:.5f}")

        wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss}, step=epoch)

        # only save every 10 epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"./model_frozen_{frozen}_epoch_{epoch}.h5")
            wandb.save(f"./model_frozen_{frozen}_epoch_{epoch}.h5")

        scheduler.step(avg_val_loss)
