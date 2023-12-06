import torch
import torch.nn as nn
import torch.nn.functional as F

from models import clap
from utils import loaders

# import wandb
# wandb.login()

if __name__ == "__main__":
    # Load data
    esc50_loader = loaders.ESC50Loader("/ESC-50-master")
    esc50_loader = torch.utils.data.DataLoader(
        esc50_loader, batch_size=16, shuffle=True
    )

    # Load model
    model = clap.CLAP()
    model = model.to("cuda")
    loss_fn = clap.ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, verbose=True
    )

    # Train
    for epoch in range(100):
        for batch in esc50_loader:
            waveform, sample_rate, text_label = batch
            waveform = waveform.to("cuda")
            text_label = text_label.to("cuda")

            audio_embeddings, text_embeddings = model(waveform, text_label)
            loss = loss_fn(audio_embeddings, text_embeddings)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step(loss)
        print("Epoch: {} | Loss: {:.5f}".format(epoch, loss.item()))
