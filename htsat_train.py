"""
File for pretraining HTSAT audio encoder. Procedure follows simCLR's paper 
on contrastive learning. 
"""
from torch.utils.data import random_split
import wandb
import torch
from utils import loaders
from models.htsat import HTSAT_Swin_Transformer as HTSATAudioEncoder
from models import config
from models import clap
from utils.visualization import generate_umap
from utils.transforms import AudioAugmentations
import torch.nn.functional as F
from joblib import Parallel, delayed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 220500


def fix_length(waveform, target_length=MAX_LENGTH):
    """
    Adjust the waveform to a target length by either padding with zeros or truncating.
    """
    current_length = len(waveform)
    if current_length > target_length:
        # Truncate the waveform
        waveform = waveform[:target_length]
    elif current_length < target_length:
        # Pad the waveform with zeros
        padding = target_length - current_length
        waveform = F.pad(waveform, (0, padding))
    return waveform


def process_waveform(waveform, sample_rate):
    augmentor = AudioAugmentations(sample_rate=sample_rate)
    aug_1, aug_2 = augmentor.random_transforms(waveform)
    return fix_length(aug_1).unsqueeze(0), fix_length(aug_2).unsqueeze(0)


def split_data(data, val_split=0.2):
    val_size = int(len(data) * val_split)
    train_size = len(data) - val_size
    return random_split(data, [train_size, val_size])


if __name__ == "__main__":
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.01
    epochs = 20
    frozen = False
    val_split = 0.1
    init_temp = 2.1
    use_amp = False

    wandb.login()
    wandb.init(
        # set the wandb project where this run will be logged
        project="resonance",
        group="pretraining_audio",
        # track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "architecture": "HTSAT",
            "dataset": "ESC-50",
            "epochs": epochs,
            "batch_size": batch_size,
            "val_split": val_split,
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

    print("Loading model...")
    audio_encoder = HTSATAudioEncoder(config=config).to(device)
    loss_fn = clap.ContrastiveLoss(init_temp=init_temp).to(device)
    optimizer = torch.optim.Adam(
        [{"params": audio_encoder.parameters()}, {"params": loss_fn.t}],
        lr=learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, verbose=True
    )

    # Training
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print("Training...")
    for epoch in range(1, epochs + 1):
        # Training mode
        audio_encoder.train()
        train_loss = 0.0
        for batch_index, batch in enumerate(esc50_train_loader, 1):
            waveforms, sample_rates, text_labels, _ = batch
            # if waveforms is a single vector, increase dims
            if len(waveforms.shape) == 1:
                waveforms = waveforms.unsqueeze(0)

            # augmentations
            processed_waveforms = Parallel(n_jobs=-1)(
                delayed(process_waveform)(waveform, sample_rate)
                for waveform, sample_rate in zip(waveforms, sample_rates)
            )
            augmentations_1, augmentations_2 = zip(*processed_waveforms)

            augmentations_1_torch = torch.cat(augmentations_1, dim=0).to(device)
            augmentations_2_torch = torch.cat(augmentations_2, dim=0).to(device)

            optimizer.zero_grad()
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=use_amp
            ):
                aug_1_embeddings = audio_encoder(augmentations_1_torch)["latent_output"]
                aug_2_embeddings = audio_encoder(augmentations_2_torch)["latent_output"]
                loss = loss_fn(aug_1_embeddings, aug_2_embeddings)

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
        audio_encoder.eval()
        avg_val_loss = 0.0
        for batch_index, batch in enumerate(esc50_val_loader, 1):
            waveforms, sample_rates, text_labels, audio_ids = batch

            # if waveforms is a single vector, increase dims
            if len(waveforms.shape) == 1:
                waveforms = waveforms.unsqueeze(0)

            # augmentations
            processed_waveforms = Parallel(n_jobs=-1)(
                delayed(process_waveform)(waveform, sample_rate)
                for waveform, sample_rate in zip(waveforms, sample_rates)
            )
            augmentations_1, augmentations_2 = zip(*processed_waveforms)

            augmentations_1_torch = torch.cat(augmentations_1, dim=0).to(device)
            augmentations_2_torch = torch.cat(augmentations_2, dim=0).to(device)

            with torch.no_grad():
                aug_1_embeddings = audio_encoder(augmentations_1_torch)["latent_output"]
                aug_2_embeddings = audio_encoder(augmentations_2_torch)["latent_output"]
                val_loss = loss_fn(aug_1_embeddings, aug_2_embeddings)

            avg_val_loss += val_loss.item()

            print(
                f"Val | Epoch: {epoch} | Batch: {batch_index}/{len(esc50_val_loader)} | temperature: {loss_fn.t.item():.5f}"
            )

            if batch_index == 1:
                with torch.no_grad():
                    audio_embeddings = audio_encoder(waveforms)["latent_output"]
                    audio_embeddings_normalized = F.normalize(
                        audio_embeddings, p=2, dim=1
                    )
                # Generate UMAP visualization
                fig = generate_umap(
                    audio_embeddings_normalized.cpu().numpy(),
                    None,
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
            torch.save(audio_encoder.state_dict(), f"./audio_epoch_{epoch}.h5")
            # wandb.save(f"./audio_epoch_{epoch}.h5")

        scheduler.step(avg_val_loss)
