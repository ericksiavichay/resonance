# from train import ContrastiveLoss
# import torch
import numpy as np
from utils.youtube import download_video_as_wav
from utils.loaders import MusicCapsLoader
from utils.visualization import generate_umap

if __name__ == "__main__":
    # loss_fn = ContrastiveLoss()
    # audio_embeddings = torch.randn(16, 64)
    # text_embeddings = torch.randn(16, 64)
    # loss = loss_fn(audio_embeddings, text_embeddings)
    # print(loss)

    # id = "-v5hgCh3M2w"
    # start_sec = 30
    # end_sec = 40
    # download_video_as_wav(id, start_sec, end_sec, ".")

    # musiccaps = MusicCapsLoader("./MusicCaps")
    # print(len(musiccaps))
    # print(musiccaps[0])

    audio_embeddings = np.random.rand(16, 64)
    text_embeddings = np.random.rand(16, 64)

    # 16 random audio IDs
    audio_ids = [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
    ]

    # 16 random text labels
    text_labels = [
        "cat",
        "dog",
        "bird",
        "fish",
        "cat",
        "dog",
        "bird",
        "fish",
        "cat",
        "dog",
        "bird",
        "fish",
        "cat",
        "dog",
        "bird",
        "fish",
    ]

    generate_umap(audio_embeddings, text_embeddings, text_labels, audio_ids)
