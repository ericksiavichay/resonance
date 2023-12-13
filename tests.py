# from train import ContrastiveLoss
# import torch
from utils.youtube import download_video_as_wav
from utils.loaders import MusicCapsLoader

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

    musiccaps = MusicCapsLoader("./MusicCaps")
    print(len(musiccaps))
    print(musiccaps[0])
