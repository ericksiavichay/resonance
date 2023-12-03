from train import ContrastiveLoss
import torch

if __name__ == "__main__":
    loss_fn = ContrastiveLoss()
    audio_embeddings = torch.randn(16, 64)
    text_embeddings = torch.randn(16, 64)
    loss = loss_fn(audio_embeddings, text_embeddings)
    print(loss)