import torch.nn as nn
import torch.nn.functional as F
import torch
from models.audio import AudioEncoder
from models.text import TextEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ContrastiveLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t = nn.Parameter(torch.tensor(0.7))

    def forward(self, audio_embeddings, text_embeddings):
        # Normalize embeddings
        audio_embeddings = F.normalize(audio_embeddings, p=2, dim=1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)

        similarity_matrix = torch.matmul(
            audio_embeddings, text_embeddings.T
        ) * torch.exp(self.t)

        num_batches = audio_embeddings.shape[0]
        labels = torch.arange(num_batches).to(device)
        loss_audio = F.cross_entropy(similarity_matrix, labels)
        loss_text = F.cross_entropy(similarity_matrix.T, labels)
        loss = (loss_audio + loss_text) / 2
        return loss


class CLAP(nn.Module):
    def __init__(self):
        super(CLAP, self).__init__()
        self.audio_encoder = AudioEncoder()
        self.text_encoder = TextEncoder()

    def forward(self, audio, text):
        """
        Audio and text are batched.
        """
        audio_embeddings = self.audio_encoder(audio)
        text_embeddings = self.text_encoder(text)

        return audio_embeddings, text_embeddings
