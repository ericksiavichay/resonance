import torch.nn as nn
import torch.nn.functional as F
import torch
from models.audio import AudioEncoder
from models.htsat import HTSAT_Swin_Transformer as HTSATAudioEncoder
from models.text import TextEncoder
from models import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, init_temp=0.07):
        super().__init__()
        self.init_temp = init_temp
        self.t = nn.Parameter(torch.tensor(self.init_temp))

    def forward(self, audio_embeddings, text_embeddings):
        # Normalize embeddings
        audio_embeddings = F.normalize(audio_embeddings, p=2, dim=1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)

        similarity_matrix = torch.matmul(
            audio_embeddings, text_embeddings.T
        ) * torch.exp(self.t)

        batch_size = audio_embeddings.shape[0]
        labels = torch.arange(batch_size).to(device)
        loss_audio = F.cross_entropy(similarity_matrix, labels)
        loss_text = F.cross_entropy(similarity_matrix.T, labels)
        loss = (loss_audio + loss_text) / 2
        return loss


# class CLAP(nn.Module):
#     def __init__(self, freeze_base=False):
#         super(CLAP, self).__init__()
#         self.audio_encoder = AudioEncoder(freeze_base=freeze_base).to(device)
#         self.text_encoder = TextEncoder(freeze_base=freeze_base).to(device)

#     def forward(self, audio, text):
#         """
#         Audio and text are batched.
#         """
#         audio_embeddings = self.audio_encoder(audio)
#         text_embeddings = self.text_encoder(text)

#         return audio_embeddings, text_embeddings


class CLAP(nn.Module):
    def __init__(self, freeze_base=False):
        super(CLAP, self).__init__()
        self.audio_encoder = HTSATAudioEncoder(config=config).to(device)
        self.text_encoder = TextEncoder(freeze_base=freeze_base).to(device)

    def forward(self, audio, text):
        """
        Audio and text are batched.
        embeddings are d = 768
        """
        audio_embeddings = self.audio_encoder(audio)["latent_output"]
        text_embeddings = self.text_encoder(text)

        return audio_embeddings, text_embeddings
