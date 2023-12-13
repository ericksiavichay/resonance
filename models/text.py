from sentence_transformers import SentenceTransformer
from torch import nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    def __init__(self, freeze_base=False):
        super().__init__()

        self.model_base = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")
        self.tokenizer_base = self.model_base.tokenizer
        self.fc_1 = nn.Linear(self.model_base.get_sentence_embedding_dimension(), 768)
        self.relu_1 = nn.ReLU()
        self.fc_2 = nn.Linear(768, 768)

        if freeze_base:
            for param in self.model_base.parameters():
                param.requires_grad = False

    def forward(self, text):
        tokenizer_encoding = self.tokenizer_base(
            text, return_tensors="pt", padding=True, truncation=True
        ).to("cuda")
        out = self.model_base(tokenizer_encoding).sentence_embedding
        out = self.relu_1(self.fc_1(out))
        out = self.fc_2(out)
        return out

    def embed(self, text):
        tokenizer_encoding = self.tokenizer_base(
            text, return_tensors="pt", padding=True, truncation=True
        ).to("cuda")
        out = self.model_base(tokenizer_encoding).sentence_embedding
        out = self.fc_1(out)
        out = F.normalize(out, p=2, dim=1)
        return out
