from sentence_transformers import SentenceTransformer
from torch import nn


class TextEncoder(nn.Module):
    def __init__(self, freeze_base=True, device="cuda"):
        super().__init__()

        self.device = device

        self.model_base = SentenceTransformer("BAAI/bge-large-en-v1.5")
        self.tokenizer_base = self.model_base.tokenizer
        self.fc_1 = nn.Linear(self.model_base.get_sentence_embedding_dimension(), 1024)
        self.relu_1 = nn.ReLU()
        self.fc_2 = nn.Linear(1024, 1024)

        if freeze_base:
            for param in self.model_base.parameters():
                param.requires_grad = False

    def forward(self, text):
        tokenizer_encoding = self.tokenizer_base(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        out = self.model_base(tokenizer_encoding).to(self.device)
        out = self.relu_1(self.fc_1(out))
        out = self.fc_2(out)
        return out

