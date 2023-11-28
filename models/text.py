from sentence_transformers import SentenceTransformer
from torch import nn


class TextEncoder(nn.Module):
    def __init__(self, freeze_base=True):
        super().__init__()

        if freeze_base:
            for param in self.model_base.parameters():
                param.requires_grad = False

        self.model_base = SentenceTransformer("BAAI/bge-large-en-v1.5")
        self.fc_1 = nn.Linear(self.model_base.config.hidden_size, 1024)
        self.relu_1 = nn.ReLU()
        self.fc_2 = nn.Linear(1024, 1024)

    def forward(self, text):
        out = self.model_base(text)
        out = self.relu_1(self.fc_1(out))
        out = self.fc_2(out)
        return out

