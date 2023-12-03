import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.audio_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.audio_model = AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to("cuda")

        # Remove the classifier head
        self.audio_model.classifier = nn.Identity()

        # Add custom FC layers
        self.fc1 = nn.Linear(768, 768)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(768, 768)

    def forward(self, audio):
        # Pass input through the audio model (excluding the classifier head)
        inputs = self.audio_extractor(audio, sampling_rate=16000, return_tensors="pt").to(device)
        out = self.audio_model(
            **inputs
        )

        x = self.fc1(out.logits)
        x = self.relu1(x)
        x = self.fc2(x)

        return x