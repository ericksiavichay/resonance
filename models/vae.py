"""
Variational autoencoder
"""

import torch
from torch import nn
from torch.nn import functional as F
import math


class LinearVAE(nn.Module):
    """
    A simple VAE with linear layers
    """

    def __init__(self, input_dim, latent_dim=768):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim[0] * input_dim[1], 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
        )

        self.mean = nn.Linear(1024, latent_dim)
        self.log_var = nn.Linear(1024, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, input_dim[0] * input_dim[1]),
            nn.Unflatten(1, (1, input_dim[0], input_dim[1])),
        )

    def reparameterize(self, mean, log_var):
        """
        Reparameterization trick for VAE
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mean

    def encode(self, x):
        x = self.encoder(x)

        mean = self.mean(x)
        log_var = self.log_var(x)
        z = self.reparameterize(mean, log_var)

        return z, mean, log_var

    def decode(self, z):
        return self.decoder(z)


class SimpleVAE(nn.Module):
    """
    Simple VAE with seperate linear layers for mean and log variance. Also uses
    skip connections in the decoder.
    """

    def __init__(self, latent_dim=768):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # these will dynamically change based on the input shape
        self.mean = None
        self.log_var = None

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512 * 1 * 6),
            nn.ReLU(),
            nn.Unflatten(1, (512, 1, 6)),
            nn.Upsample(scale_factor=2, mode="nearest"),  # First upsampling
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),  # Second upsampling
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),  # Third upsampling
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),  # Fourth upsampling
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(
                scale_factor=2, mode="nearest"
            ),  # Optional: Adjust or remove based on desired output size
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(
                scale_factor=2, mode="nearest"
            ),  # Optional: Adjust or remove based on desired output size
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

    def reparameterize(self, mean, log_var):
        """
        Reparameterization trick for VAE
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mean

    def encode(self, x):
        x = self.encoder(x)

        if self.mean is None or self.log_var is None:
            self.mean = nn.Linear(x.shape[1], 768).to("cuda")
            self.log_var = nn.Linear(x.shape[1], 768).to("cuda")

        mean = self.mean(x)
        log_var = self.log_var(x)
        z = self.reparameterize(mean, log_var)

        return z, mean, log_var

    def decode(self, z):
        return self.decoder(z)


class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, causal_mask=False):
        """
        x: (B, seq_len, dim)

        return: (B, seq_len, dim)
        """

        input_shape = x.shape

        B, seq_len, d_embed = input_shape

        interim_shape = (B, seq_len, self.n_heads, self.d_head)

        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        out = weight @ v
        out = out.transpose(1, 2)
        out = out.reshape(input_shape)

        out = self.out_proj(out)

        return out


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x):
        """
        x: (B, channels, height, width)
        """

        residual = x

        n, c, h, w = x.shape
        x = x.view(n, c, h * w)

        x = x.transpose(-1, -2)
        x = self.attention(x)
        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))
        x += residual

        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0
            )

    def forward(self, x):
        """
        x: (B, in_channels, H, W)
        """

        residual = x

        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residual)


class Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(1, 128, kernel_size=3, padding=1),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            ResidualBlock(256, 512),
            ResidualBlock(512, 512),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            AttentionBlock(512),
            ResidualBlock(512, 512),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )

    def forward(self, x, noise):
        """
        x: (B, C, H, W)
        noise: (B, C_out, H/8, W/8)
        """

        for module in self:
            if getattr(module, "stride", None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))

            x = module(x)

        mean, log_var = torch.chunk(x, 2, dim=1)
        log_var = torch.clamp(log_var, -30, 20)
        var = log_var.exp()
        std = var.sqrt()

        x = mean + std * noise
        x *= 0.18215  # no idea why we scale it. it is commonly used in many VAEs and other diffusion based models

        return x, mean, log_var


class Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 512),
            AttentionBlock(512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            ResidualBlock(256, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        """
        input is our latent
        """

        x /= 0.18215

        for module in self:
            x = module(x)

        return x
