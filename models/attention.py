import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Optional

# implementation from https://nn.labml.ai/diffusion/stable_diffusion/model/unet_attention.html


class CrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_cond: int,
        n_heads: int,
        # d_head: int,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_cond = d_cond
        self.n_heads = n_heads

        # Multihead attention layer
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True
        )

        # Projection for conditional input
        self.cond_proj = nn.Linear(d_cond, d_model)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None):
        if cond is not None:
            # Project cond to match d_model, and repeat it to match x's sequence length
            cond = self.cond_proj(cond)
            cond = cond.repeat(1, x.size(1), 1)  # Repeat along sequence length

        # Multi-head attention
        # If cond is None, it will be ignored in multihead attention
        attn_output, _ = self.multihead_attn(
            query=x, key=cond, value=cond, need_weights=False
        )
        return attn_output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_mult=4):
        super().__init__()
        self.net = nn.Sequential(
            GeGLU(d_model, d_model * d_mult),
            # nn.Dropout(0.),
            nn.Linear(d_model * d_mult, d_model),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class GeGLU(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        self.proj = nn.Linear(d_in, d_out * 2)

    def forward(self, x: torch.Tensor):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_cond: int,
    ):
        super().__init__()

        self.attn1 = CrossAttention(d_model, d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)

        self.attn2 = CrossAttention(d_model, d_cond, n_heads)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = FeedForward(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), cond=cond) + x
        x = self.ff(self.norm3(x)) + x

        return x


class SpatialTransformer(nn.Module):
    def __init__(
        self,
        channels: int,
        n_heads: int,
        n_layers: int,
        d_cond: int,
    ):
        """
        Following from: https://nn.labml.ai/diffusion/stable_diffusion/model/unet_attention.html
        """
        super().__init__()
        self.norm = nn.GroupNorm(
            num_groups=32, num_channels=channels, eps=1e-6, affine=True
        )  # Group norm paper sets G = 32, could be a hyperparameter
        self.proj_in = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(channels, n_heads, d_cond=d_cond)
                for _ in range(n_layers)
            ]
        )
        self.proj_out = nn.Conv2d(
            channels, channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None):
        b, c, h, w = x.shape
        x_in = x
        x = self.proj_in(self.norm(x))
        x = x.permute(0, 2, 3, 1).view(b, h * w, c)

        for block in self.transformer_blocks:
            x = block(x, cond)

        x = x.view(b, h, w, c).permute(0, 3, 1, 2)
        x = self.proj_out(x)
        return x + x_in
