"""
Contains code for U-Net architecture based on Stable Diffusion for generative purposes

This model will be trained to estimate the noise added to some image at some time step, 

model = UNet(in_channels=(3,5,6,767,), out_channels=(34,434,6,45))

Based on: https://nn.labml.ai/diffusion/stable_diffusion/model/unet.html

"""

import torch.nn as nn
import torch.nn.functional as F
import torch
from models.attention import SpatialTransformer


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels):
    return GroupNorm32(32, channels)


class UpSample(nn.Module):
    def __init__(self, channels):
        super().__init__(self, channels)

        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")

        return self.conv(x)


class DownSample(nn.Module):
    def __init__self(self, channels):
        super().__init__()

        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class ResBlock(nn.Module):
    def __init__(self, channels, d_t_embed, *, out_channels=None):
        super().__init__()

        if out_channels is None:
            out_channels = channels

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )

        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(d_t_embed, out_channels))

        self.out_layers = nn.Sequential(
            normalization(out_channels),
            nn.SiLU(),
            # nn.Dropout(0.),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        if out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, out_channels, 1)

    def forward(self, x, t_embed):
        h = self.in_layers(x)
        t_embed = self.emb_layers(t_embed).type(h.dtype)
        h = h + t_embed[:, :, None, None]
        h = self.out_layers(h)

        return self.skip_connection(x) + h


class TimestepEmbedSequential(nn.Sequential):
    def forward(self, x, t_embed, cond=None):
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, t_embed)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, cond)
            else:
                x = layer(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels,
        int,
        channels: int,
        n_res_blocks: int,
        attention_levels: List[int],
        channel_multipliers: List[int],
        n_heads: int,
        tf_layers: int = 1,
        d_cond: int = 768
    ):
        """
        Args
        in_channels: the dim of the channels in the input
        out_channels: the dim of the channels in the output
        channels: the base dimmension c_u as shown in AudioLDM
        n_res_blocks: number of residual blocks at each level
        attention_levels: levels at which attention should be performed
        channel_multipliers: the coefficient of c_u at each decoder level as shown in AudioLDM
        n_heads: the number of attention heads
        tf_layers: the number of transformer layers in the transformers
        d_cond: the dimmension of the conditional vector (text embedding)
        """
        super().__init__()
        self.channels = channels

        levels = len(channel_multipliers)

        d_time_embed = channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(channels, d_time_embed),
            nn.SiLU(),
            nn.Linear(d_time_embed, d_time_embed),
        )

        self.input_blocks = nn.ModuleList()

        self.input_blocks.append(
            TimestepEmbedSequential(nn.Conv2d(in_channels, channels, 3, padding=1))
        )

        input_block_channels = [channels]
        channels_list = [channels * m for m in channel_multipliers]

        for i in range(levels):
            for _ in range(n_res_blocks):
                layers = [
                    ResBlock(channels, d_time_embed, out_channels=channels_list[i])
                ]
                channels = channels_list[i]

                if i in attention_levels:
                    layers.append(
                        SpatialTransformer(channels, n_heads, tf_layers, d_cond)
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_channels.append(channels)

            if i != levels - 1:
                self.input_blocks.append(TimestepEmbedSequential(DownSample(channels)))
                input_block_channels.append(channels)

        self.middle_block = TimestepEmbedSequential(
            ResBlock(channels, d_t_embed),
            SpatialTransformer(channels, n_heads, tf_layers, d_cond),
            ResBlock(channels, d_time_embed),
        )

        self.output_blocks = nn.ModuleList([])

        for i in reversed(range(levels)):
            for j in range(n_res_blocks + 1):
                layers = [
                    ResBlock(
                        channels + input_block_channels.pop(),
                        d_t_embed,
                        out_channels=channels_list[i],
                    )
                ]
                channels = channels_list[i]

                if i in attention_levels:
                    layers.append(
                        SpatialTransformer(channels, n_heads, tf_layers, d_cond)
                    )

                if i != 0 and j == n_res_blocks:
                    layers.append(UpSample(channels))

                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )

    def time_step_embedding(self, time_steps, max_period=1000):
        half = self.channels // 2

        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=time_steps.device)
        args = time_steps[:, None].float() * frequencies[None]

        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, x, time_steps, cond):
        x_input_block = []

        t_emb = self.time_step_embedding(time_steps)
        t_emb = self.time_embed(t_emb)

        for module in self.input_blocks:
            x = module(x, t_emb, cond)
            x_input_block.append(x)

        x = self.middle_block(x, t_emb, cond)

        for module in self.output_blocks:
            x = torch.cat([x, x_input_block.pop()], dim=1)
            x = module(x, t_emb, cond)

        return self.out(x)
