"""
Contains code for U-Net architecture based on Stable Diffusion for generative purposes

This model will be trained to estimate the noise added to some image at some time step, 

model = UNet(in_channels=(3,5,6,767,), out_channels=(34,434,6,45))

Based on: https://nn.labml.ai/diffusion/stable_diffusion/model/unet.html

"""



import torch.nn as nn

class UNet(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels, int,
        channels: int,
        n_res_blocks: int,
        attention_levels: List[int],
        channel_multipliers: List[int],
        n_heads: int,
        tf_layers: int = 1,
        d_cond: int = 768
    ):
    """
    in_
    """
    pass
    



