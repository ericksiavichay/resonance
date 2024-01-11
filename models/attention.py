import torch.nn as nn
import torch.nn.functional as F
import torch

# implementation from https://nn.labml.ai/diffusion/stable_diffusion/model/unet_attention.html


class CrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_cond: int,
        n_heads: int,
        d_head: int,
        is_inplace: bool = True,
        use_flash_attention: bool = True,
    ):
        super().__init__()

        self.is_inplace = is_inplace
        self.n_heads = n_heads
        self.d_head = d_head
        self.scale = d_head**-0.5

        d_attn = d_head * n_heads
        self.to_q = nn.Linear(d_model, d_attn, bias=False)
        self.to_k = nn.Linear(d_cond, d_attn, bias=False)
        self.to_v = nn.Linear(d_cond, d_attn, bias=False)

        self.to_out = nn.Sequential(nn.Linear(d_attn, d_model))

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None):
        """
        Function assumes a condition. If no conditional vector, set the conditional equal to x
        """

        if cond is None:
            cond = x

        q = self.to_q(x)
        k = self.to_k(cond)
        v = self.to_v(cond)

        return self.normal_attention(q, k, v)

    def normal_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # Reshaping tensors for multi-headed attention
        q = q.view(*q.shape[:2], self.n_heads, -1)
        k = k.view(*k.shape[:2], self.n_heads, -1)
        v = v.view(*v.shape[:2], self.n_heads, -1)

        # Transpose k for matrix multiplication: [batch_size, num_heads, head_dim, seq_len]
        k_transposed = k.transpose(-2, -1)

        # Batch matrix multiplication
        # q shape: [batch_size, seq_len, num_heads, head_dim]
        # k_transposed shape: [batch_size, num_heads, head_dim, seq_len]
        # Resultant attn shape: [batch_size, num_heads, seq_len, seq_len]
        attn = q @ k_transposed

        # Scale the attention scores
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)

        # Transpose v for matrix multiplication: [batch_size, num_heads, seq_len, head_dim]\
        v_transposed = v.transpose(1, 2)

        # attn shape: [batch_size, num_heads, seq_len, seq_len]
        # v_transposed shape: [batch_size, num_heads, seq_len, head_dim]
        # Resultant shape: [batch_size, num_heads, seq_len, head_dim]
        weighted_v = attn @ v_transposed
        out = weighted_v.transpose(1, 2).reshape(*weighted_v.shape[:2], -1)

        return self.to_out(out)


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
        d_head: int,
        d_cond: int,
    ):
        super().__init__()

        self.attn1 = CrossAttention(d_model, d_model, n_heads, d_head)
        self.norm1 = nn.LayerNorm(d_model)

        self.attn2 = CrossAttention(d_model, d_cond, n_heads, d_head)
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
        device: str,
    ):
        """
        Following from: https://nn.labml.ai/diffusion/stable_diffusion/model/unet_attention.html
        """
        super().__init__()
        self.norm = nn.GroupNorm(
            num_groups=32, num_channels=channels, eps=1e-6, affine=True, device=device
        )  # Group norm paper sets G = 32, could be a hyperparameter
        self.proj_in = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(channels, n_heads, channels // n_heads, d_cond=d_cond)
                for _ in range(n_layers)
            ]
        )
        self.proj_out = nn.Conv2d(
            channels, channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        b, c, h, w = x.shape
        x_in = x
        x = self.proj_in(self.norm(x))
        x = x.permute(0, 2, 3, 1).view(b, h * w, c)

        for block in self.transformer_blocks:
            x = block(x, cond)

        x = x.view(b, h, w, c).permute(0, 3, 1, 2)
        x = self.proj_out(x)
        return x + x_in
