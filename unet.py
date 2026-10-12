import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t is continuous in [0, 1]. Scale it like DDPM timesteps for embeddings.
        t = t * 1000.0
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).view(B, 3, C, H * W).unbind(1)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return x + self.proj(out)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_ch):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(t_ch, out_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return self.shortcut(x) + h


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        base_ch = 128
        t_ch = base_ch * 4
        self.time_mlp = nn.Sequential(
            TimeEmbedding(base_ch),
            nn.Linear(base_ch, t_ch),
            nn.SiLU(),
            nn.Linear(t_ch, t_ch)
        )

        self.inc = nn.Conv2d(3, base_ch, 3, padding=1)

        # Down stages
        self.down1 = nn.ModuleList([ResBlock(base_ch, base_ch, t_ch), ResBlock(base_ch, base_ch, t_ch)])
        self.down2 = nn.ModuleList([nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1), ResBlock(base_ch * 2, base_ch * 2, t_ch),
                                    ResBlock(base_ch * 2, base_ch * 2, t_ch)])
        self.down3 = nn.ModuleList(
            [nn.Conv2d(base_ch * 2, base_ch * 2, 4, 2, 1), ResBlock(base_ch * 2, base_ch * 2, t_ch),
             AttentionBlock(base_ch * 2)])

        # Mid
        self.mid1 = ResBlock(base_ch * 2, base_ch * 2, t_ch)
        self.mid_attn = AttentionBlock(base_ch * 2)
        self.mid2 = ResBlock(base_ch * 2, base_ch * 2, t_ch)

        # Up stages
        self.up1 = nn.ModuleList([ResBlock(base_ch * 4, base_ch * 2, t_ch), AttentionBlock(base_ch * 2),
                                  nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 4, 2, 1)])
        self.up2 = nn.ModuleList([ResBlock(base_ch * 4, base_ch, t_ch), ResBlock(base_ch, base_ch, t_ch),
                                  nn.ConvTranspose2d(base_ch, base_ch, 4, 2, 1)])
        self.up3 = nn.ModuleList([ResBlock(base_ch * 2, base_ch, t_ch), ResBlock(base_ch, base_ch, t_ch)])

        self.outc = nn.Sequential(
            nn.GroupNorm(32, base_ch), nn.SiLU(), nn.Conv2d(base_ch, 3, 3, padding=1)
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x1 = self.inc(x)

        # Down
        h1 = self.down1[0](x1, t_emb);
        h1 = self.down1[1](h1, t_emb)
        h2 = self.down2[0](h1);
        h2 = self.down2[1](h2, t_emb);
        h2 = self.down2[2](h2, t_emb)
        h3 = self.down3[0](h2);
        h3 = self.down3[1](h3, t_emb);
        h3 = self.down3[2](h3)

        # Mid
        h = self.mid1(h3, t_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)

        # Up (concatenating skips)
        h = self.up1[0](torch.cat([h, h3], dim=1), t_emb);
        h = self.up1[1](h);
        h = self.up1[2](h)
        h = self.up2[0](torch.cat([h, h2], dim=1), t_emb);
        h = self.up2[1](h, t_emb);
        h = self.up2[2](h)
        h = self.up3[0](torch.cat([h, h1], dim=1), t_emb);
        h = self.up3[1](h, t_emb)

        return self.outc(h)