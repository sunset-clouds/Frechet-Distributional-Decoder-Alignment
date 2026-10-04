
import torch
import torch.nn as nn
import torch.nn.functional as F


### Building blocks
def nonlinearity(x):
    return x * torch.sigmoid(x)  # swish

def Normalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0.0):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        h = self.conv1(nonlinearity(self.norm1(x)))
        h = self.conv2(self.dropout(nonlinearity(self.norm2(h))))
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.k = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.v = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1, 1, 0)

    def forward(self, x):
        h = self.norm(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        b, c, hh, ww = q.shape
        q = q.reshape(b, c, hh * ww).permute(0, 2, 1)   # b, hw, c
        k = k.reshape(b, c, hh * ww)                     # b, c, hw
        w = torch.bmm(q, k) * (c ** -0.5)
        w = F.softmax(w, dim=2).permute(0, 2, 1)
        v = v.reshape(b, c, hh * ww)
        h = torch.bmm(v, w).reshape(b, c, hh, ww)
        return x + self.proj_out(h)


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 2, 0)

    def forward(self, x):
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)

### Encoder / Decoder
class Encoder(nn.Module):
    def __init__(self, in_channels=3, ch=128, ch_mult=(1, 1, 2, 2, 4),
                 num_res_blocks=2, dropout=0.0, z_channels=256):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.conv_in = nn.Conv2d(in_channels, ch, 3, 1, 1)

        in_ch_mult = (1,) + tuple(ch_mult)
        self.conv_blocks = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.Module()
            res, attn = nn.ModuleList(), nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                res.append(ResnetBlock(block_in, block_out, dropout=dropout))
                block_in = block_out
                if i_level == self.num_resolutions - 1:
                    attn.append(AttnBlock(block_in))
            block.res, block.attn = res, attn
            if i_level != self.num_resolutions - 1:
                block.downsample = Downsample(block_in)
            self.conv_blocks.append(block)

        self.mid = nn.ModuleList([
            ResnetBlock(block_in, block_in, dropout=dropout),
            AttnBlock(block_in),
            ResnetBlock(block_in, block_in, dropout=dropout),
        ])
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, z_channels, 3, 1, 1)

    def forward(self, x):
        h = self.conv_in(x)
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.downsample(h)
        for mid in self.mid:
            h = mid(h)
        return self.conv_out(nonlinearity(self.norm_out(h)))


class Decoder(nn.Module):
    def __init__(self, z_channels=256, ch=128, ch_mult=(1, 1, 2, 2, 4),
                 num_res_blocks=2, dropout=0.0, out_channels=3):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = ch * ch_mult[self.num_resolutions - 1]
        self.conv_in = nn.Conv2d(z_channels, block_in, 3, 1, 1)

        self.mid = nn.ModuleList([
            ResnetBlock(block_in, block_in, dropout=dropout),
            AttnBlock(block_in),
            ResnetBlock(block_in, block_in, dropout=dropout),
        ])

        self.conv_blocks = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.Module()
            res, attn = nn.ModuleList(), nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks + 1):
                res.append(ResnetBlock(block_in, block_out, dropout=dropout))
                block_in = block_out
                if i_level == self.num_resolutions - 1:
                    attn.append(AttnBlock(block_in))
            block.res, block.attn = res, attn
            if i_level != 0:
                block.upsample = Upsample(block_in)
            self.conv_blocks.append(block)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_channels, 3, 1, 1)

    def forward(self, z):
        h = self.conv_in(z)
        for mid in self.mid:
            h = mid(h)
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks + 1):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.upsample(h)
        return self.conv_out(nonlinearity(self.norm_out(h)))

### Vector quantizer 
class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, l2_norm=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.l2_norm = l2_norm
        self.embedding = nn.Embedding(n_e, e_dim)

    def _codebook(self):
        w = self.embedding.weight
        return F.normalize(w, p=2, dim=-1) if self.l2_norm else w

    def forward(self, z):
        # z: (B, C, H, W) -> z_q: (B, C, H, W), indices: (B*H*W,)
        z = torch.einsum("b c h w -> b h w c", z).contiguous()
        z_flat = z.view(-1, self.e_dim)
        if self.l2_norm:
            z = F.normalize(z, p=2, dim=-1)
            z_flat = F.normalize(z_flat, p=2, dim=-1)
        embedding = self._codebook()
        d = (z_flat ** 2).sum(1, keepdim=True) + (embedding ** 2).sum(1) \
            - 2 * torch.einsum("bd,nd->bn", z_flat, embedding)
        indices = torch.argmin(d, dim=1)
        z_q = embedding[indices].view(z.shape)
        z_q = torch.einsum("b h w c -> b c h w", z_q)
        return z_q, indices

    def get_codebook_entry(self, indices, shape=None, channel_first=True):
        # indices: (B*H*W,) ; shape: (B, C, H, W) if channel_first
        z_q = self._codebook()[indices]
        if shape is not None:
            if channel_first:
                z_q = z_q.reshape(shape[0], shape[2], shape[3], shape[1])
                z_q = z_q.permute(0, 3, 1, 2).contiguous()
            else:
                z_q = z_q.view(shape)
        return z_q