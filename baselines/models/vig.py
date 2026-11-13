"""Pyramid Vision GNN (ViG) - Han et al., "Vision GNN: An Image is Worth Graph of Nodes", NeurIPS 2022.

Self-contained re-implementation whose parameter names mirror the official code
(huawei-noah/Efficient-AI-Backbones, vig_pytorch/pyramid_vig.py) so the official
ImageNet checkpoints load directly.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.layers import DropPath
except ImportError:  # older timm
    from timm.models.layers import DropPath

PVIG_CONFIGS = {
    "pvig_ti": dict(channels=[48, 96, 240, 384], blocks=[2, 2, 6, 2]),
    "pvig_s": dict(channels=[80, 160, 400, 640], blocks=[2, 2, 6, 2]),
    "pvig_m": dict(channels=[96, 192, 384, 768], blocks=[2, 2, 16, 2]),
    "pvig_b": dict(channels=[128, 256, 512, 1024], blocks=[2, 2, 18, 2]),
}


# ----------------------------------------------------------------------------- positional terms
def _1d_sincos(embed_dim, pos):
    omega = 1.0 / 10000 ** (np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0))
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def _2d_sincos(embed_dim, grid_size):
    gh, gw = np.arange(grid_size, dtype=np.float32), np.arange(grid_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(gw, gh), axis=0).reshape(2, 1, grid_size, grid_size)
    return np.concatenate([_1d_sincos(embed_dim // 2, grid[0]), _1d_sincos(embed_dim // 2, grid[1])], axis=1)


def relative_pos_embed(embed_dim, grid_size):
    pe = _2d_sincos(embed_dim, grid_size)
    return 2 * np.matmul(pe, pe.T) / pe.shape[1]


# ----------------------------------------------------------------------------- graph ops
def _pairwise_dist(x, y):
    # x: B,N,C  y: B,M,C
    inner = -2 * torch.matmul(x, y.transpose(2, 1))
    return (x * x).sum(-1, keepdim=True) + inner + (y * y).sum(-1, keepdim=True).transpose(2, 1)


def _batched_index_select(x, idx):
    """x: B,C,N,1   idx: B,Nq,k  ->  B,C,Nq,k"""
    b, c, n = x.shape[:3]
    _, nq, k = idx.shape
    idx = (idx + torch.arange(b, device=idx.device).view(-1, 1, 1) * n).reshape(-1)
    feat = x.transpose(2, 1).reshape(b * n, -1)[idx]
    return feat.view(b, nq, k, c).permute(0, 3, 1, 2).contiguous()


class DenseDilatedKnnGraph(nn.Module):
    def __init__(self, k=9, dilation=1):
        super().__init__()
        self.k, self.dilation = k, dilation

    @torch.no_grad()
    def forward(self, x, y=None, relative_pos=None):
        x = F.normalize(x.float(), p=2.0, dim=1)
        y = x if y is None else F.normalize(y.float(), p=2.0, dim=1)
        xs, ys = x.transpose(2, 1).squeeze(-1), y.transpose(2, 1).squeeze(-1)
        b, n, _ = xs.shape
        dist = _pairwise_dist(xs, ys)
        if relative_pos is not None:
            dist = dist + relative_pos.float()
        k = min(self.k * self.dilation, ys.shape[1])
        _, nn_idx = torch.topk(-dist, k=k)
        center = torch.arange(n, device=x.device).repeat(b, k, 1).transpose(2, 1)
        edge = torch.stack((nn_idx, center), dim=0)
        return edge[:, :, :, :: self.dilation]


class MRConv2d(nn.Module):
    """Max-Relative graph convolution."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.nn = nn.Sequential(nn.Conv2d(in_channels * 2, out_channels, 1, bias=True, groups=4),
                                nn.BatchNorm2d(out_channels), nn.GELU())

    def forward(self, x, edge_index, y=None):
        x_i = _batched_index_select(x, edge_index[1])
        x_j = _batched_index_select(x if y is None else y, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        b, c, n, _ = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _)
        return self.nn(x)


class DyGraphConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, r=1):
        super().__init__()
        self.gconv = MRConv2d(in_channels, out_channels)
        self.r = r
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation)

    def forward(self, x, relative_pos=None):
        b, c, h, w = x.shape
        y = None
        if self.r > 1:
            y = F.avg_pool2d(x, self.r, self.r).reshape(b, c, -1, 1).contiguous()
        x = x.reshape(b, c, -1, 1).contiguous()
        edge_index = self.dilated_knn_graph(x, y, relative_pos)
        x = self.gconv(x, edge_index, y)
        return x.reshape(b, -1, h, w).contiguous()


class Grapher(nn.Module):
    def __init__(self, in_channels, kernel_size=9, dilation=1, r=1, n=196, drop_path=0.0, relative_pos=True):
        super().__init__()
        self.n, self.r = n, r
        self.fc1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels))
        self.graph_conv = DyGraphConv2d(in_channels, in_channels * 2, kernel_size, dilation, r)
        self.fc2 = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, 1), nn.BatchNorm2d(in_channels))
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.relative_pos = None
        if relative_pos:
            rp = torch.from_numpy(np.float32(relative_pos_embed(in_channels, int(n ** 0.5)))).unsqueeze(0).unsqueeze(1)
            rp = F.interpolate(rp, size=(n, n // (r * r)), mode="bicubic", align_corners=False)
            self.relative_pos = nn.Parameter(-rp.squeeze(1), requires_grad=False)

    def _get_relative_pos(self, h, w):
        if self.relative_pos is None or h * w == self.n:
            return self.relative_pos
        n = h * w
        return F.interpolate(self.relative_pos.unsqueeze(0), size=(n, n // (self.r * self.r)),
                             mode="bicubic", align_corners=False).squeeze(0)

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        _, _, h, w = x.shape
        x = self.graph_conv(x, self._get_relative_pos(h, w))
        x = self.fc2(x)
        return self.drop_path(x) + shortcut


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features, drop_path=0.0):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Conv2d(in_features, hidden_features, 1), nn.BatchNorm2d(hidden_features))
        self.act = nn.GELU()
        self.fc2 = nn.Sequential(nn.Conv2d(hidden_features, in_features, 1), nn.BatchNorm2d(in_features))
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        return self.drop_path(self.fc2(self.act(self.fc1(x)))) + x


class Stem(nn.Module):
    def __init__(self, in_dim=3, out_dim=768):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim // 2, 3, stride=2, padding=1), nn.BatchNorm2d(out_dim // 2), nn.GELU(),
            nn.Conv2d(out_dim // 2, out_dim, 3, stride=2, padding=1), nn.BatchNorm2d(out_dim), nn.GELU(),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1), nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        return self.convs(x)


class Downsample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1), nn.BatchNorm2d(out_dim))

    def forward(self, x):
        return self.conv(x)


class PyramidViG(nn.Module):
    def __init__(self, channels, blocks, num_classes=1000, k=9, drop_path_rate=0.0, dropout=0.0, img_size=224):
        super().__init__()
        self.num_classes = num_classes
        reduce_ratios = [4, 2, 1, 1]
        n_blocks = sum(blocks)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_blocks)]
        max_dilation = 49 // k
        self.stem = Stem(out_dim=channels[0])
        self.pos_embed = nn.Parameter(torch.zeros(1, channels[0], img_size // 4, img_size // 4))
        hw = (img_size // 4) ** 2
        layers, idx = [], 0
        for i, nb in enumerate(blocks):
            if i > 0:
                layers.append(Downsample(channels[i - 1], channels[i]))
                hw //= 4
            for _ in range(nb):
                layers.append(nn.Sequential(
                    Grapher(channels[i], k, min(idx // 4 + 1, max_dilation), reduce_ratios[i], n=hw,
                            drop_path=dpr[idx], relative_pos=True),
                    FFN(channels[i], channels[i] * 4, drop_path=dpr[idx]),
                ))
                idx += 1
        self.backbone = nn.ModuleList(layers)
        self.prediction = nn.Sequential(nn.Conv2d(channels[-1], 1024, 1, bias=True), nn.BatchNorm2d(1024),
                                        nn.GELU(), nn.Dropout(dropout), nn.Conv2d(1024, num_classes, 1, bias=True))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        pos = self.pos_embed
        if pos.shape[-2:] != x.shape[-2:]:
            pos = F.interpolate(pos, size=x.shape[-2:], mode="bicubic", align_corners=False)
        x = x + pos
        for blk in self.backbone:
            x = blk(x)
        x = F.adaptive_avg_pool2d(x, 1)
        return self.prediction(x).flatten(1)


def _load_checkpoint(url: str):
    sd = torch.hub.load_state_dict_from_url(url, map_location="cpu", progress=True, check_hash=False,
                                            weights_only=False)
    for key in ("state_dict", "model", "state_dict_ema"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
            break
    return {k.replace("module.", "", 1): v for k, v in sd.items()}


def build_pvig(arch: str, num_classes: int, k: int = 9, drop_path_rate: float = 0.0, dropout: float = 0.0,
               pretrained_url: str | None = None, img_size: int = 224) -> PyramidViG:
    if arch not in PVIG_CONFIGS:
        raise KeyError(f"Unknown ViG arch '{arch}'. Choose from {list(PVIG_CONFIGS)}")
    model = PyramidViG(**PVIG_CONFIGS[arch], num_classes=num_classes, k=k, drop_path_rate=drop_path_rate,
                       dropout=dropout, img_size=img_size)
    if pretrained_url:
        sd = _load_checkpoint(pretrained_url)
        own = model.state_dict()
        # drop the ImageNet classifier and anything with a mismatched shape (e.g. other image size)
        sd = {k_: v for k_, v in sd.items() if k_ in own and own[k_].shape == v.shape}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[ViG] loaded ImageNet weights ({len(sd)} tensors); re-initialised: "
              f"{[m for m in missing if 'num_batches_tracked' not in m]}")
    return model
