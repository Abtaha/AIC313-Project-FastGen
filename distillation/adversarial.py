"""Small shared-backbone GAN branch for DMD2-inspired one-pass refinement.

Reference: https://arxiv.org/abs/2405.14867, sections 4.3 and F.1.
Uses our existing attributed SongUNet, without external pretrained networks.
"""

from contextlib import contextmanager
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from dataset import PokemonDataset
from src.edm.networks import UNetBlock


def bottleneck_features(fake, images, sigma, labels):
    """SongUNet's standard encoder and middle blocks; omit unused decoder."""
    net, config = fake.backbone, fake.config
    sigma = torch.as_tensor(sigma, device=images.device).float().reshape(-1).expand(len(images))
    x = (sigma[:, None, None, None].square() + config.sigma_data**2).rsqrt() * images.float()
    emb = net.map_noise(sigma.log() / 4)
    emb = emb.reshape(len(images), 2, -1).flip(1).reshape_as(emb)
    condition = F.one_hot(labels, config.num_classes).float()
    emb = emb + net.map_label(condition * math.sqrt(config.num_classes))
    emb = F.silu(net.map_layer1(F.silu(net.map_layer0(emb))))
    for name, block in net.enc.items():
        if 'aux' in name:
            raise ValueError("GAN branch supports only the standard SongUNet encoder")
        x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
    resolution = x.shape[-1]
    for index in (0, 1):
        x = net.dec[f'{resolution}x{resolution}_in{index}'](x, emb)
    return x


class GANHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        channels = config.model_channels * config.channel_mult[-1]
        self.layers = nn.Sequential(nn.Conv2d(channels, 128, 4, stride=2, padding=1),
            nn.GroupNorm(16, 128), nn.SiLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 1))

    def forward(self, fake, images, labels, *, sigma=None, noise=None):
        if sigma is None:
            sigma = (torch.randn(len(images), device=images.device)*1.2-1.2).exp().clamp(.05, 20)
        if noise is None:
            noise = torch.randn_like(images)
        # Match the bounded image domain used at inference for both classes.
        noisy = images.clamp(-1, 1) + sigma[:, None, None, None]*noise
        return self.layers(bottleneck_features(fake, noisy, sigma, labels)).flatten()


@contextmanager
def frozen_parameters(*modules):
    """Freeze weights while preserving the differentiable path to input images."""
    parameters = [parameter for module in modules for parameter in module.parameters()]
    flags = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, flag in zip(parameters, flags):
            parameter.requires_grad_(flag)


class BalancedTrainingImages:
    """Uniform classes, uniform image within class; opens only train_split.txt."""
    def __init__(self, data_root, split_dir, mapping):
        self.dataset = PokemonDataset(data_root, Path(split_dir)/'train_split.txt',
                                     return_category=True, category_to_id=mapping)
        categories = torch.tensor(self.dataset.category_ids)
        self.indices = [(categories == label).nonzero().flatten() for label in range(len(mapping))]
        if any(len(indices) == 0 for indices in self.indices):
            raise ValueError("Every class needs training images for GAN supervision")

    def sample(self, count, device):
        labels = torch.randint(len(self.indices), (count,))
        selected = [int(self.indices[int(label)][torch.randint(len(self.indices[int(label)]), ())])
                    for label in labels]
        images = torch.stack([self.dataset[index][0] for index in selected]).to(device)
        return images, labels.to(device)


def random_flip(images):
    flip = torch.rand(len(images), 1, 1, 1, device=images.device) < .5
    return torch.where(flip, images.flip(-1), images)
