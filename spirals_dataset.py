import torch
from torch.utils.data import Dataset
import numpy as np


class SpiralsDataset(Dataset):
    def __init__(self, n_samples, noise_std=0., rotations=1.):
        theta_max = rotations * 2 * np.pi
        thetas = torch.linspace(0, 1, n_samples) ** 0.5 * theta_max
        rs = thetas / theta_max
        signs = torch.randint(2, (n_samples,)) * 2 - 1
        xs = rs * signs * torch.cos(thetas) + torch.randn(n_samples) * noise_std
        ys = rs * signs * torch.sin(thetas) + torch.randn(n_samples) * noise_std
        self.points = torch.stack([xs, ys], axis=1)
        self.labels = (signs > 0)

    def __getitem__(self, index):
        return self.points[index], self.labels[index]

    def __len__(self):
        return len(self.points)
