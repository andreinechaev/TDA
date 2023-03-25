import torch
import torch.nn as nn
import numpy as np

class TDAActivation(nn.Module):
    def __init__(self, time_window, max_value, scaling_factor):
        super(TDAActivation, self).__init__()
        self.time_window = time_window
        self.max_value = max_value
        self.scaling_factor = scaling_factor

    def weight(self, t):
        return self.scaling_factor * np.exp(-t / self.time_window)

    def forward(self, x):
        t = torch.arange(x.shape[2], dtype=torch.float32, device=x.device)
        weights = self.scaling_factor * torch.exp(-t / self.time_window)
        weights = weights.view(1, 1, -1).expand_as(x)  # Reshape and expand weights to match the input tensor dimensions
        weighted_x = x * weights
        clamped_x = torch.clamp(weighted_x, min=0, max=self.max_value)
        # To compute the weighted sum instead of the mean, you can replace the last line with:
        # return clamped_x.sum(dim=2)
        return clamped_x.mean(dim=2)  # Compute the mean along the time_window dimension