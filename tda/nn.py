import torch
import torch.nn as nn

class TDA(nn.Module):
    def __init__(self, time_window, scaling_factor):
        super(TDA, self).__init__()
        self.time_window = time_window
        self.scaling_factor = scaling_factor

    def weight(self, t):
        return self.scaling_factor * torch.exp(-t / self.time_window)

    def forward(self, x):
        t = torch.arange(x.shape[2], dtype=torch.float32, device=x.device)
        weights = self.scaling_factor * torch.exp(-t / self.time_window)
        weights = weights.view(1, 1, -1).expand_as(x)  # Reshape and expand weights to match the input tensor dimensions
        weighted_x = x * weights
        return weighted_x


class TDAClip(nn.Module):
    def __init__(self, max_value, op=torch.mean):
        super(TDAClip, self).__init__()
        self.max_value = max_value
        self.op = op

    def forward(self, x):
        clamped_x = torch.clamp(x, min=0, max=self.max_value)
        return self.op(clamped_x, dim=2)  # Compute the op along the time_window dimension
