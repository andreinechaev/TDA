import torch
import torch.nn as nn
import numpy as np


class TDAActivation(nn.Module):
    """
    Temporal Decaying Accumulator (TDA) function

    The function designed for handling time-dependent data in neural networks.
    The primary objective of TDA is to emphasize the importance of recent information
    while diminishing the influence of older data.
    This is achieved through the use of an exponential decay function that computes the weights 
    for inputs based on their age, and a clipping mechanism that limits the maximum value of the output
    """

    def __init__(self, time_window: float, max_value: float, scaling_factor: float, op='mean'):
        """
        :param time_window: Time window for the exponential decay
        :param max_value: Maximum value of the activation
        :param scaling_factor: Scaling factor for the exponential decay
        :param op: Operation to perform on the activation. Either "sum" or "mean"

        sum: If you use the sum of the weighted inputs, 
            the TDA activation function computes the total accumulated effect of the inputs within the time window.
            The output value depends on the number of input values and their magnitudes.
            This approach might be more sensitive to the presence of multiple inputs,
            and the output value can grow larger if many inputs are present within the time window.
        mean: If you use the mean of the weighted inputs, 
            the TDA activation function computes the average effect of the inputs within the time window.
            This approach normalizes the output by the number of input values,
            making the output less sensitive to the presence of multiple inputs within the time window.
            Using the mean can help reduce the impact of outliers and noise in the input data.
        """
        super(TDAActivation, self).__init__()
        self.time_window = time_window
        self.max_value = max_value
        self.scaling_factor = scaling_factor

        if op == 'sum':
            self.op = torch.sum
        elif op == 'mean':
            self.op = torch.mean
        else:
            raise ValueError('op must be either "sum" or "mean"')

    def weight(self, t):
        return self.scaling_factor * np.exp(-t / self.time_window)

    def forward(self, x: torch.Tensor):
        """
        :param x: Input tensor of shape (batch_size, num_channels, num_timesteps)
        """
        t = torch.arange(x.shape[2], dtype=torch.float32, device=x.device)
        weights = self.scaling_factor * torch.exp(-t / self.time_window)
        # Reshape and expand weights to match the input tensor dimensions
        weights = weights.view(1, 1, -1).expand_as(x)
        weighted_x = x * weights
        clamped_x = torch.clamp(weighted_x, min=0, max=self.max_value)
        return self.op(clamped_x, dim=2)
