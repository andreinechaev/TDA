import torch
import torch.nn as nn
import numpy as np


class TDA(nn.Module):
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
        super(TDA, self).__init__()
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


class TDAV2(nn.Module):
    """
    Temporal Decaying Accumulator (TDA) function an optimization attempt
    """

    def __init__(self, time_window: float, max_value: float, scaling_factor: float, op='mean'):
        """
        :param time_window: Time window for the exponential decay
        :param max_value: Maximum value of the activation
        :param scaling_factor: Scaling factor for the exponential decay
        :param op: Operation to perform on the activation. Either "sum" or "mean"
        """
        super(TDAV2, self).__init__()
        self.time_window = time_window
        self.max_value = max_value
        self.scaling_factor = scaling_factor

        if op == 'sum':
            self.op = torch.sum
        elif op == 'mean':
            self.op = torch.mean
        else:
            raise ValueError('op must be either "sum" or "mean"')

        t = torch.arange(time_window, dtype=torch.float32)
        self.register_buffer('weights', self.scaling_factor * torch.exp(-t / self.time_window))

    def forward(self, x: torch.Tensor):
        """
        :param x: Input tensor of shape (batch_size, num_channels, num_timesteps)
        """
        # Truncate or pad the weights if the input tensor's time dimension is different than the precomputed weights
        if x.shape[2] != self.weights.shape[0]:
            if x.shape[2] > self.weights.shape[0]:
                padding = x.shape[2] - self.weights.shape[0]
                adjusted_weights = torch.cat([self.weights, torch.zeros(padding, device=self.weights.device)], dim=0)
            else:
                adjusted_weights = self.weights[:x.shape[2]]
        else:
            adjusted_weights = self.weights

        # Reshape and expand adjusted weights to match the input tensor dimensions
        weights = adjusted_weights.view(1, 1, -1).expand_as(x)
        weighted_x = x * weights
        clamped_x = torch.clamp(weighted_x, min=0, max=self.max_value)
        return self.op(clamped_x, dim=2)
