# Temporal Decaying Accumulator (TDA)

## General

Temporal Decaying Accumulator (TDA) is a novel activation function designed for handling time-dependent data in neural networks. The primary objective of TDA is to emphasize the importance of recent information while diminishing the influence of older data. This is achieved through the use of an exponential decay function that computes the weights for inputs based on their age, and a clipping mechanism that limits the maximum value of the output.
Key Components of TDA:

1. Exponential Decay Function: The TDA activation function incorporates an exponential decay function to compute the weights for each input based on its age. The decay function is defined as: $w(t) = a * exp(-t / τ)$ where t is the time step or age of the input, τ is the time constant that controls the rate of decay, and a is a scaling factor that adjusts the overall magnitude of the weights.
2. Weighted Sum of Inputs: TDA computes the weighted sum of inputs within a specified time window. The weighted sum is calculated as:TDA(x_t) = sum(w(t) * x_(t-i)) for i in range(0, N)where x_t is the input at the current time step t, N is the size of the time window, w(t) is the weight function defined earlier, and x_(t-i) is the input at the previous time step t-i.
3. Clipping Mechanism: The output of the TDA function is clipped to a maximum value to prevent it from growing indefinitely. The clipped output is computed as:TDA_clipped(x_t) = min(TDA(x_t), V_max)where V_max is the maximum value allowed for the TDA function output.

## Dependencies

TDA is implemented in Python 3 and Pytorch that requires the following packages:

1. numpy
2. pytorch (2.0.0 and 1.13.1 were tested)

To run examples you will also need:

1. matplotlib
2. torchtext
3. torchdata
4. jupyter

## Usage

### TDA Activation Function

```python
import torch
import torch.nn as nn
import numpy as np

from tda import TDAActivation

class TDANetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, time_window, max_value, scaling_factor):
        super(TDANetwork, self).__init__()
        self.layer1 = nn.Linear(input_size * time_window, hidden_size)
        self.ac = TDAActivation(time_window, max_value, scaling_factor)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.shape[0], -1)  # Flatten the input tensor along the time_window dimension
        x = self.layer1(x)
        x = x.unsqueeze(-1)  # Add an extra dimension for the time_window
        x = self.ac(x)
        x = self.layer2(x)
        return x
```

More examples can be found in the example notebooks.

### Example Notebooks

- [Houshold Power Consumption](tda_time_series.ipynb)
- [IMDB Sentiment Analysis](nlp.ipynb)
- [US Unemployment Rate](unemployment_us.ipynb)

### Hyperparameters

#### Time Window

The `time_window` parameter plays a critical role in determining how the TDA activation function weighs the influence of input values based on their age. Choosing the right value for time_window can significantly impact the performance of the neural network when dealing with time-series data.
It directly affects the exponential decay function used to compute the age-based weights for each input value.
The time_window parameter affects the results in the following ways:

- Emphasis on recent information: A smaller time_window results in a faster decay of the weights associated with older data points. This causes the model to focus more on recent information and less on historical data, which might be useful when recent data is more relevant to the prediction task.
- Retention of historical information: A larger time_window results in a slower decay of the weights associated with older data points. This causes the model to retain more historical information, which might be useful when the past data points still have a significant influence on the prediction task.
- Sensitivity to noise and outliers: A smaller time_window might make the model more sensitive to noise and outliers in the input data, as it focuses more on recent data points. Conversely, a larger time_window might make the model more robust to noise and outliers, as it averages the influence of data points over a more extended period.

#### Scaling Factor

A higher scaling_factor will result in larger weights for each time step, which in turn will lead to larger weighted input values before applying the chosen operation (either 'sum' or 'mean') and the clipping mechanism. This could potentially increase the influence of each input value on the output of the TDA activation function.

Conversely, a lower scaling_factor will result in smaller weights for each time step, which will decrease the influence of each input value on the output of the TDA activation function.

The choice of scaling_factor can impact the performance of the neural network, as it determines the magnitude of the weights applied to the input data. It is essential to experiment with different values of scaling_factor to find the optimal balance between emphasizing recent information and diminishing the influence of older data, depending on the specific problem and dataset being used.

#### Max Value (V_max)

It serves as an upper limit for the weighted input values before applying the chosen operation ('sum' or 'mean'). It is used in the clipping step, which ensures that the weighted input values are limited to a specific range, i.e., between 0 and max_value.

The max_value affects the results in the following ways:

- Regularization: By clipping the weighted input values, it can help prevent large output values from dominating the activation function output. This acts as a form of regularization, making the model more robust to outliers and noise in the input data.
- Non-linearity: Clipping introduces a non-linear behavior into the activation function, which can help the neural network learn complex, non-linear relationships between the input data and the target variable.
- Gradient saturation: Setting a very low max_value might cause gradient saturation during the backpropagation, as the gradient of the clipped values will be zero. Gradient saturation can slow down the training process and result in a poorly trained model. On the other hand, if the max_value is too high, the clipping mechanism might not have a significant effect on the output values, and the benefits of regularization and non-linearity may be reduced.

## Advantages and Challenges

**Advantages**:

- Improved handling of time-dependent data: TDA could be particularly useful in applications where recent information is more critical, such as in financial markets, weather prediction, or natural language processing with a focus on recent context.
- Adaptability: The TDA function has tunable parameters (τ, a, and V_max), which makes it adaptable to various applications and datasets. It can be optimized based on the specific requirements of the task.
- Reduced noise influence: By emphasizing recent information and diminishing the influence of older data, the TDA function might help reduce the impact of noise or outliers in time series data.

**Challenges**:

- Applicability: TDA may not be suitable for all types of time-dependent data. In some cases, older information could be more important than recent information, or the importance of data may not decay exponentially.
- Complexity: The TDA function is more computationally complex than some other activation functions, such as ReLU or Sigmoid, which could lead to increased training time and resource requirements.
- Local optima: The TDA function, like other activation functions, might suffer from issues related to local optima during training, which can negatively impact the performance of the neural network.

## Further work

1. Adaptive time constant: Instead of using a fixed time constant τ, implementing an adaptive time constant that adjusts during training might lead to better performance. This way, the model can learn the appropriate decay rate for the problem at hand, making it more flexible and better suited to various tasks.
2. Content-based weighting: Integrating content-based weighting alongside the temporal weighting could lead to a more robust activation function. This would allow the model to not only consider the age of the input but also the relevance of the input content, leading to more informed decisions about which information to prioritize.
3. Learnable clipping threshold: Instead of using a fixed maximum value V_max for the clipping mechanism, making the clipping threshold learnable could allow the model to adapt the threshold based on the specific problem and dataset. This could lead to better performance by preventing the loss of important information due to a suboptimal fixed threshold.
4. Dynamic time window: TDA currently uses a fixed time window size N. Allowing the model to learn a dynamic time window size based on the input data could improve its ability to focus on relevant information over different time scales.
5. Multi-scale TDA: Combining multiple TDAs with different time constants and time windows can capture information at various temporal scales. This would enable the model to better understand both short-term and long-term patterns in the data.
6. Incorporating gating mechanisms: Adding gating mechanisms, similar to those used in LSTMs and GRUs, can help control the flow of information within the network. This could allow the model to learn when to store or discard information, potentially leading to better overall performance.
7. Hybrid approach: Combining TDA with other activation functions or attention mechanisms could lead to an improved understanding of the data. A hybrid approach might leverage the strengths of different techniques while mitigating their respective weaknesses.
