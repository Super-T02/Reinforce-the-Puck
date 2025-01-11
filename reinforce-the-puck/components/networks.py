from typing import Any

import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
from utils.config import global_config


class Feedforward(torch.nn.Module):
    """Simple Feedforward Neural Network with Tanh activation functions."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        activation: any = torch.nn.Tanh,
        output_activation: Any | None = None,
        loss_fn=torch.nn.SmoothL1Loss(),
        device: str = global_config.base_config.device,
        dtype: torch.dtype = global_config.base_config.dtype,
        **kwargs,
    ):
        """Initialize the Feedforward Neural Network.

        Args:
            input_size (int): Input size.
            hidden_sizes (list[int]): List of hidden layer sizes.
            output_size (int): Output size.
            activation (any, optional): Activation function. Defaults to torch.nn.Tanh.
            output_activation (any, optional): Output activation function. Defaults to None.
            loss_fn (callable, optional): Loss function. Defaults to torch.nn.SmoothL1Loss().
            device (str, optional): Device to run the model on. Defaults to config.DEVICE.
            dtype (torch.dtype, optional): Data type. Defaults to config.DTYPE.
        """

        super(Feedforward, self).__init__()
        self._input_size = input_size
        self._hidden_sizes = hidden_sizes
        self._output_size = output_size
        self._output_activation = output_activation
        self._loss_fn = loss_fn
        self._dtype = dtype
        self._device = device

        layer_sizes = [self._input_size] + self._hidden_sizes
        self._layers = torch.nn.ModuleList(
            [torch.nn.Linear(i, o) for i, o in zip(layer_sizes[:-1], layer_sizes[1:])]
        )
        self._activations = [activation() for _ in self._layers]
        self._readout = torch.nn.Linear(self._hidden_sizes[-1], self._output_size)
        self.to(self._device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = x.to(self._device)
        for layer, activation_fun in zip(self._layers, self._activations):
            x = activation_fun(layer(x))
        if self._output_activation is not None:
            return self._output_activation(self._readout(x))
        else:
            return self._readout(x)

    def predict(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Predict the output for a given input.

        Args:
            x (torch.Tensor | np.ndarray): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=self._dtype)
        x = x.to(self._device)
        with torch.no_grad():
            return self.forward(x).cpu()

    def save(self, path: str) -> "Feedforward":
        """Save the model to a file.

        Args:
            path (str): Path to save the model.

        Returns:
            Network: Network object.
        """
        torch.save(self.state_dict(), path)
        return self

    def load(self, path: str) -> "Feedforward":
        """Load the model from a file.

        Args:
            path (str): Path to load the model from.

        Returns:
            Network: Network object.
        """
        self.load_state_dict(torch.load(path))
        return self

    def get_loss(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> object:
        """Fit the model to the data.

        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): Target tensor.

        Returns:
            object: Loss object.
        """
        raise NotImplementedError


class QFunction(Feedforward):
    """Q-Function Network."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        activation: any = torch.nn.Tanh,
        output_activation: Any | None = None,
        loss_fn: callable = torch.nn.SmoothL1Loss(),
        device: str = global_config.base_config.device,
        dtype: torch.dtype = global_config.base_config.dtype,
        **kwargs,
    ):
        """Initialize the Q-Function Network.

        Args:
            input_size (int): Input size.
            hidden_sizes (list[int]): List of hidden layer sizes.
            output_size (int): Output size.
            activation (any, optional): Activation function. Defaults to torch.nn.Tanh.
            output_activation (any, optional): Output activation function. Defaults to None.
            loss_fn (callable, optional): Loss function. Defaults to torch.nn.SmoothL1Loss().
            device (str, optional): Device to run the model on. Defaults to config.DEVICE.
            dtype (torch.dtype, optional): Data type. Defaults to config.DTYPE.
        """
        super(QFunction, self).__init__(
            input_size,
            hidden_sizes,
            output_size,
            activation,
            output_activation,
            loss_fn,
            device,
            dtype,
        )

    def get_loss(self, x: torch.Tensor, y: torch.Tensor) -> object:
        """Calculate the loss for the given input and target.

        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): Target tensor.
            loss_fn (callable): Loss function.

        Returns:
            object: Loss object.
        """
        x = x.to(self._device)
        y = y.to(self._device)
        return self._loss_fn(self.forward(x), y)

    def Qvalues(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Get the Q-Values for the given observations and actions.

        Args:
            observations (torch.Tensor): Observations tensor.
            actions (torch.Tensor): Actions tensor.

        Returns:
            torch.Tensor: Q-Values tensor.
        """
        observations = observations.to(self._device)
        actions = actions.to(self._device)
        return self.forward(self.prepare(observations, actions)).cpu()

    def prepare(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Prepare the input for the Q-Function.

        Args:
            observations (torch.Tensor): Observations tensor.
            actions (torch.Tensor): Actions tensor.

        Returns:
            torch.Tensor: Input tensor.
        """
        return torch.cat((observations, actions), dim=-1)


class StochasticPolicyNetwork(Feedforward):
    """Stochastic Policy Network."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        activation: any = torch.nn.Tanh,
        output_activation: Any | None = None,
        loss_fn: callable = torch.nn.SmoothL1Loss(),
        device: str = None,
        dtype: torch.dtype = None,
        log_std_min: float = -20,
        log_std_max: float = 2,
        **kwargs,
    ):
        super().__init__(
            input_size,
            hidden_sizes,
            output_size,
            activation,
            output_activation,
            loss_fn,
            device,
            dtype,
            **kwargs,
        )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self._mean_layer = nn.Linear(self._hidden_sizes[-1], self._output_size)
        self._log_std_layer = nn.Linear(self._hidden_sizes[-1], self._output_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Pass through the Feedforward hidden layers (up to the last hidden layer)
        for layer, activation_fun in zip(self._layers, self._activations):
            x = activation_fun(layer(x))

        """
        C. Enforcing Action Boun
        https://arxiv.org/pdf/1801.01290
        """
        mean = self._mean_layer(x)
        log_std = torch.clamp(
            self._log_std_layer(x), self.log_std_min, self.log_std_max
        )

        return mean, log_std

    def predict(
        self, x: torch.Tensor | np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts the output for the given input tensor or numpy array.
        Args:
            x (torch.Tensor | np.ndarray): Input data, either as a PyTorch tensor or a NumPy array.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: The predicted output as a tuple of PyTorch tensors.
        """

        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=self._dtype)
        with torch.no_grad():
            return self.forward(x)

    def sample(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Samples an action and its log probability from a normal distribution parameterized by the network's output.

        Args:
            x (torch.Tensor): Input tensor to the network.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - action (torch.Tensor): The sampled action after applying the tanh function.
                - log_prob (torch.Tensor): The log probability of the sampled action.
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=self._dtype)
        mean, log_std = self.forward(x)
        std = log_std.exp()

        normal_dist = dist.Normal(mean, std)

        # reparameterization trick
        z = normal_dist.rsample()
        action = torch.tanh(z)
        log_prob = normal_dist.log_prob(z).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        return action, log_prob

    def mean(self, x: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(x)
        return torch.tanh(mean)
