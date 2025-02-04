from typing import Any

import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
from utils.config import global_config


class BatchRenorm1d(nn.Module):
    """
    Batch Renormalization (a better version of Batch Normalization introduced by:
    Ioffe, Sergey. "Batch renormalization: Towards reducing minibatch dependence in batch-normalized models." Advances in neural information processing systems 30 (2017)
    https://proceedings.neurips.cc/paper/2017/file/c54e7837e0cd0ced286cb5995327d1ab-Paper.pdf
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.99, r_max=5.0, d_max=3.0):
        """
        Initialize the BatchRenorm1d layer.

        Args:
            num_features (int): Number of features.
            eps (float, optional): Epsilon for numerical stabilities (sqrt). Defaults to 1e-5.
            momentum (float, optional): Keep previous smoothing. Defaults to 0.99.
            r_max (float, optional): Maximal value for r. Defaults to 5.0.
            d_max (float, optional): Maximal value for d. Defaults to 3.0.
        """
        super(BatchRenorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.r_max = r_max  # Maximum allowed value for r
        self.d_max = d_max  # Maximum allowed value for d

        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        # Running statistics
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

        # Initialize r and d
        self.r = 1.0
        self.d = 0.0

    def forward(self, x):
        """
        Forward pass through the BatchRenorm1d layer.

        Args:
            x (torch.Tensor): Input tensor.

        Raises:
            ValueError: If input is not 2D.

        Returns:
            torch.Tensor: Output tensor.
        """
        if self.training:
            # Ensure input is 2D (batch_size, num_features)
            if x.dim() != 2:
                raise ValueError(f"Expected input to be 2D, but got {x.dim()}D tensor")

            # Compute batch statistics over [batch_size]
            batch_mean = x.mean(0, keepdim=True)  # Shape: [1, num_features]
            batch_var = x.var(
                0, keepdim=True, unbiased=False
            )  # Shape: [1, num_features]

            # Update running statistics
            self.running_mean = (
                self.momentum * batch_mean.squeeze()
                + (1 - self.momentum) * self.running_mean
            ).detach()
            self.running_var = (
                self.momentum * batch_var.squeeze()
                + (1 - self.momentum) * self.running_var
            ).detach()

            # Compute r and d as per the paper
            r = (batch_var + self.eps).sqrt() / (
                self.running_var + self.eps
            ).sqrt().detach()
            r = torch.clamp(
                r, 1 / self.r_max, self.r_max
            ).detach()  # Clamp r to [1/r_max, r_max]

            d = (batch_mean - self.running_mean) / (
                self.running_var + self.eps
            ).sqrt().detach()
            d = torch.clamp(
                d, -self.d_max, self.d_max
            ).detach()  # Clamp d to [-d_max, d_max]

            # Apply batch renormalization
            x_hat = (x - batch_mean) / (batch_var + self.eps).sqrt().detach()
            x_hat = x_hat * r + d
        else:
            # During inference, use running statistics
            x_hat = (x - self.running_mean) / (self.running_var + self.eps).sqrt()

        # Apply learnable scale and shift
        return self.weight * x_hat.detach() + self.bias


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

        self.use_batch_norm = kwargs.get("use_batch_norm", False)
        self._batch_norms = torch.nn.ModuleList(
            [BatchRenorm1d(size) for size in layer_sizes[1:]]
            if self.use_batch_norm
            else [None] * len(layer_sizes)
        )

        self.to(self._device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = x.to(self._device)
        for layer, batch_norm, activation_fun in zip(
            self._layers, self._batch_norms, self._activations
        ):
            x = layer(x)
            if batch_norm is not None and x.dim() > 1:
                x = batch_norm(x)
            x = activation_fun(x)
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
        if isinstance(x, np.ndarray) or isinstance(x, list):
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
        self.load_state_dict(torch.load(path, weights_only=False))
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
            **kwargs,
        )

    def get_loss(
        self, x: torch.Tensor, y: torch.Tensor, do_forward: bool = True
    ) -> object:
        """Calculate the loss for the given input and target.

        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): Target tensor.
            loss_fn (callable): Loss function.
            do_forward(bool, optional): Whether to do a forward pass. Defaults to True.

        Returns:
            object: Loss object.
        """
        x = x.to(self._device)
        y = y.to(self._device)
        return self._loss_fn(self.forward(x), y) if do_forward else self._loss_fn(x, y)

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
        dtype: torch.dtype = global_config.base_config.dtype,
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
        self._device = device

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self._mean_layer = nn.Linear(self._hidden_sizes[-1], self._output_size)
        self._log_std_layer = nn.Linear(self._hidden_sizes[-1], self._output_size)
        self.to(self._device)

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

        x = x.to(self._device)
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

        x = x.to(self._device)
        mean, log_std = self.forward(x)
        std = log_std.exp()

        normal_dist = dist.Normal(mean, std)

        # reparameterization trick
        z = normal_dist.rsample()
        action = torch.tanh(z)
        log_prob = normal_dist.log_prob(z).sum(dim=-1, keepdim=True)
        # C. Enforcing Action Bounds
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(
            dim=-1, keepdim=True
        )  # 1e-6 prevent log(0)
        return action, log_prob

    def mean(self, x: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(x)
        return torch.tanh(mean)


class HLGaussQFunction(QFunction):
    """Uses a HL Gaussian Q-Function to model the Q-Values as a Categorial Distribution."""

    def __init__(
        self,
        input_size,
        hidden_sizes,
        output_size,
        activation=torch.nn.Tanh,
        output_activation=None,
        device=global_config.base_config.device,
        dtype=global_config.base_config.dtype,
        num_bins: int = 51,
        v_min: float = -10,
        v_max: float = 10,
        sigma_ratio: float = 0.75,
        **kwargs,
    ):
        super(HLGaussQFunction, self).__init__(
            input_size,
            hidden_sizes,
            num_bins,  # Output size is the number of bins
            activation,
            output_activation,
            self.hl_cross_entropy,
            device,
            dtype,
            **kwargs,
        )
        self.v_min = v_min
        self.v_max = v_max
        self.num_bins = num_bins
        self.sigma = ((v_max - v_min) / num_bins) * sigma_ratio
        self.support = torch.linspace(v_min, v_max, num_bins + 1).to(device)
        self.goal_output_size = output_size

    def hl_cross_entropy(self, logits, target):
        """
        Compute the cross-entropy loss for the HL Gaussian Q-Function.

        Args:
            probs (torch.Tensor): Predicted Probabilities.
            target (torch.Tensor): Target values.

        Returns:
            torch.Tensor: Cross-entropy loss.
        """
        loss = torch.nn.functional.cross_entropy(
            logits.squeeze(), self.transform_to_probs(target.squeeze())
        )

        # Debug
        # print("Shape of logits: ", logits.shape)
        # print("Min/Max Logits: ", torch.min(logits), torch.max(logits))
        # print("Shape of target: ", target.shape)
        # print("Min/Max Target: ", torch.min(target), torch.max(target))
        # print("Shape of probs: ", self.transform_to_probs(target).shape)
        # print("Shape of loss: ", loss.shape)
        # print(
        #     "Min/Max Prob: ",
        #     torch.min(self.transform_to_probs(target)),
        #     torch.max(self.transform_to_probs(target)),
        # )
        # print("Loss: ", loss)
        if torch.isnan(loss).any():
            raise ValueError("Loss is NaN")

        return loss

    def transform_to_probs(self, target: torch.Tensor) -> torch.Tensor:
        """Transform the target tensor to probs.

        Args:
            target (torch.Tensor): Target tensor.

        Returns:
            torch.Tensor: Transformed tensor.
        """
        # Put all on the same device
        target = target.to(self._device)  # Shape: [batch_size, batch_size]
        support = self.support.to(self._device)  # Shape: [num_bins]

        cdf_evals = torch.special.erf(
            (support - target.unsqueeze(-1))
            / (torch.sqrt(torch.tensor(2.0)) * self.sigma)
        )
        z = cdf_evals[..., -1] - cdf_evals[..., 0]
        bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]

        # Debug
        # print("Shape of target: ", target.shape)
        # print("Shape of support: ", support.shape)
        # print("Shape of cdf_evals: ", cdf_evals.shape)
        # print("Shape of z: ", z.shape)
        # print("Any NaNs in z: ", torch.isnan(z).any())

        return bin_probs / (z.unsqueeze(-1) + 1e-5)

    def transform_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """Transform the probs tensor to target.

        Args:
            probs (torch.Tensor): Probs tensor.

        Returns:
            torch.Tensor: Transformed tensor.
        """
        centers = (self.support[:-1] + self.support[1:]) / 2
        return torch.sum(probs * centers, dim=-1)

    def Qvalues(self, observations, actions):
        probs = super().Qvalues(observations, actions).to(self._device)
        return self.transform_from_probs(probs).cpu().reshape(-1, self.goal_output_size)

    def forward(self, x):
        result = super().forward(x).to(self._device)
        probs = torch.softmax(result, dim=-1)
        return probs
