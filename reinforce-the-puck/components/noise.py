import abc

import numpy as np


@abc.ABC
class AbstractNoise:
    """Abstract class for noise."""

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        """Initialize the noise."""
        pass

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> float:
        """Call the noise.

        Returns:
            float: Noise value.
        """
        pass

    @abc.abstractmethod
    def reset(self) -> "AbstractNoise":
        """Reset the noise.

        Returns:
            AbstractNoise: Noise object.
        """
        pass

    @abc.abstractmethod
    def __repr__(self) -> str:
        """Return the string representation of the noise.

        Returns:
            str: String representation of the noise.
        """
        pass


class OUNoise(AbstractNoise):
    """Ornstein-Uhlenbeck noise."""

    def __init__(self, shape: tuple[int], theta: float = 0.15, dt: float = 1e-2):
        """Initialize the OUNoise object.

        Args:
            shape (tuple[int]): Shape of the noise.
            theta (float, optional): Theta value. Defaults to 0.15.
            dt (float, optional): Time step. Defaults to 1e-2.
        """
        self._shape = shape
        self._theta = theta
        self._dt = dt

        self._noise_prev = np.zeros(self._shape)
        self.reset()

    def __call__(self) -> np.ndarray:
        """Call the noise.

        Returns:
            np.ndarray: Noise value.
        """
        noise = (
            self._noise_prev
            + self._theta * (-self._noise_prev) * self._dt
            + np.sqrt(self._dt) * np.random.normal(size=self._shape)
        )
        self._noise_prev = noise
        return noise

    def reset(self) -> "OUNoise":
        """Reset the noise.

        Returns:
            OUNoise: Noise object.
        """
        self._noise_prev = np.zeros(self._shape)
        return self

    def __repr__(self) -> str:
        """Return the string representation of the noise.

        Returns:
            str: String representation of the noise.
        """
        return f"OUNoise(shape={self._shape}, theta={self._theta}, dt={self._dt})"
