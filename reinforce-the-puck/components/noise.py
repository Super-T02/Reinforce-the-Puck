import abc

import numpy as np


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


class GaussianNoise(AbstractNoise):
    """Gaussian noise."""

    def __init__(self, shape: tuple[int], sigma: float = 0.1):
        """Initialize the GaussianNoise object.

        Args:
            shape (tuple[int]): Shape of the noise.
            sigma (float, optional): Sigma value. Defaults to 0.1.
        """
        self._shape = shape
        self._sigma = sigma

    def __call__(self) -> np.ndarray:
        """Call the noise.

        Returns:
            np.ndarray: Noise value.
        """
        return np.random.normal(0, self._sigma, self._shape)

    def reset(self) -> "GaussianNoise":
        """Reset the noise.

        Returns:
            GaussianNoise: Noise object.
        """
        return self

    def __repr__(self) -> str:
        """Return the string representation of the noise.

        Returns:
            str: String representation of the noise.
        """
        return f"GaussianNoise(shape={self._shape}, sigma={self._sigma})"


class ClippedGaussianNoise(AbstractNoise):
    """Clipped Gaussian noise."""

    def __init__(self, shape: tuple[int], sigma: float = 0.1, clip: float = 0.5):
        """Initialize the ClippedGaussianNoise object.

        Args:
            shape (tuple[int]): Shape of the noise.
            sigma (float, optional): Sigma value. Defaults to 0.1.
            clip (float, optional): Clip value. Defaults to 0.5.
        """
        self._shape = shape
        self._sigma = sigma
        self._clip = clip

    def __call__(self) -> np.ndarray:
        """Call the noise.

        Returns:
            np.ndarray: Noise value.
        """
        noise = np.random.normal(0, self._sigma, self._shape)
        return np.clip(noise, -self._clip, self._clip)

    def reset(self) -> "ClippedGaussianNoise":
        """Reset the noise.

        Returns:
            ClippedGaussianNoise: Noise object.
        """
        return self

    def __repr__(self) -> str:
        """Return the string representation of the noise.

        Returns:
            str: String representation of the noise.
        """
        return f"ClippedGaussianNoise(shape={self._shape}, sigma={self._sigma}, clip={self._clip})"


class ColoredNoise:
    """Generate colored noise with a given power spectral density."""

    def __init__(self, shape: tuple[int], sigma: float = 0.1, beta: float = 1.0):
        """
        Initialize the ColoredNoise object.

        Args:
            shape (tuple[int]): Shape of the noise.
            sigma (float, optional): Standard deviation of the noise. Defaults to 0.1.
            beta (float, optional): Beta value for the PSD. Defaults to 1.0 (Pink Noise).
        """
        self._shape = shape
        self._sigma = sigma
        self._beta = beta
        self.reset()

    def __call__(self) -> np.ndarray:
        """Generate colored noise.

        Returns:
            np.ndarray: Colored noise array.
        """
        # Generate white noise |-> frequency domain
        white_noise = np.random.normal(0, self._sigma, self._shape)
        fft = np.fft.rfft(white_noise)
        freqs = np.fft.rfftfreq(self._shape[0])
        freqs[0] = 1.0

        # Colorize the noise
        scaling_factors = 1 / (freqs ** (self._beta / 2.0))
        fft *= scaling_factors

        # |-> time domain
        colored_noise = np.fft.irfft(fft, n=self._shape[0])

        return colored_noise

    def reset(self) -> "ColoredNoise":
        """Reset the noise generator.

        Returns:
            ColoredNoise: The noise object.
        """
        return self

    def __repr__(self) -> str:
        """Return the string representation of the noise.

        Returns:
            str: String representation of the noise.
        """
        return (
            f"ColoredNoise(shape={self._shape}, sigma={self._sigma}, beta={self._beta})"
        )


class ClippedColoredNoise(ColoredNoise):
    """Generate clipped colored noise."""

    def __init__(
        self,
        shape: tuple[int],
        sigma: float = 0.1,
        beta: float = 1.0,
        clip: float = 0.5,
    ):
        """
        Initialize the ClippedColoredNoise object.

        Args:
            shape (tuple[int]): Shape of the noise.
            sigma (float, optional): Standard deviation of the noise. Defaults to 0.1.
            beta (float, optional): Beta value for the PSD. Defaults to 1.0 (Pink Noise).
            clip (float, optional): Clipping value. Defaults to 0.5.
        """
        super().__init__(shape, sigma, beta)
        self._clip = clip

    def __call__(self) -> np.ndarray:
        """Generate clipped colored noise.

        Returns:
            np.ndarray: Clipped colored noise array.
        """
        noise = super().__call__()
        return np.clip(noise, -self._clip, self._clip)

    def __repr__(self) -> str:
        """Return the string representation of the noise.

        Returns:
            str: String representation of the noise.
        """
        return f"ClippedColoredNoise(shape={self._shape}, sigma={self._sigma}, beta={self._beta}, clip={self._clip})"
