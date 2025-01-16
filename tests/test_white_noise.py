import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "reinforce-the-puck"))

from components.noise import (
    ClippedColoredNoise,
    ClippedGaussianNoise,
    ColoredNoise,
    GaussianNoise,
)

if __name__ == "__main__":
    pink_noise = ColoredNoise(shape=(1000,), sigma=0.1, beta=1.0)
    noise = pink_noise()

    clipped_pink_noise = ClippedColoredNoise(
        shape=(1000,), sigma=0.1, beta=1.0, clip=0.5
    )
    clipped_noise = clipped_pink_noise()

    gaussian_noise = GaussianNoise(shape=(1000,), sigma=0.1)
    g_noise = gaussian_noise()

    clipped_gaussian_noise = ClippedGaussianNoise(shape=(1000,), sigma=0.1, clip=0.5)
    clipped_g_noise = clipped_gaussian_noise()

    # Plot the noise
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.plot(noise, label="Pink Noise")
    plt.plot(clipped_noise, label="Clipped Pink Noise")
    plt.plot(g_noise, label="Gaussian Noise")
    plt.plot(clipped_g_noise, label="Clipped Gaussian Noise")
    plt.legend()
    plt.show()
