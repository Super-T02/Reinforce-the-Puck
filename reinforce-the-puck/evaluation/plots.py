from typing import List

import matplotlib.pyplot as plt
import numpy as np
from components.networks import Feedforward, QFunction
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def plot_value_distribution(
    function: Feedforward,
    observations: List[float],
    actions: List[any],
    label_dim1: str,
    label_dim2: str,
    plot_dim1=0,
    plot_dim2=2,
) -> plt.Figure:
    """
    Plots the value distribution of a given function based on observations and actions
    Args:
        function (Feedforward): The function used to predict values.
        observations (List[float]): A list of observation vectors.
        actions (List[any]): A list of actions corresponding to the observations.
        label_dim1 (str): Label for the x-axis.
        label_dim2 (str): Label for the y-axis.
        plot_dim1 (int, optional): Index of the observation dimension to plot on the x-axis. Defaults to 0.
        plot_dim2 (int, optional): Index of the observation dimension to plot on the y-axis. Defaults to 2.
    Returns:
        plt.Figure: The matplotlib figure object containing the plot.
    """
    values = function.predict(np.hstack([observations, actions]))
    fig = plt.figure(figsize=[10, 8])
    ax = fig.add_subplot()
    ax.scatter(
        observations[:, plot_dim1],
        observations[:, plot_dim2],
        c=values,
        cmap=cm.coolwarm,
    )
    ax.set_xlabel(label_dim1)
    ax.set_ylabel(label_dim2)

    return fig


def plot_value_function(
    value_function: Feedforward,
    x_label: str,
    y_label: str,
    xxs: np.ndarray = np.linspace(-np.pi / 2, np.pi / 2),
    yys: np.ndarray = np.linspace(-8, 8),
) -> plt.Figure:
    """Plots the value function in 3D space."""
    plot_function_in_3d_space(value_function, x_label, y_label, "Value", xxs, yys)


def plot_q_function(
    q_function: QFunction,
    x_label: str,
    y_label: str,
    xxs: np.ndarray = np.linspace(-np.pi / 2, np.pi / 2),
    yys: np.ndarray = np.linspace(-8, 8),
) -> plt.Figure:
    """Plots the Q function in 3D space."""
    plot_function_in_3d_space(q_function, x_label, y_label, "Q", xxs, yys)


def plot_function_in_3d_space(
    function: Feedforward,
    x_label: str,
    y_label: str,
    z_label: str,
    xxs: np.ndarray = np.linspace(-np.pi / 2, np.pi / 2),
    yys: np.ndarray = np.linspace(-8, 8),
) -> plt.Figure:
    """Plots a arbitrary function in 3D space."""
    XX, YY = np.meshgrid(xxs, yys)
    dots = np.asarray([XX.ravel(), YY.ravel()]).T
    values = function.predict(dots).reshape(XX.shape)

    fig = plt.figure(figsize=[10, 8])
    ax = fig.gca(projection="3d")
    surf = ax.plot_surface(
        XX, YY, values, cmap=cm.coolwarm, linewidth=0, antialiased=False
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    return fig


def plot_rewards(rewards: List[float]) -> plt.Figure:
    """
    Plots the rewards received during the episodes.
    Args:
        rewards (List[float]): A list of rewards received during the episodes.
    Returns:
        plt.Figure: The matplotlib figure object containing the plot.
    """
    fig = plt.figure(figsize=[10, 8])
    ax = fig.add_subplot()
    ax.plot(rewards)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    return fig
