import matplotlib.pyplot as plt
import pandas as pd
from tueplots import bundles

plt.rcParams.update({"figure.dpi": 250})
plt.rcParams.update(bundles.neurips2024(family="sans-serif"))
plt.rcParams.update(
    {
        "font.size": 15,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "legend.fontsize": 9,
        "legend.title_fontsize": 11,
    }
)

import matplotlib.pyplot as plt
import numpy as np


def plot_4(data_list, titles, x_labels, y_labels, output_file, window=10):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i in range(4):
        # Sort data by Step to ensure proper plotting
        df = data_list[i].sort_values(by="Step")
        steps = df["Step"]
        values = df["Value"]

        if window > 1:
            # Exponential smoothing for moving average and standard deviation
            moving_avg = values.ewm(span=window).mean()
            std_ewm = values.ewm(span=window).std()
            lower_bound = moving_avg - std_ewm
            upper_bound = moving_avg + std_ewm
        else:
            # Kein Smoothing, falls window <= 1
            moving_avg = values
            lower_bound = values
            upper_bound = values

        # Plot the moving average line
        axes[i].plot(steps, moving_avg, label="Moving Average", alpha=0.8)
        # Fill the error band based on the exponential standard deviation
        if window > 1:
            axes[i].fill_between(
                steps, lower_bound, upper_bound, alpha=0.2, label="Error Bounds"
            )

        axes[i].set_title(titles[i])
        axes[i].set_xlabel(x_labels[i])
        axes[i].set_ylabel(y_labels[i])
        axes[i].grid(True)
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(f"../images/{output_file}.png")


import pandas as pd


def build_stats_table(data_list, dataset_names):
    stats = []

    # Compute mean and std for each dataset and store in list of dictionaries
    for name, data in zip(dataset_names, data_list):
        mean_val = data["Value"].mean()
        std_val = data["Value"].std()
        stats.append({"Dataset": name, "Mean": mean_val, "Std": std_val})

    # Build and return the table as a DataFrame
    table = pd.DataFrame(stats)
    return table


def plot_data(data, title, x_label, y_label, output_file):
    # Read CSV file into a DataFrame

    # Create a plot of Value vs. Step
    plt.figure()
    plt.plot(data["Step"], data["Value"], marker="o", linestyle="-")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.savefig(f"../images/{output_file}.png")


foundation_eval_data = pd.read_csv("data/hockey_eval_reward_foundation_sac.csv")
selfplay_eval_data = pd.read_csv("data/hockey_eval_reward_selfplay_sac.csv")
foundation_train_data = pd.read_csv("data/hockey_train_reward_foundation_sac.csv")
selfplay_train_data = pd.read_csv("data/hockey_train_reward_selfplay_sac.csv")
# plot_data(foundation_eval_data, 'Hockey Evaluation Reward', 'Time Step', 'Eval Reward', "hockey_eval_reward_foundation_sac")
plot_4(
    [
        foundation_eval_data,
        selfplay_eval_data,
        foundation_train_data,
        selfplay_train_data,
    ],
    [
        "SAC Foundation Eval Reward",
        "SAC Selfplay Eval Reward",
        "SAC Foundation Train Reward",
        "SAC Selfplay Train Reward",
    ],
    ["Time Step", "Time Step", "Time Step", "Time Step"],
    ["Eval Reward", "Eval Reward", "Train Reward", "Train Reward"],
    "hockey_sac_rewards",
)
table = build_stats_table(
    [
        foundation_eval_data,
        selfplay_eval_data,
        foundation_train_data,
        selfplay_train_data,
    ],
    [
        "Foundation Eval Reward",
        "Selfplay Eval Reward",
        "Foundation Train Reward",
        "Selfplay Train Reward",
    ],
)
print(table)
