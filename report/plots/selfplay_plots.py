import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tueplots import bundles

# Set plotting configurations
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


def plot_data(ax, df, label, window):
    """Plot single dataset with optional EWM smoothing and band."""
    df = df.sort_values(by="Step")
    steps = df["Step"]
    values = df["Value"]

    if window > 1:
        moving_avg = values.ewm(span=window).mean()
        std_ewm = values.ewm(span=window).std()
        lower_bound = moving_avg - std_ewm
        upper_bound = moving_avg + std_ewm
        ax.fill_between(steps, lower_bound, upper_bound, alpha=0.2)
    else:
        moving_avg = values

    ax.plot(steps, moving_avg, label=label, alpha=0.8, linewidth=1)


def plot_all_data(sac_df, td3_df, title, x_label, y_label, output_file, window=10):
    """Plot SAC and TD3 data in one figure."""
    fig, ax = plt.subplots()

    # Plot SAC
    plot_data(ax, sac_df, "SAC", window)
    # Plot TD3
    plot_data(ax, td3_df, "TD3", window)

    ax.set_title(title, fontsize=plt.rcParams["axes.titlesize"])
    ax.set_xlabel(x_label, fontsize=plt.rcParams["axes.labelsize"])
    ax.set_ylabel(y_label, fontsize=plt.rcParams["axes.labelsize"])
    ax.grid(True)
    ax.legend()

    plt.savefig(f"../images/{output_file}.png")
    plt.close(fig)


# Load CSV data
sac_selfplay_eval_data = pd.read_csv("data/eval_sac_multi_2025-02-07_22-23-56_100.csv")
sac_selfplay_train_data = pd.read_csv(
    "data/train_sac_multi_2025-02-07_22-23-56_100.csv"
)
td3_selfplay_eval_data = pd.read_csv("data/eval_td3_multi_2025-02-07_22-23-52_100.csv")
td3_selfplay_train_data = pd.read_csv(
    "data/train_td3_multi_2025-02-07_22-23-52_100.csv"
)

# Call plot_all_data twice to generate two separate images
plot_all_data(
    sac_selfplay_train_data,
    td3_selfplay_train_data,
    "Self-Play - Training",
    "Steps",
    "Reward",
    "selfplay_rewards_train",
    window=10,
)

plot_all_data(
    sac_selfplay_eval_data,
    td3_selfplay_eval_data,
    "Self-Play - Evaluation",
    "Steps",
    "Reward",
    "selfplay_rewards_eval",
    window=10,
)
