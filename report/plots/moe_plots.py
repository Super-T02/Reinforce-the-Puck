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


def plot_all_data(a, b, c, title, x_label, y_label, output_file, window=10):
    """Plot SAC and TD3 data in one figure."""
    fig, ax = plt.subplots()

    # Plot SAC
    plot_data(ax, a, "Pretrained without finetuning experts", window)
    # Plot TD3
    plot_data(ax, b, "Trained experts from scratch", window)

    plot_data(ax, c, "Pretrained with finetuning of experts", window)

    ax.set_title(title, fontsize=plt.rcParams["axes.titlesize"])
    ax.set_xlabel(x_label, fontsize=plt.rcParams["axes.labelsize"])
    ax.set_ylabel(y_label, fontsize=plt.rcParams["axes.labelsize"])
    ax.grid(True)
    ax.legend()

    plt.savefig(f"../images/{output_file}.png")
    plt.close(fig)


# Load CSV data
predtrained_and_no_ft = pd.read_csv(
    "data/no training_experts_pretrained_moe_hockey_2025-02-20_09-46-28_100.csv"
)
from_scratch = pd.read_csv("data/no_pretrained_moe_hockey_2025-02-20_10-59-12_100.csv")
pretrained_and_ft = pd.read_csv(
    "data/train_experts_and pretrained_moe_hockey_2025-02-20_10-14-52_100.csv"
)

# Call plot_all_data twice to generate two separate images
plot_all_data(
    pretrained_and_ft,
    from_scratch,
    pretrained_and_ft,
    "Comparison of MOE training strategies",
    "Steps",
    "Evaluation Reward",
    "moe_train_strategies",
    window=10,
)
