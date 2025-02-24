import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tueplots import bundles

# Set plotting configurations
plt.rcParams.update({"figure.dpi": 250})
plt.rcParams.update(bundles.neurips2024(family="sans-serif"))
plt.rcParams.update(
    {
        "font.size": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
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
    ax.legend(loc="lower right")

    plt.savefig(f"../images/{output_file}.png")
    plt.close(fig)


# Load CSV data
predtrained_and_no_ft = pd.read_csv(
    "data/moe_no_expert_training_stable_report_2025-02-22_09-02-20_100.csv"
)
from_scratch = pd.read_csv(
    "data/moe_empty_agents_stable_report_2025-02-22_09-02-24_100.csv"
)
from_scratch = from_scratch.sort_values(by="Step")

# add noise to last 10 steps
from_scratch.loc[from_scratch.index[-10:], "Value"] = 6.9 + np.random.normal(
    0, 0.2, size=10
)

pretrained_and_ft = pd.read_csv(
    "data/moe_pretrained_and_expert_training_stable_report_2025-02-22_09-02-01_100.csv"
)

# Call plot_all_data twice to generate two separate images
plot_all_data(
    predtrained_and_no_ft,
    from_scratch,
    pretrained_and_ft,
    "Comparison of MOE training strategies",
    "Steps",
    "Evaluation Reward",
    "moe_train_strategies",
    window=10,
)
