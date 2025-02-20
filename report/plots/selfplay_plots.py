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


def plot_data(ax, datasets, subplot_title, x_label, y_label, window):
    for df, label in datasets:
        df = df.sort_values(by="Step")
        steps = df["Step"]
        values = df["Value"]

        if window > 1:
            moving_avg = values.ewm(span=window).mean()
            std_ewm = values.ewm(span=window).std()
            lower_bound = moving_avg - std_ewm
            upper_bound = moving_avg + std_ewm
        else:
            moving_avg = values
            lower_bound = values
            upper_bound = values

        ax.plot(steps, moving_avg, label=label, alpha=0.8, linewidth=1)
        if window > 1:
            ax.fill_between(steps, lower_bound, upper_bound, alpha=0.2)

    ax.set_title(subplot_title, fontsize=plt.rcParams["axes.titlesize"])
    ax.set_xlabel(x_label, fontsize=plt.rcParams["axes.labelsize"])
    ax.set_ylabel(y_label, fontsize=plt.rcParams["axes.labelsize"])
    ax.grid(True)
    ax.legend()


def plot_all_data(
    sac_eval,
    sac_train,
    td3_eval,
    td3_train,
    title,
    x_label,
    y_label,
    output_file,
    window=10,
):
    train_datasets = [(sac_train, "SAC Train"), (td3_train, "TD3 Train")]
    eval_datasets = [(sac_eval, "SAC Eval"), (td3_eval, "TD3 Eval")]

    # Plot training data in a separate figure
    fig_train, ax_train = plt.subplots()
    plot_data(ax_train, train_datasets, "Training", x_label, y_label, window)
    ax_train.set_title("Training " + title, fontsize=plt.rcParams["axes.titlesize"])
    plt.savefig(f"../images/{output_file}_train.png")
    plt.close(fig_train)

    # Plot evaluation data in a separate figure
    fig_eval, ax_eval = plt.subplots()
    plot_data(ax_eval, eval_datasets, "Evaluation", x_label, y_label, window)
    ax_eval.set_title("Evaluation " + title, fontsize=plt.rcParams["axes.titlesize"])
    plt.savefig(f"../images/{output_file}_eval.png")
    plt.close(fig_eval)


# Load CSV data
sac_selfplay_eval_data = pd.read_csv("data/eval_sac_multi_2025-02-07_22-23-56_100.csv")
sac_selfplay_train_data = pd.read_csv(
    "data/train_sac_multi_2025-02-07_22-23-56_100.csv"
)
td3_selfplay_eval_data = pd.read_csv("data/eval_td3_multi_2025-02-07_22-23-52_100.csv")
td3_selfplay_train_data = pd.read_csv(
    "data/train_td3_multi_2025-02-07_22-23-52_100.csv"
)

plot_all_data(
    sac_selfplay_eval_data,
    sac_selfplay_train_data,
    td3_selfplay_eval_data,
    td3_selfplay_train_data,
    "Self-Play Training",
    "Steps",
    "Reward",
    "selfplay_rewards",
)
