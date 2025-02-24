import os
import random
from turtle import position

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tueplots import bundles

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


def load_tensorboard_data(base_dir, env_name):
    """
    Load TensorBoard CSV data into a single DataFrame.

    Args:
        base_dir (str): The base directory containing metric folders and CSV files.

    Returns:
        pd.DataFrame: A DataFrame containing all the data with additional columns for metric, runname, and algorithm.
    """
    data_frames = []

    # Walk through the directory structure
    for metric in os.listdir(base_dir):
        metric_path = os.path.join(base_dir, metric)
        if os.path.isdir(metric_path):
            for file in os.listdir(metric_path):
                if file.endswith(".csv"):
                    file_path = os.path.join(metric_path, file)
                    df = pd.read_csv(file_path)

                    # Split in runname and timestamp
                    runname_timestamp = os.path.splitext(file)[0]
                    runname = runname_timestamp.split("_")[0]
                    split = runname.split("-")

                    alg = split[0]
                    buffer = "er"
                    if len(split) > 1:
                        buffer = split[1]

                    if buffer in ["bper"]:
                        continue

                    df["Metric"] = metric
                    df["RunName"] = runname_timestamp
                    df["Algorithm"] = alg
                    df["Buffer"] = buffer
                    df["Algorithm-Buffer"] = f"{alg}-{buffer}"

                    data_frames.append(df)

    combined_df = pd.concat(data_frames, ignore_index=True)
    combined_df["Environment"] = env_name
    return combined_df


# Create a combined DataFrame
hockey = load_tensorboard_data("HockeyEval", "Hockey Own Reward")
basic_reward = load_tensorboard_data("BasicRewardEval", "Hockey Original Reward")
noise = load_tensorboard_data("NoiseEval", "Hockey Own Reward")
combined_data = pd.concat([hockey, basic_reward, noise], ignore_index=True).reset_index(
    drop=True
)
copy = combined_data.copy()

# Devide critic loss of sac by 2
copy.loc[(copy["Algorithm"] == "sac") & (copy["Metric"] == "Loss"), "Value"] = (
    copy.loc[(copy["Algorithm"] == "sac") & (copy["Metric"] == "Loss"), "Value"] / 2
)


# Generate plots
buffer2name = {
    "er": "Own",
    "per": "PER",
    "ber": "BER",
    "bper": "BPER",
    "white": "White",
    "blue": "Blue",
    "original": "Original",
}

# Metrics
metric2window = {
    "Reward": 100,
    "EvalReward": 100,
    "Loss": 100,
    "ActorLoss": 100,
    "AlphaLoss": 100,
}

metric2name = {
    "Reward": "Reward",
    "EvalReward": "Eval Reward",
    "Loss": "Critic Loss",
    "ActorLoss": "Actor Loss",
    "AlphaLoss": "Alpha Loss",
}

# Sorting order
sort_order = {
    "original": 0,
    "white": 1,
    "blue": 2,
    "er": 3,
    "per": 4,
    "ber": 5,
}

metric2axis = {
    "Reward": ["linear", "linear"],
    "EvalReward": ["linear", "linear"],
    "Loss": ["linear", "linear"],
    "ActorLoss": ["linear", "linear"],
    "AlphaLoss": ["linear", "linear"],
}

metric2ylim = {
    "Reward": [-50, 15],
    "EvalReward": [-11, 12],
    "Loss": [0, 1],
    "ActorLoss": [-5, 5],
    "AlphaLoss": [-5, 1],
}

alg2line = {
    "sac": "-",
    "td3": "-.",
}


# Plotting
def plot_metrics(data, b2n, m2w, metrics, environments):
    for e, env in enumerate(environments):
        env_data = data[data["Environment"] == env]
        max_steps = max(env_data.groupby("Algorithm-Buffer")["Step"].max())
        for i, metric in enumerate(metrics):
            fig, ax = plt.subplots(1, 1)
            metric_data = env_data[env_data["Metric"] == metric]
            axs = ax
            metric_data = metric_data.sort_values(
                by=["Algorithm", "Buffer"], key=lambda x: x.map(sort_order)
            )
            for alg_buffer in metric_data["Algorithm-Buffer"].unique():
                alg_data = metric_data[
                    metric_data["Algorithm-Buffer"] == alg_buffer
                ].sort_values(by="Step")

                buffer = alg_buffer.split("-")[1]
                alg = alg_buffer.split("-")[0]
                name = f"{alg.upper()} ({b2n[buffer]})"
                window = m2w[metric]

                if buffer in ["bper", "white", "blue"]:
                    continue

                # Exponential smoothing
                moving_avg = alg_data["Value"].ewm(span=window).mean()
                bounds = (
                    moving_avg - alg_data["Value"].ewm(span=window).std(),
                    moving_avg + alg_data["Value"].ewm(span=window).std(),
                )

                if window <= 1:
                    # Smoothing not possible
                    moving_avg = alg_data["Value"]

                axs.plot(
                    alg_data["Step"],
                    moving_avg,
                    label=name,
                    alpha=0.8,
                    linestyle=alg2line[alg],
                )

                if window > 1:
                    axs.fill_between(
                        alg_data["Step"],
                        bounds[0],
                        bounds[1],
                        alpha=0.2,
                        linestyle=alg2line[alg],
                    )
            axs.set_xlabel("Step")
            axs.set_ylabel(metric2name[metric])
            axs.set_xscale(metric2axis[metric][0])
            axs.set_yscale(metric2axis[metric][1])
            axs.set_ylim(metric2ylim[metric])
            axs.set_title(f"{metric2name[metric]} over Steps for {env}")
            axs.grid()
            axs.legend()
            plt.savefig(f"../images/baseline_{env}_{metric}.png")


def create_avg_reward_table_latex(data):
    """
    Create a table with the average reward for each environment and algorithm,
    using only the last 100 steps, round values to 2 decimal places, format the
    best scores in bold, sort by custom order, and export it to LaTeX.

    Args:
        data (pd.DataFrame): The combined DataFrame containing all environments and metrics.

    Returns:
        str: A LaTeX-formatted table as a string.
    """
    order = {
        "sac": 0,
        "td3": 1,
        "sac-er": 2,
        "td3-er": 3,
        "td3-white": 4,
        "td3-blue": 5,
        "td3-per": 6,
        "td3-ber": 7,
    }

    # Filter for the "Reward" metric
    reward_data = data[data["Metric"] == "EvalReward"]

    # Filter to include only the last 100 steps for each Environment and Algorithm-Buffer
    last_100_steps = (
        reward_data.groupby(["Environment", "Algorithm-Buffer"])
        .apply(lambda group: group.nlargest(100, "Step"))
        .reset_index(drop=True)
    )

    # Group by Environment and Algorithm-Buffer, then calculate the mean reward
    avg_rewards = (
        last_100_steps.groupby(["Environment", "Algorithm-Buffer"])["Value"]
        .mean()
        .reset_index()
    )

    # Round the values to 2 decimal places
    avg_rewards["Value"] = avg_rewards["Value"].round(2)

    # Pivot the table for better readability
    reward_table = avg_rewards.pivot(
        index="Environment", columns="Algorithm-Buffer", values="Value"
    )

    # Merge alg-original and alg-er columns
    reward_table["sac"] = reward_table["sac-original"]
    reward_table["td3"] = reward_table["td3-original"]
    reward_table.drop(columns=["sac-original", "td3-original"], inplace=True)
    reward_table["sac"] = reward_table["sac"].combine_first(reward_table["sac-er"])
    reward_table["td3"] = reward_table["td3"].combine_first(reward_table["td3-er"])
    reward_table.drop(columns=["sac-er", "td3-er"], inplace=True)

    reward_table = reward_table[
        [col for col in sorted(reward_table.columns, key=lambda x: order[x])]
    ]

    # Rename columns for better readability
    for alg in reward_table.columns:
        if "-" in alg:
            reward_table.rename(
                columns={
                    alg: f"{alg.split('-')[0].upper()} ({buffer2name[alg.split('-')[1]]})"
                },
                inplace=True,
            )
        else:
            reward_table.rename(columns={alg: alg.upper()}, inplace=True)

    # Reset index for a clean table
    reward_table.reset_index(inplace=True)

    # Format the best scores per environment in bold
    for index, row in reward_table.iterrows():
        max_value = row[1:].max()  # Exclude the 'Environment' column
        for col in reward_table.columns[1:]:
            if row[col] == max_value:
                reward_table.at[index, col] = f"\\textbf{{{row[col]:.2f}}}"
            else:
                reward_table.at[index, col] = (
                    f"{row[col]:.2f}" if not np.isnan(row[col]) else "-"
                )

    # Convert the table to LaTeX format
    latex_table = reward_table.to_latex(
        index=False,  # Do not include the index in the LaTeX table
        caption="Average Evaluation Reward for Hockey Environment}\\centering {",
        label="tab:avg_reward",
        column_format="l" + "r" * (len(reward_table.columns) - 1),  # Align columns
        escape=False,  # Allow LaTeX symbols like \textbf{}
        position="H",
        na_rep="-",
    )

    return latex_table


hockey = copy.copy()
hockey["Environment"] = "Hockey"

plot_metrics(
    hockey,
    buffer2name,
    metric2window,
    ["Reward", "EvalReward", "Loss", "ActorLoss", "AlphaLoss"],
    ["Hockey"],
)

reward_table = create_avg_reward_table_latex(copy)
with open("../tables/avg_reward_table_hockey.tex", "w") as f:
    f.write(reward_table)
