import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tueplots import bundles

# Anpassung der Matplotlib-Konfiguration
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


def plot_separate(sac_df, td3_df, title, x_label, y_label, output_file, window=10):
    """
    Erstellt einen Plot, der SAC- und TD3-Daten miteinander vergleicht.
    Die Daten werden geglättet (Exponentielles gleitendes Mittel), und
    der dazugehörige Standardabweichungsbereich wird als Fläche markiert.

    Parameter:
    -----------
    sac_df: DataFrame
        Enthält die Daten für SAC (Spalten: "Step", "Value").
    td3_df: DataFrame
        Enthält die Daten für TD3 (Spalten: "Step", "Value").
    title: str
        Titel des Plots.
    x_label: str
        Beschriftung der x-Achse.
    y_label: str
        Beschriftung der y-Achse.
    output_file: str
        Dateiname (Pfad) für das zu speichernde Plot-Bild (PNG).
    window: int
        Span-Wert für die exponentielle Glättung. Bei 1 keine Glättung.
    """
    fig, ax = plt.subplots()

    # Sortierung der Daten nach Step
    sac_df = sac_df.sort_values(by="Step")
    if not td3_df is None:
        td3_df = td3_df.sort_values(by="Step")

    # SAC-Daten
    steps_sac = sac_df["Step"]
    values_sac = sac_df["Value"]
    if window > 1:
        moving_avg_sac = values_sac.ewm(span=window).mean()
        std_sac = values_sac.ewm(span=window).std()
        lower_sac = moving_avg_sac - std_sac
        upper_sac = moving_avg_sac + std_sac
    else:
        moving_avg_sac = values_sac
        lower_sac = values_sac
        upper_sac = values_sac

    ax.plot(steps_sac, moving_avg_sac, label="SAC", alpha=0.8, linewidth=1)
    if window > 1:
        ax.fill_between(steps_sac, lower_sac, upper_sac, alpha=0.2)

    # TD3-Daten
    if not td3_df is None:
        steps_td3 = td3_df["Step"]
        values_td3 = td3_df["Value"]
        if window > 1:
            moving_avg_td3 = values_td3.ewm(span=window).mean()
            std_td3 = values_td3.ewm(span=window).std()
            lower_td3 = moving_avg_td3 - std_td3
            upper_td3 = moving_avg_td3 + std_td3
        else:
            moving_avg_td3 = values_td3
            lower_td3 = values_td3
            upper_td3 = values_td3

        ax.plot(steps_td3, moving_avg_td3, label="TD3", alpha=0.8, linewidth=1)
        if window > 1:
            ax.fill_between(steps_td3, lower_td3, upper_td3, alpha=0.2)

    # Beschriftungen und Titel
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

    # Speichern und Schließen des Plots
    plt.savefig(f"../images/{output_file}.png")
    plt.close()


sac_foundation_eval_data = pd.read_csv("data/hockey_eval_reward_foundation_sac.csv")
sac_foundation_train_data = pd.read_csv("data/hockey_train_reward_foundation_sac.csv")


sac_foundation_actor_loss = pd.read_csv(
    "data/actor_loss agent_sac_hokey_2025-01-30_21-31-12_100.csv"
)
sac_foundation_critic_loss = pd.read_csv(
    "data/critic_loss_agent_sac_hokey_2025-01-30_21-31-12_100.csv"
)

sac_foundation_alpha_loss = pd.read_csv(
    "data/alpha_loss agent_sac_hokey_2025-01-30_21-31-12_100.csv"
)

# Platzhalter-Datenframes für TD3 (hier nur 1er-Werte als Demonstration)
td3_foundation_eval_data = pd.DataFrame(
    {
        "Step": sac_foundation_eval_data["Step"],
        "Value": np.ones(len(sac_foundation_eval_data)),
    }
)
td3_foundation_train_data = pd.DataFrame(
    {
        "Step": sac_foundation_train_data["Step"],
        "Value": np.ones(len(sac_foundation_train_data)),
    }
)


# 1. Plot: Eval Rewards
plot_separate(
    sac_foundation_eval_data,
    td3_foundation_eval_data,
    title="Foundation Training - Evaluation",
    x_label="Time Step",
    y_label="Reward",
    output_file="foundation_rewards_eval",
    window=10,
)

# 2. Plot: Train Rewards
plot_separate(
    sac_foundation_train_data,
    td3_foundation_train_data,
    title="Foundation Training - Training",
    x_label="Time Step",
    y_label="Reward",
    output_file="foundation_rewards_train",
    window=10,
)

# 3. Plot: Actor Loss
plot_separate(
    sac_foundation_actor_loss,
    td3_foundation_eval_data,  # Nur Platzhalter für TD3
    title="Foundation Training - Actor Loss",
    x_label="Time Step",
    y_label="Actor Loss",
    output_file="foundation_actor_loss",
    window=10,
)

# 4. Plot: Critic Loss
plot_separate(
    sac_foundation_critic_loss,
    td3_foundation_train_data,  # Nur Platzhalter für TD3
    title="Foundation Training - Critic Loss",
    x_label="Time Step",
    y_label="Critic Loss",
    output_file="foundation_critic_loss",
    window=10,
)

# 5. Plot: Alpha Loss
plot_separate(
    sac_foundation_alpha_loss,
    None,
    title="Foundation Training - Alpha Loss",
    x_label="Time Step",
    y_label="Alpha Loss",
    output_file="foundation_alpha_loss",
    window=10,
)
