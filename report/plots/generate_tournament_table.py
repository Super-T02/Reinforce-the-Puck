import pandas as pd
from openskill.models import PlackettLuce, PlackettLuceRating

# Load the game results from the CSV file
file_path = "../../logs/tournaments/tournament_results2025-02-25_13-22-23.csv"
results = pd.read_csv(file_path)

model = PlackettLuce()


class Player:
    def __init__(self, name: str, rate_obj: PlackettLuceRating):
        self.name: str = name
        self.type = name.split("_")[0].split("-")[0]
        self.rate_obj = rate_obj


def map_agent_name(agent_name):
    """
    Maps an agent name to a generalized category based on its naming pattern.
    """
    # Check for specific patterns in the agent name
    if agent_name.startswith("td3-") or agent_name.startswith("td3_"):
        if "white" in agent_name:
            return "TD3 (White)"
        elif "blue" in agent_name:
            return "TD3 (Blue)"
        elif "ber" in agent_name:
            return "TD3 (BER)"
        elif "per" in agent_name:
            return "TD3 (PER)"
        elif "foundation" in agent_name:
            return "TD3 (Foundation)"
        elif "selfplay" in agent_name:
            return "TD3 (Selfplay)"
        elif "er" in agent_name:
            return "TD3 (Own)"
        else:
            return "TD3 (Original)"
    elif agent_name.startswith("sac-") or agent_name.startswith("sac_"):
        if "own" in agent_name:
            return "SAC (Own)"
        elif "foundation" in agent_name:
            return "SAC (Foundation)"
        elif "selfplay" in agent_name:
            return "SAC (Selfplay)"
        elif "ft" in agent_name:
            return "SAC (Fine-Tuned)"
        else:
            return "SAC (Original)"
    elif agent_name.startswith("moe-") or agent_name.startswith("moe_"):
        if "selfplay" in agent_name:
            return "Hybrid Agent (Selfplay)"
        elif "v" in agent_name:
            version = agent_name.split("v")[-1]
            return "Hybrid Agent (v{})".format(version)
        elif "ft" in agent_name:
            return "Hybrid Agent (Fine-Tuned)"
        elif "expert" in agent_name:
            return "Hybrid Agent (Expert Pretrain)"
        else:
            return "Hybrid Agent"
    elif agent_name.startswith("BASIC_OPPONENT"):
        if "WEAK" in agent_name:
            return "Basic Opponent (Weak)"
        elif "STRONG" in agent_name:
            return "Basic Opponent (Strong)"
        else:
            return "Basic Opponent (Original)"
    else:
        return "Unknown Agent"


# Add players
players = {}

names = list(set(results["p1"].unique()).union(set(results["p2"].unique())))
for name in names:
    players[name] = Player(name, model.rating(name=name))

# Add games
for _, row in results.iterrows():
    p1 = players[row["p1"]]
    p2 = players[row["p2"]]

    p1_score, p2_score = int(row["p1_score"]), int(row["p2_score"])
    p1_new, p2_new = model.rate(
        [[p1.rate_obj], [p2.rate_obj]], scores=[p1_score, p2_score]
    )
    p1.rate_obj = p1_new[0]
    p2.rate_obj = p2_new[0]
    players[row["p1"]] = p1
    players[row["p2"]] = p2

# Generate leaderboard table
leaderboard = pd.DataFrame(
    [
        {
            "Player": player.name,
            "Type": player.type,
            "Rating": player.rate_obj.mu - player.rate_obj.sigma,
            "$\mu$": player.rate_obj.mu,
            "$\sigma$": player.rate_obj.sigma,
        }
        for player in players.values()
    ]
)
leaderboard = leaderboard.sort_values(by="Rating", ascending=False)
leaderboard = leaderboard.reset_index(drop=True)
type2type = {
    "moe": "Hybrid Agent",
    "td3": "TD3",
    "sac": "SAC",
    "BASIC": "Basic Opponent",
}
leaderboard["Type"] = leaderboard["Type"].apply(lambda x: type2type[x])
leaderboard["Player"] = leaderboard["Player"].apply(map_agent_name)

# Add rank
leaderboard["Rank"] = leaderboard.index + 1
leaderboard = leaderboard[["Rank", "Player", "Type", "Rating", "$\mu$", "$\sigma$"]]
print(leaderboard)

leaderboard.to_latex(
    "../tables/tournament_leaderboard.tex",
    index=False,
    escape=False,
    float_format="%.2f",
    caption="Leaderboard of the tournament} \\centering {",
    label="tab:tournament_leaderboard",
    position="H",
)
