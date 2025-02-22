import os

import pandas as pd
import yaml


def parse_yaml_to_latex_tables_with_pandas(yaml_file):
    """
    Parse a YAML file and generate LaTeX tables using Pandas' to_latex method.

    Args:
        yaml_file (str): Path to the YAML file.

    Returns:
        str: A string containing LaTeX tables for all environments.
    """
    # Load the YAML file
    with open(yaml_file, "r") as file:
        data = yaml.safe_load(file)

    # Map environment IDs to environment names
    env_map = {0: "Pendulum", 1: "LunarLander", 2: "Hockey", 3: "HalfCheetah"}

    # Group agents by environment
    env_agents = {env: [] for env in env_map.values()}
    for key, value in data.items():
        if key.startswith("agent"):
            env_id = value.get("env_id", None)
            if env_id is not None:
                env_name = env_map.get(env_id, "Unknown")
                env_agents[env_name].append((key, value))

    # Generate LaTeX tables for each environment
    latex_tables = ""
    for env_name, agents in env_agents.items():
        if not agents:
            continue

        # Create a DataFrame for the current environment
        rows = []
        for agent_name, agent_data in agents:
            algorithm = agent_data.get("type", "N/A").upper()
            noise = (
                f"$\\sigma$={agent_data.get('noise_sigma', 'N/A')}, clip={agent_data.get('noise_clip', 'N/A')}, $\\beta$={agent_data.get('noise_beta', 'N/A')}"
                if "noise_sigma" in agent_data
                else "N/A"
            )
            actor_sizes = agent_data.get("actor_hidden_sizes", [])
            critic_sizes = agent_data.get("critic_hidden_sizes", [])
            buffer_type = agent_data.get("buffer_type", "ER")
            discount = agent_data.get("discount", "N/A")

            # Append a row to the DataFrame
            rows.append(
                {
                    "Agent": algorithm,
                    "Noise": noise,
                    "Actor Sizes": actor_sizes,
                    "Critic Sizes": critic_sizes,
                    "Buffer Type": buffer_type,
                    "Discount": discount,
                }
            )

        # Convert rows to a Pandas DataFrame
        df = pd.DataFrame(rows)

        # Generate LaTeX table using Pandas' to_latex method
        latex_table = df.to_latex(
            index=False,  # Do not include the index column
            caption="Hyperparameter Summary for " + env_name + " }\\centering {",
            label=f"tab:hyperparameters_{env_name.lower()}",
            column_format="llllll",  # Align columns with vertical lines
            float_format="%.2f",  # Round values to 2 decimal places
            escape=False,  # Allow LaTeX symbols like $ in the table
            position="H",  # Place the table "Here"
        )

        # Append the LaTeX table to the result
        latex_tables += latex_table + "\n\n"

    return latex_tables


# Use the function to generate the LaTeX tables
yaml_file = "../../config/evaluation/simple_evaluation.yaml"
latex_tables = parse_yaml_to_latex_tables_with_pandas(yaml_file)
goal = "../tables/hyperparameters_tables.tex"

os.makedirs(os.path.dirname(goal), exist_ok=True)

# Save the LaTeX tables to a file
with open(goal, "w") as output_file:
    output_file.write(latex_tables)

print("LaTeX tables generated and saved to 'hyperparameters_tables.tex'")
