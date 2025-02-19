import os

import yaml


def parse_yaml_to_latex_tables(yaml_file):
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

        # Start building the LaTeX table for the current environment
        latex_table = f"""
\\begin{{table}}[htbp]
    \\centering
    \\caption{{Hyperparameter Summary for {env_name}}}
    \\label{{tab:hyperparameters_{env_name.lower()}}}
    \\begin{{tabular}}{{|l|l|l|l|l|l|}}
        \\hline
        \\textbf{{Agent}} & \\textbf{{Noise}} & \\textbf{{Actor Sizes}} & \\textbf{{Critic Sizes}} & \\textbf{{Buffer Type}} & \\textbf{{Discount}} \\\\ \\hline
        """

        # Add rows for each agent in the environment
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

            # Add a row to the LaTeX table
            latex_table += f"""
        {algorithm} & {noise} & {actor_sizes} & {critic_sizes} & {buffer_type} & {discount} \\\\ \\hline
            """

        # Close the LaTeX table for the current environment
        latex_table += """
    \\end{tabular}
\\end{table}
        """
        latex_tables += latex_table + "\n\n"

    return latex_tables


# Use the function to generate the LaTeX tables
yaml_file = "../../config/temp.yaml"  # Replace with the path to your YAML file
latex_tables = parse_yaml_to_latex_tables(yaml_file)
goal = "../tables/hyperparameters_tables.tex"

os.makedirs(os.path.dirname(goal), exist_ok=True)

# Save the LaTeX tables to a file
with open(goal, "w") as output_file:
    output_file.write(latex_tables)

print("LaTeX tables generated and saved to 'hyperparameters_tables.tex'")
