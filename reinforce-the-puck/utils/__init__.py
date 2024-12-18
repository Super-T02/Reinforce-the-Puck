import os

global workspace_dir, config_dir
workspace_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
config_dir: str = os.path.join(workspace_dir, "config")
model_dir: str = os.path.join(workspace_dir, "models")

logs_dir: str = os.path.join(workspace_dir, "logs")

# Create directories if they do not exist
for path in [workspace_dir, config_dir, model_dir, logs_dir]:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
