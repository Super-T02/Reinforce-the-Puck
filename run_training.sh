CONFIG_FILE="config/config.yaml"

# Print Help if first argument is -h or --help
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  echo "Usage: $0 [CONFIG_FILE]"
  echo "Run training for all agents specified in the configuration file."
  echo "If CONFIG_FILE is not provided, the default config file is used."
  exit 0
fi

# Get first argument as config file
if [[ -n "$1" ]]; then
  CONFIG_FILE="$1"
fi

# Temporary directory for split configs
TEMP_DIR=$(mktemp -d)

# Check if the config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Error: Configuration file '$CONFIG_FILE' not found!"
  exit 1
fi

# Function to clean up background processes and temporary directory
cleanup() {
  echo "Cleaning up..."
  pkill -P $$
  rm -rf "$TEMP_DIR"
  exit 0
}

# Trap termination signals to run the cleanup function
trap cleanup SIGINT SIGTERM

# Parse the config file and split it
echo "Splitting configuration file into agent-specific configs..."

# Extract agent keys using yq
AGENTS=$(yq 'keys[]' "$CONFIG_FILE" | grep '^"agent.*"$' | sed 's/"//g')
echo "Found agents: $AGENTS"

# Loop through each agent and create a specific config
for AGENT in $AGENTS; do
  # Create a new config file for the agent
  AGENT_CONFIG_FILE="$TEMP_DIR/${AGENT}_config.yaml"
  AGENT_PARAMS=$(yq ".${AGENT}" "$CONFIG_FILE")

  # Extract common parameters (not starting with agent) and agent-specific parameters
  yq -y 'with_entries(select(.key | test("^agent") | not))' "$CONFIG_FILE" > "$AGENT_CONFIG_FILE"
  echo "$AGENT:" >> "$AGENT_CONFIG_FILE"
  echo "$AGENT_PARAMS" | yq -y '.' | sed 's/^/  /' >> "$AGENT_CONFIG_FILE"
  echo "Created config for $AGENT: $AGENT_CONFIG_FILE"

  # Run the training script with the agent-specific config
  echo "Running training for agent: $AGENT"
  python reinforce-the-puck/train.py -c "$AGENT_CONFIG_FILE" &
done

echo "All agent-specific configs are stored in: $TEMP_DIR"
wait
echo "All agents have finished training!"
cleanup
