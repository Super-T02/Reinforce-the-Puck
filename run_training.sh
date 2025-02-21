#!/bin/bash
# File: run_training.sh
# Author: Tom Freudenmann
# Content: Script to run training for all agents specified in the configuration file.

CONFIG_FILE="config/config.yaml"
NUM_CORES=$(nproc)
MAX_PROCESSES=$((NUM_CORES - 1))
START_TIME=$(date +%s)

# Print Help if first argument is -h or --help
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  echo "Usage: $0 [CONFIG_FILE] [MAX_PROCESSES]"
  echo "Run training for all agents specified in the configuration file."
  echo "If CONFIG_FILE is not provided, the default config file is used."
  echo "Additional arguments can be passed to specify the number of cores to use."
  exit 0
fi

# Get number of cores
if [[ -n "$2" ]]; then
  MAX_PROCESSES="$2"
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

# Determine the number of concurrent processes
if [[ $MAX_PROCESSES -lt 1 ]]; then
  MAX_PROCESSES=1
fi

if [[ $MAX_PROCESSES -gt $NUM_CORES ]]; then
  echo "Warning: Number of cores ($NUM_CORES) is less than the specified number of processes ($MAX_PROCESSES)"
  MAX_PROCESSES=$NUM_CORES
fi
echo "Max concurrent processes: $MAX_PROCESSES"

# Create a semaphore to limit the number of concurrent processes
SEMAPHORE=$(mktemp -u)
mkfifo "$SEMAPHORE"
exec 3<>"$SEMAPHORE"
rm "$SEMAPHORE"

for ((i = 0; i < MAX_PROCESSES; i++)); do
  echo >&3
done

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

  sleep 1

  # Run the training script with the agent-specific config
  (
    read -u 3
    python reinforce-the-puck/train.py -c "$AGENT_CONFIG_FILE"
    echo >&3
  ) &
done

wait
echo "All agents have finished training!"
END_TIME=$(date +%s)
echo "Total time taken: $((END_TIME - START_TIME)) seconds"
cleanup
