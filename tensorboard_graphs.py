import os
import argparse
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--input_event", type=str, required=True, help="Input TensorBoard event file.")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory for generated graphs.")

args = parser.parse_args()
input_event = args.input_event
output_dir = args.output_dir

# Get the current timestamp and format it as a string
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# Create a unique subfolder inside the output directory
output_subdir = os.path.join(output_dir, timestamp)
os.makedirs(output_subdir, exist_ok=True)

# Load the event file
event_acc = EventAccumulator(input_event)
event_acc.Reload()

# Initialize a list to store the reward data
rewards = []

print(f"Available metrics: {event_acc.Tags()['scalars']}")
# Iterate through the events
for metric in event_acc.Tags()['scalars']:
    event_data = event_acc.Scalars(metric)
    
    # Check if the metric is the reward metric
    if metric == "metrics/rollout_return":
        for event in event_data:
            rewards.append((event.step, event.value))

# Sort the rewards by episode
rewards.sort(key=lambda x: x[0])

# Extract the episode and reward values
episodes, reward_values = zip(*rewards)

# Plot the rewards per episode
plt.figure()
plt.plot(episodes, reward_values)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Rewards per Episode")

# Save the graph as an image file
output_file = os.path.join(output_subdir, "rewards_per_episode.png")
plt.savefig(output_file)
