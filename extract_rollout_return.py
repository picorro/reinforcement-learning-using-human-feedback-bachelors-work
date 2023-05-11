import os
import sys
import csv
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def print_tags(event_file):
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()
    print("Tags in the event file:")
    for tag in event_acc.Tags()["scalars"]:
        print(tag)


def extract_rollout_return(event_file):
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()

    if "metrics/rollout_return" not in event_acc.Tags()["scalars"]:
        raise ValueError("metrics/rollout_return tag not found in the event file.")

    rollout_return_data = [(event.step, event.value) for event in event_acc.Scalars("metrics/rollout_return")]

    return rollout_return_data


def save_to_csv(rollout_return_data, output_csv_file):
    with open(output_csv_file, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["step", "rollout_return"])

        for step, value in rollout_return_data:
            csv_writer.writerow([step, value])


def main(input_tensorboard_file, output_csv_file):
    print_tags(input_tensorboard_file)
    rollout_return_data = extract_rollout_return(input_tensorboard_file)
    save_to_csv(rollout_return_data, output_csv_file)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_rollout_return.py input_tensorboard_file output_csv_file")
        sys.exit(1)
    input_tensorboard_file = sys.argv[1]
    output_csv_file = sys.argv[2]
    main(input_tensorboard_file, output_csv_file)
