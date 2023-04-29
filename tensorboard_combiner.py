import os
import sys
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import tensorflow as tf


def extract_scalar_summaries(event_file):
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()
    scalar_summaries = {}
    for tag in event_acc.Tags()["scalars"]:
        scalar_summaries[tag] = [
            (event.step, event.value) for event in event_acc.Scalars(tag)
        ]
    return scalar_summaries


def average_scalar_summaries(scalar_summaries_list):
    averaged_summaries = defaultdict(list)
    steps = None
    for tag in scalar_summaries_list[0].keys():
        for summaries in scalar_summaries_list:
            current_steps, values = zip(*summaries[tag])
            if steps is None:
                steps = current_steps
            averaged_summaries[tag].append(np.array(values))
        averaged_summaries[tag] = np.mean(averaged_summaries[tag], axis=0)

    return averaged_summaries, steps


def main(logdir_list, output_dir):
    summaries_list = [extract_scalar_summaries(logdir) for logdir in logdir_list]
    averaged_summaries, steps = average_scalar_summaries(summaries_list)

    os.makedirs(output_dir, exist_ok=True)
    summary_writer = tf.summary.create_file_writer(output_dir)

    with summary_writer.as_default():
        for tag, values in averaged_summaries.items():
            for step, value in zip(steps, values):
                tf.summary.scalar(tag, value, step=step)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python average_tensorboard.py output_dir logdir1 [logdir2 ...]")
        sys.exit(1)
    output_dir = sys.argv[1]
    logdirs = sys.argv[2:]
    main(logdirs, output_dir)
