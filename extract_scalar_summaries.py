# import csv
# import os
# import sys
# from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# def extract_scalar_summaries(event_file):
#     event_acc = EventAccumulator(event_file)
#     event_acc.Reload()
#     scalar_summaries = {}
#     for tag in event_acc.Tags()['scalars']:
#         scalar_summaries[tag] = [(event.step, event.value) for event in event_acc.Scalars(tag)]
#     return scalar_summaries

# def save_scalar_summaries_to_csv(scalar_summaries, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#     for tag, summaries in scalar_summaries.items():
#         tag = tag.replace('/', '_')
#         with open(os.path.join(output_dir, f"{tag}.csv"), 'w', newline='') as csvfile:
#             csvwriter = csv.writer(csvfile)
#             csvwriter.writerow(['Step', 'Value'])
#             for step, value in summaries:
#                 csvwriter.writerow([step, value])

# def find_event_files(logdir):
#     event_files = []
#     for root, _, files in os.walk(logdir):
#         for file in files:
#             if file.startswith("events.out"):
#                 event_files.append(os.path.join(root, file))
#     return event_files

# if __name__ == "__main__":
#     if len(sys.argv) != 3:
#         print("Usage: python extract_scalar_summaries.py logdir output_dir")
#         sys.exit(1)
#     logdir = sys.argv[1]
#     output_dir = sys.argv[2]
#     event_files = find_event_files(logdir)
#     for event_file in event_files:
#         scalar_summaries = extract_scalar_summaries(event_file)
#         save_scalar_summaries_to_csv(scalar_summaries, output_dir)
import csv
import os
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorflow.core.framework import tensor_pb2
from tensorflow.python.framework import tensor_util


def extract_scalar_summaries(event_file):
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()
    scalar_summaries = {}
    for tag in event_acc.Tags()["scalars"]:
        scalar_summaries[tag] = [
            (event.step, event.value) for event in event_acc.Scalars(tag)
        ]
    return scalar_summaries


def save_scalar_summaries_to_csv(scalar_summaries, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for tag, summaries in scalar_summaries.items():
        tag = tag.replace("/", "_")
        with open(os.path.join(output_dir, f"{tag}.csv"), "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["Step", "Value"])
            for step, value in summaries:
                csvwriter.writerow([step, value])


def find_event_files(logdir):
    event_files = []
    for root, _, files in os.walk(logdir):
        for file in files:
            if file.startswith("events.out"):
                event_files.append(os.path.join(root, file))
    return event_files


def extract_tensor_summaries(event_file):
    size_guidance = {
        "tensors": 10000,  # Load up to 10000 tensors
    }
    event_acc = EventAccumulator(event_file, size_guidance=size_guidance)
    event_acc.Reload()
    tensor_summaries = {}
    for tag in event_acc.Tags()["tensors"]:
        tensor_summaries[tag] = []
        for event in event_acc.Tensors(tag):
            tensor = tensor_util.MakeNdarray(event.tensor_proto)
            tensor_summaries[tag].append((event.step, tensor.item()))
    return tensor_summaries


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_scalar_summaries.py logdir output_dir")
        sys.exit(1)
    logdir = sys.argv[1]
    output_dir = sys.argv[2]
    event_files = find_event_files(logdir)
    print(f"Found {len(event_files)} event files in {logdir}:")
    for event_file in event_files:
        print(event_file)
        event_acc = EventAccumulator(event_file)
        event_acc.Reload()
        print(f"All available tags in {event_file}:")
        print(event_acc.Tags())
        tensor_summaries = extract_tensor_summaries(event_file)
        print(f"Extracted {len(tensor_summaries)} tensor summaries from {event_file}:")
        for tag in tensor_summaries:
            print(tag)
        save_scalar_summaries_to_csv(tensor_summaries, output_dir)
