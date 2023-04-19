import os
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
    ScalarEvent,
)
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.compat.proto import event_pb2
from tensorboard.compat.proto.summary_pb2 import Summary


def combine_tensorboard_logs(root_directory):
    pathExists = os.path.exists(f"./{root_directory}/combined")
    if not pathExists:
        os.makedirs(f"{root_directory}/combined")

    output_file_path = f"{root_directory}/combined"

    all_events = []
    step_offset = 0

    # Iterate through all immediate subdirectories in the root directory
    for log_directory in Path(root_directory).iterdir():
        if log_directory.is_dir():
            max_step = 0

            # Iterate through all TensorBoard log files in the current directory, without going deeper
            for log_file in log_directory.glob("events.out.tfevents.*"):
                # Load the event data from the log file
                event_acc = EventAccumulator(str(log_file))
                event_acc.Reload()

                # Extract the scalar events for each available tag
                for tag in event_acc.Tags()["scalars"]:
                    scalar_events = event_acc.Scalars(tag)

                    # Add the scalar events to the list of all events with adjusted steps
                    all_events.extend(
                        [
                            (
                                tag,
                                ScalarEvent(
                                    wall_time=scalar_event.wall_time,
                                    step=scalar_event.step + step_offset,
                                    value=scalar_event.value,
                                ),
                            )
                            for scalar_event in scalar_events
                        ]
                    )

                    # Update max_step with the maximum step from the current log file
                    max_step = max(
                        max_step, scalar_events[-1].step if scalar_events else 0
                    )

            # Update the step_offset for the next training session
            step_offset += max_step

    # Sort the events by their wall time
    all_events.sort(key=lambda event: event[1].wall_time)

    # Create an EventFileWriter instance and write the sorted events to the output file
    event_writer = EventFileWriter(output_file_path)
    for tag, scalar_event in all_events:
        # Create an event_pb2.Event proto for the scalar event
        event_proto = event_pb2.Event(
            wall_time=scalar_event.wall_time,
            step=scalar_event.step,
            summary=Summary(
                value=[Summary.Value(tag=tag, simple_value=scalar_event.value)]
            ),
        )
        event_writer.add_event(event_proto)

    # Close the EventFileWriter
    event_writer.close()

    print(f"Combined log file written to: {output_file_path}")
