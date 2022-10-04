import csv
import numpy as np
from absl import app, flags
from tabulate import tabulate
import datetime

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "run_file", None, "The file where the runtimes of the different steps are stored."
)
flags.DEFINE_string(
    "cpu_file", None, "The file where the CPU and memory utilization traces are stored."
)
flags.DEFINE_string(
    "gpu_file", None, "The file where the GPU utilization traces are stored."
)
flags.register_multi_flags_validator(
    ["run_file", "cpu_file", "gpu_file"],
    lambda values: values["run_file"] is not None
    and values["cpu_file"] is not None
    and values["gpu_file"] is not None,
    message="The three files need to be provided for results.",
)


def main(args):
    decode_time = []
    preprocess_time = []
    prediction_time = []
    postprocess_time = []

    with open(FLAGS.run_file, "r") as run_file:
        for line in run_file:
            event_time = float(line.strip().rsplit(" ", 1)[-1][:-1])
            if "decoding" in line:
                decode_time.append(event_time)
            elif "pre-processing" in line:
                preprocess_time.append(event_time)
            elif "prediction" in line:
                prediction_time.append(event_time)
            elif "postprocessing" in line:
                postprocess_time.append(event_time)

    # Retrieve the CPU events and put them in the correct bin.
    decode_events = []
    preprocess_events = []
    prediction_events = []
    postprocess_events = []
    with open(FLAGS.cpu_file, "r") as cpu_file:
        for line in csv.reader(cpu_file, delimiter=" "):
            if "." in line[2] and "." in line[3]:
                event_time = (line[0] + " " + line[1])[:-3]
                event_time = datetime.datetime.strptime(
                    event_time, "%Y/%m/%d %H:%M:%S.%f"
                ).timestamp()
                if event_time >= decode_time[0] and event_time <= decode_time[1]:
                    decode_events.append((event_time, float(line[2]), float(line[3])))
                elif (
                    event_time >= preprocess_time[0]
                    and event_time <= preprocess_time[1]
                ):
                    preprocess_events.append(
                        (event_time, float(line[2]), float(line[3]))
                    )
                elif (
                    event_time >= prediction_time[0]
                    and event_time <= prediction_time[1]
                ):
                    prediction_events.append(
                        (event_time, float(line[2]), float(line[3]))
                    )
                elif (
                    event_time >= postprocess_time[0]
                    and event_time <= postprocess_time[1]
                ):
                    postprocess_events.append(
                        (event_time, float(line[2]), float(line[3]))
                    )

    # Retrieve the GPU events and put them in the correct bin.
    decode_gpu_events = []
    preprocess_gpu_events = []
    prediction_gpu_events = []
    postprocess_gpu_events = []

    with open(FLAGS.gpu_file, "r") as gpu_file:
        reader = csv.reader(gpu_file, delimiter=",")
        next(reader)
        for line in reader:
            event_time = datetime.datetime.strptime(
                line[0], "%Y/%m/%d %H:%M:%S.%f"
            ).timestamp()
            if not line[1].strip().endswith("MiB"):
                raise ValueError(f"Non-MiB value encountered in GPU trace: {line}")
            if event_time >= decode_time[0] and event_time <= decode_time[1]:
                decode_gpu_events.append(
                    (
                        event_time,
                        float(line[1].strip()[:-3]),
                        float(line[2].strip()[:-1]),
                    )
                )
            elif event_time >= preprocess_time[0] and event_time <= preprocess_time[1]:
                preprocess_gpu_events.append(
                    (
                        event_time,
                        float(line[1].strip()[:-3]),
                        float(line[2].strip()[:-1]),
                    )
                )
            elif event_time >= prediction_time[0] and event_time <= prediction_time[1]:
                prediction_gpu_events.append(
                    (
                        event_time,
                        float(line[1].strip()[:-3]),
                        float(line[2].strip()[:-1]),
                    )
                )
            elif (
                event_time >= postprocess_time[0] and event_time <= postprocess_time[1]
            ):
                postprocess_gpu_events.append(
                    (
                        event_time,
                        float(line[1].strip()[:-3]),
                        float(line[2].strip()[:-1]),
                    )
                )

    # Calculate the runtime of each steps.
    results = []
    results.append(
        [
            "Decode",
            str((decode_time[1] - decode_time[0]) * 1000) + " ms",
            np.average([event[1] for event in decode_events]),
            np.average([event[2] for event in decode_events]),
            np.average([event[1] for event in decode_gpu_events]),
            np.average([event[2] for event in decode_gpu_events]),
        ]
    )
    results.append(
        [
            "Preprocess",
            str((preprocess_time[1] - preprocess_time[0]) * 1000) + " ms",
            np.average([event[1] for event in preprocess_events]),
            np.average([event[2] for event in preprocess_events]),
            np.average([event[1] for event in preprocess_gpu_events]),
            np.average([event[2] for event in preprocess_gpu_events]),
        ]
    )
    results.append(
        [
            "Predict",
            str((prediction_time[1] - prediction_time[0]) * 1000) + " ms",
            np.average([event[1] for event in prediction_events]),
            np.average([event[2] for event in prediction_events]),
            np.average([event[1] for event in prediction_gpu_events]),
            np.average([event[2] for event in prediction_gpu_events]),
        ]
    )
    results.append(
        [
            "Postprocess",
            str((postprocess_time[1] - postprocess_time[0]) * 1000) + " ms",
            np.average([event[1] for event in postprocess_events]),
            np.average([event[2] for event in postprocess_events]),
            np.average([event[1] for event in postprocess_gpu_events]),
            np.average([event[2] for event in postprocess_gpu_events]),
        ]
    )

    print(
        tabulate(
            results,
            headers=["Step", "Runtime", "%CPU", "%MEM", "GPU_MEM", "%GPU"],
            tablefmt="grid",
            showindex=True,
        )
    )


if __name__ == "__main__":
    app.run(main)
