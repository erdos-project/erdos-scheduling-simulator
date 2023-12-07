import csv
import json

from absl import app, flags

flags.DEFINE_string("csv_file", None, "CSV file to analyze")
flags.mark_flag_as_required("csv_file")
flags.DEFINE_string("output", None, "Output file to write to")
flags.mark_flag_as_required("output")

FLAGS = flags.FLAGS


def main(argv):
    print("[x] Analyzing performance from: {}".format(FLAGS.csv_file))

    trace = {
        "traceEvents": [],
        "displayTimeUnit": "ms",
        "otherData": {
            "csv_path": FLAGS.csv_file,
        },
    }

    processed_event_count = 0
    with open(FLAGS.csv_file, "r") as csv_file:
        event_stack = []
        for line in csv.reader(csv_file):
            try:
                if line[0] == "BEGIN":
                    event_stack.append(line)
                elif line[0] == "END":
                    begin_event = event_stack.pop()
                    if begin_event[1] != line[1] or begin_event[-1] != line[-3]:
                        raise RuntimeError(
                            f"Mismatched events: {begin_event} and {line}"
                        )

                    trace_event = {
                        "name": begin_event[1],
                        "ph": "X",
                        "ts": begin_event[-1],
                        "dur": int(line[-2]) - int(begin_event[-1]),
                        "pid": "libtetrisched",
                        "args": {
                            "function": begin_event[1],
                            "simulator_time": begin_event[2],
                        },
                    }
                    trace["traceEvents"].append(trace_event)
                    processed_event_count += 1
                else:
                    raise RuntimeError(f"Invalid first column: {line[0]}")
            except Exception as e:
                print(f"Error processing line: {line}")
                print(f"The last event in stack is: {event_stack[-1]}")
                raise e
    print(
        "[x] Processed {} events from: {}".format(processed_event_count, FLAGS.csv_file)
    )

    print("[x] Output Chrome Trace to: {}".format(FLAGS.output))
    with open(FLAGS.output, "w") as output_file:
        json.dump(trace, output_file, default=lambda obj: obj.__dict__, indent=4)


if __name__ == "__main__":
    app.run(main)
