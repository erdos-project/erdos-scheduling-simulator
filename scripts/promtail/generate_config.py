import copy
from datetime import datetime
import os
import re
import sys
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "base_folder",
    help="Base folder of the experiment run. It should contain many subfolders, each of which contains the log/csv files of each run.",
)
args = parser.parse_args()

base_folder = args.base_folder
today_date = datetime.now().strftime("%m/%d/%Y")

promtail_config = {
    "server": {"http_listen_port": 0, "grpc_listen_port": 0},
    "positions": {"filename": "/tmp/positions.yaml"},
    "clients": [
        {
            "url": "https://761696:glc_eyJvIjoiMTAwNzgwNiIsIm4iOiJzdGFjay04MDg2MDEtaGwtd3JpdGUtc3lzbWwtMDIiLCJrIjoiOTRIMjhXNTlXbnRNV3Y1dlMxNHc0aXNZIiwibSI6eyJyIjoicHJvZC11cy1lYXN0LTAifX0=@logs-prod-006.grafana.net/loki/api/v1/push",
            "batchsize": 848576,
        }
    ],
    "scrape_configs": [
        {
            "job_name": "dag_sched",
            "static_configs": [],
        }
    ],
}

simulator_config_value_regex = r"--([^=]+)=(.+)"
for folder in os.listdir(base_folder):
    # find .conf and extract all labels
    for file_name in os.listdir(os.path.join(base_folder, folder)):
        if file_name.endswith(".conf"):
            labels = {}
            with open(os.path.join(base_folder, folder, file_name)) as f:
                lines = f.readlines()
                for line in lines:
                    # Performing the regex match
                    match = re.match(simulator_config_value_regex, line)
                    if match:
                        key = match.group(1)
                        value = match.group(2)
                        labels[key] = value
            break

    labels["date"] = today_date

    if "log_level" in labels:
        del labels["log_level"]
    if "execution_mode" in labels:
        del labels["execution_mode"]
    if "replay_trace" in labels:
        del labels["replay_trace"]
    if "log_dir" in labels:
        del labels["log_dir"]
    if "csv_file_name" in labels:
        del labels["csv_file_name"]
    if "log_file_name" in labels:
        del labels["log_file_name"]
    if "log_dir" in labels:
        del labels["log_dir"]
    if "randomize_start_time_max" in labels:
        del labels["randomize_start_time_max"]
    if "scheduler_log_times" in labels:
        del labels["scheduler_log_times"]
    if "scheduler_runtime" in labels:
        del labels["scheduler_runtime"]
    if "machine" in labels:
        del labels["machine"]
    if "alibaba_loader_task_cpu_divisor" in labels:
        del labels["alibaba_loader_task_cpu_divisor"]
    if "min_deadline_variance" in labels:
        del labels["min_deadline_variance"]

    for extension in ["csv", "log", "conf", "output"]:
        labels[
            "__path__"
        ] = f"{os.path.abspath(os.path.join(base_folder, folder))}/*.{extension}"
        labels["file_type"] = extension
        promtail_config["scrape_configs"][0]["static_configs"].append(
            {"targets": ["localhost"], "labels": copy.copy(labels)}
        )

    # Hanlde tetrisched_.*.log
    labels[
        "__path__"
    ] = f"{os.path.abspath(os.path.join(base_folder, folder))}/tetrisched_.*.log"
    labels["file_type"] = "tetrisched_log"
    promtail_config["scrape_configs"][0]["static_configs"].append(
        {"targets": ["localhost"], "labels": copy.copy(labels)}
    )

# Generate the YAML file
yaml_content = yaml.dump(promtail_config, default_flow_style=False)
yaml_content = yaml_content.replace(
    """source_labels:
    - __path__""",
    "source_labels: ['__path__']",
)

# Save to a file
file_path = "scripts/promtail/promtail_config.yaml"
with open(file_path, "w") as file:
    file.write(yaml_content)
