# ERDOS Simulator

## Installation

To install the repository and set up the required paths, install Python 3.7+,
and run

```console
python3 setup.py develop
```

## Executing Experiments

Run the following command in order to run experiments:

```console
python3 main.py --flagfile=configs/default.conf
```

Next, run the following command in order to plot graphs (e.g., resource
utilization, task placement delay, task slack) from the logs):

```console
python3 analyze.py \
  --csv_files={PATH_TO_CSV_LOG_FILE} \
  --csv_labels={SCHEDULER_NAME} \
  --inter_task_time \
  --task_placement \
  --task_placement_delay \
  --task_slack \
  --resource_utilization
  --plot
```

or plot all the graphs using:

```console
python3 analyze.py
  --csv_files={PATH_TO_CSV_LOG_FILE} \
  --csv_labels={SCHEDULER_NAME}
  --all
  --plot
```

To just output detailed statistics for all graphs, do

```console
python3 analyze.py
  --csv_files={PATH_TO_CSV_LOG_FILE} \
  --csv_labels={SCHEDULER_NAME}
  --all
```

and to convert the given CSV files into Chrome traces (to be visualized in chrome://tracing), do

```console
python3 analyze.py
  --csv_files={PATH_TO_CSV_LOG_FILE} \
  --csv_labels={SCHEDULER_NAME}
  --chrome_trace=task
```

The `scripts` directory provides helper scripts to spawn the execution of a large number of
experiments. To execute the experiments, change the exploration space in `scripts/run_experiments.sh`,
and then do

```console
export ERDOS_SIMULATOR_DIR=/path/to/cloned/repository
./scripts/run_experiments.sh /path/to/store/results
```

To check on the status of the experiments periodically, run

```console
watch -c -n 10 ./scripts/check_experiment_status.sh /results/path
```
where `/results/path` is the path specified while invoking `run_experiments.sh`


## Promtail Setup
- Download Promtail binary from https://github.com/grafana/loki/releases/tag/v2.8.6
- To start a run
  - First comment out these lines in your experiment script
    ```bash
    if ! time python3 main.py --flagfile=${LOG_DIR}/${LOG_BASE}/${LOG_BASE}.conf > ${LOG_DIR}/${LOG_BASE}/${LOG_BASE}.output; then
            echo "[x] Failed in the execution of ${LOG_BASE}. Exiting."
            exit 3
        fi
    ```
  - Run your experiment script to generate .conf files for each run
  - Then At the root of the repository, run `python scripts/promtail/generate_config.py <path_to_log_dir>`. This will generate a promtail-config.yaml file in `scripts/promtail/promtail_config.yaml`
  - Finally, start the promtail agent `./promtail-linux-amd64 -config.file=promtail-config.yaml`. It will watch for the log files and push them to Loki.
  - Uncomment the `python3 main.py` lines and start running your experiment : )