# ERDOS Simulator

## Installation

To install the repository and set up the required paths, run

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
