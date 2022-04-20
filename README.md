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
cd plotting;
python3 plot_simulator_graphs.py \
  --csv_files={PATH_TO_CSV_LOG_FILE} \
  --csv_labels={SCHEDULER_NAME} \
  --plot_inter_task_time \
  --plot_task_placement_delay \
  --plot_task_placement_stats \
  --plot_task_slack \
  --plot_utilization
```
