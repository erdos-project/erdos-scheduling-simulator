#!/bin/bash

# Run experiment
python main.py --flagfile=configs/simple_sample_experiment.yaml

# Analyze result
python analyze.py --csv_files=./simple_sampling.csv --conf_files=./configs/simple_sampling_workload.conf --task_stats=job