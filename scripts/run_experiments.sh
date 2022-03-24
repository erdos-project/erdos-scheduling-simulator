#!/bin/bash
# $1 directory where to save the logs.

# Scheduler runtimes in us.
schedulers=(edf lsf z3 gurobi)
scheduler_runtimes=(1 1000 5000 10000 -1)

max_timestamp=50
runtime_variance=10
deadline_variance=10

# Move to the simulator directory.
if [[ -z ${ERDOS_SIMULATOR_DIR} ]]; then
    echo "ERRROR: ERDOS_SIMULATOR_DIR is not set"
    exit 1
fi
cd ${ERDOS_SIMULATOR_DIR}

LOG_DIR=$1
if [[ -z ${LOG_DIR} ]]; then
    echo "WARNING: Log directory argument wasn't passed to the script. Setting log dir to `pwd`."
    LOG_DIR=`pwd`
fi

for runtime in ${scheduler_runtimes[@]}; do
    for scheduler in ${schedulers[@]}; do
        log_base=$1/scheduler_${scheduler}_runtime_${runtime}_timestamps_${max_timestamp}_runtime_var_${runtime_variance}_deadline_var_${deadline_variance}
        if [ ! -f "${log_base}.csv" ]; then
            python3 main.py --graph_path=data/pylot-complete-graph.dot --resource_path=data/pylot_resource_profile.json --worker_profile_path=data/worker_profile.json --max_timestamp=${max_timestamp} --deadline_variance=${deadline_variance} --runtime_variance=${runtime_variance} --scheduler_runtime=${runtime} --scheduler=${scheduler} --log=${log_base}.log --csv=${log_base}.csv
        else
            echo "${log_base}.csv already exists."
        fi
    done
done
