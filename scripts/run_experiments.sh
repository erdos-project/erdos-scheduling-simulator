#!/bin/bash
# $1 directory where to save the logs.

# Scheduler runtimes in us.
SCHEDULERS=(EDF LSF Z3 Gurobi)
SCHEDULER_RUNTIMES=(1 1000 5000 10000 -1)
SCHEDULING_HORIZONS=(0 1000 5000 10000)
RUNTIME_VARIANCES=(0 10)
DEADLINE_VARIANCES=(0 10)
MAX_TIMESTAMP=50
RESOURCE_CONFIGS=(pylot_1_camera_1_lidar_resource_profile)
WORKER_CONFIGS=(worker_2_machines_24_cpus_8_gpus_profile worker_2_machines_24_cpus_9_gpus_profile worker_2_machines_24_cpus_10_gpus_profile)
EXECUTION_MODE=synthetic


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

for WORKER_CONFIG in ${WORKER_CONFIGS[@]}; do
    for RESOURCE_CONFIG in ${RESOURCE_CONFIGS[@]}; do
        for DEADLINE_VAR in ${DEADLINE_VARIANCES[@]}; do
            for RUNTIME_VAR in ${RUNTIME_VARIANCES[@]}; do
                for RUNTIME in ${SCHEDULER_RUNTIMES[@]}; do
                    for SCHEDULER in ${SCHEDULERS[@]}; do
                        for SCHEDULING_HORIZON in ${SCHEDULING_HORIZONS[@]}; do
                            # Other schedulers do not support scheduling horizons.
                            if [[ ${SCHEDULING_HORIZON} -ne 0 && ${SCHEDULER} != gurobi && ${SCHEDULER} != z3 ]] ; then
                                continue
                            fi
                            LOG_BASE=$1/${EXECUTION_MODE}_scheduler_${SCHEDULER}_horizon_${SCHEDULING_HORIZON}_runtime_${RUNTIME}_timestamps_${MAX_TIMESTAMP}_runtime_var_${RUNTIME_VAR}_deadline_var_${DEADLINE_VAR}_${WORKER_CONFIG}_${RESOURCE_CONFIG}
                            echo "Running ${LOG_BASE}"
                            if [ ! -f "${LOG_BASE}.csv" ]; then
                                python3 main.py --graph_path=profiles/workload/pylot-complete-graph.dot \
                                        --resource_path=profiles/workload/${RESOURCE_CONFIG}.json \
                                        --worker_profile_path=profiles/workers/${WORKER_CONFIG}.json \
                                        --max_timestamp=${MAX_TIMESTAMP} \
                                        --max_deadline_variance=${DEADLINE_VAR} \
                                        --runtime_variance=${RUNTIME_VAR} \
                                        --scheduler_runtime=${RUNTIME} \
                                        --scheduler=${SCHEDULER} \
                                        --scheduling_horizon=${SCHEDULING_HORIZON} \
                                        --log=${LOG_BASE}.log \
                                        --csv=${LOG_BASE}.csv \
                                        --preemption=False \
                                        --synchronize_sensors \
                                        --timestamp_difference=100000 \
                                        --execution_mode=${EXECUTION_MODE}
                            else
                                echo "${LOG_BASE}.csv already exists."
                            fi
                            python3 analyze.py --csv_files=${LOG_BASE}.csv --csv_labels=${SCHEDULER} --all
                            mkdir -p ${LOG_BASE}
                            mv *.png ${LOG_BASE}/
                        done
                    done
                done
            done
        done
    done
done
