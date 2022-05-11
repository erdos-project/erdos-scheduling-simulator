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
PARALLEL_FACTOR=16


# Move to the simulator directory.
if [[ -z ${ERDOS_SIMULATOR_DIR} ]]; then
    echo "[x] ERRROR: ERDOS_SIMULATOR_DIR is not set"
    exit 1
fi
cd ${ERDOS_SIMULATOR_DIR}

LOG_DIR=$1
if [[ -z ${LOG_DIR} ]]; then
    echo "[x] ERROR: Please provide a directory to output results to as the first argument."
    exit 2
fi

execute_experiment () {
    LOG_DIR=$1
    LOG_BASE=$2
    echo "[x] Initiating the execution of ${LOG_BASE}"
    if [ ! -f "${LOG_DIR}/${LOG_BASE}/${LOG_BASE}.csv" ]; then
	echo "\
--graph_path=profiles/workload/pylot-complete-graph.dot
--resource_path=profiles/workload/${RESOURCE_CONFIG}.json
--worker_profile_path=profiles/workers/${WORKER_CONFIG}.json
--max_timestamp=${MAX_TIMESTAMP}
--max_deadline_variance=${DEADLINE_VAR}
--runtime_variance=${RUNTIME_VAR}
--scheduler_runtime=${RUNTIME}
--scheduler=${SCHEDULER}
--scheduling_horizon=${SCHEDULING_HORIZON}
--log=${LOG_DIR}/${LOG_BASE}/${LOG_BASE}.log
--csv=${LOG_DIR}/${LOG_BASE}/${LOG_BASE}.csv
--preemption=False
--synchronize_sensors
--timestamp_difference=100000
--execution_mode=${EXECUTION_MODE}" > ${LOG_DIR}/${LOG_BASE}/${LOG_BASE}.conf
	if ! python3 main.py --flagfile=${LOG_DIR}/${LOG_BASE}/${LOG_BASE}.conf; then
	    echo "[x] Failed in the execution of ${LOG_BASE}. Exiting."
	    exit 3
	fi
    else
	echo "[x] ${LOG_DIR}/${LOG_BASE}/${LOG_BASE}.csv already exists."
    fi
    if ! python3 analyze.py \
	--csv_files=${LOG_DIR}/${LOG_BASE}/${LOG_BASE}.csv \
	--csv_labels=${SCHEDULER} \
	--all \
	--plot \
	--output_dir=${LOG_DIR}/${LOG_BASE} > /dev/null; then
	echo "[x] Failed in analyzing the results of ${LOG_BASE}. Exiting."
	exit 4
    fi
   echo "[x] Finished execution of ${LOG_BASE}."
}

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

                            LOG_BASE=${EXECUTION_MODE}_scheduler_${SCHEDULER}_horizon_${SCHEDULING_HORIZON}_runtime_${RUNTIME}_timestamps_${MAX_TIMESTAMP}_runtime_var_${RUNTIME_VAR}_deadline_var_${DEADLINE_VAR}_${WORKER_CONFIG}
                            # Synthetic execution mode does not support resource configs.
                            if [[ ${EXECUTION_MODE} != synthetic ]]; then
                                LOG_BASE="${LOG_BASE}_${RESOURCE_CONFIG}"
                            fi

                            mkdir -p ${LOG_DIR}/${LOG_BASE}
                            execute_experiment ${LOG_DIR} ${LOG_BASE} &

			    if [[ $(jobs -r -p | wc -l) -ge $PARALLEL_FACTOR ]]; then
				echo "[x] Waiting for a job to terminate because $PARALLEL_FACTOR jobs are running."
		                wait -n 
			    fi
                        done
                    done
                done
            done
        done
    done
done

wait
echo "[x] Finished executing all experiments."
