#!/bin/bash
# $1 directory where to save the logs.

# Scheduler runtimes in us.TetriSched
# SCHEDULERS=(EDF TetriSched_Gurobi)
# SCHEDULERS=(EDF TetriSched)
SCHEDULERS=(EDF TetriSched)
DEADLINE_VARIANCES=(50 100 200)
SCHEDULER_TIME_DISCRETIZATIONS=(1 5 10)
# SCHEDULERS=(EDF TetriSched)
# DEADLINE_VARIANCES=(50 100 200)
# SCHEDULER_TIME_DISCRETIZATIONS=(1 5 10)

SCHEDULER_LOG_TIMES=10
SCHEDULER_TIME_DISCRETIZATION=1
SCHEDULER_RUNTIME=0
LOG_LEVEL=debug
REPLAY_TRACE=alibaba
WORKLOAD_PROFILE_PATH=./traces/alibaba-cluster-trace-v2018/alibaba_random_50_dags.pkl
EXECUTION_MODE=replay
WORKER_CONFIG=alibaba_cluster
RELEASE_POLICY=fixed

PARALLEL_FACTOR=2
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
	MYCONF="\
    --log_file_name=${LOG_DIR}/${LOG_BASE}/${LOG_BASE}.log
    --csv_file_name=${LOG_DIR}/${LOG_BASE}/${LOG_BASE}.csv
    --log_level=${LOG_LEVEL}
    --execution_mode=${EXECUTION_MODE}
    --replay_trace=${REPLAY_TRACE}
    --max_deadline_variance=${DEADLINE_VAR}
    --min_deadline_variance=50
    --workload_profile_path=${WORKLOAD_PROFILE_PATH}
    --override_num_invocations=1
    --override_arrival_period=10
    --randomize_start_time_max=100
    --worker_profile_path=profiles/workers/${WORKER_CONFIG}.yaml
    --scheduler_runtime=${SCHEDULER_RUNTIME}
    --scheduler=${SCHEDULER}"
    if [[ ${SCHEDULER} != EDF ]]; then
    MYCONF+="
    --enforce_deadlines
    --retract_schedules
    --drop_skipped_tasks
    --release_taskgraphs
    --scheduler_log_times=${SCHEDULER_LOG_TIMES}
    --scheduler_time_discretization=${SCHEDULER_TIME_DISCRETIZATION}
    "
    fi 
    echo "${MYCONF}" > ${LOG_DIR}/${LOG_BASE}/${LOG_BASE}.conf
	if ! python3 main.py --flagfile=${LOG_DIR}/${LOG_BASE}/${LOG_BASE}.conf > ${LOG_DIR}/${LOG_BASE}/${LOG_BASE}.output; then
	    echo "[x] Failed in the execution of ${LOG_BASE}. Exiting."
	    exit 3
	fi
    else
	echo "[x] ${LOG_DIR}/${LOG_BASE}/${LOG_BASE}.csv already exists."
    fi
    echo "[x] Finished execution of ${LOG_BASE}."
}

for DEADLINE_VAR in ${DEADLINE_VARIANCES[@]}; do
    for SCHEDULER in ${SCHEDULERS[@]}; do
        for SCHEDULER_TIME_DISCRETIZATION in ${SCHEDULER_TIME_DISCRETIZATIONS[@]}; do
            if [[ ${SCHEDULER} == EDF && "${SCHEDULER_TIME_DISCRETIZATION}" -ne "${SCHEDULER_TIME_DISCRETIZATIONS[0]}"  ]]; then
                continue
            fi

            LOG_BASE=${REPLAY_TRACE}_scheduler_${SCHEDULER}_release_policy_${RELEASE_POLICY}_deadline_var_${DEADLINE_VAR}

            if [[ ${SCHEDULER} != EDF ]]; then
             LOG_BASE+="_scheduler_discretization_${SCHEDULER_TIME_DISCRETIZATION}"
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
wait
echo "[x] Finished executing all experiments."