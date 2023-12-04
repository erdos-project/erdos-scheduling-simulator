#!/bin/bash
# $1 directory where to save the logs.

# Scheduler runtimes in us.TetriSched
# SCHEDULERS=(EDF TetriSched_Gurobi)
# SCHEDULERS=(EDF TetriSched)
SCHEDULERS=(EDF TetriSched TetriSched_DYN)
DEADLINE_VARIANCES=(50 100 200)
SCHEDULER_TIME_DISCRETIZATIONS=(1 5)
# SCHEDULERS=(EDF TetriSched)
# DEADLINE_VARIANCES=(50 100 200)
# SCHEDULER_TIME_DISCRETIZATIONS=(1 5 10)
NUM_INVOCATIONS=50
SCHEDULER_LOG_TIMES=10
SCHEDULER_TIME_DISCRETIZATION=1
SCHEDULER_MAX_DISCRETIZATION=5
SCHEDULER_RUNTIME=0
LOG_LEVEL=debug
REPLAY_TRACE=alibaba
WORKLOAD_PROFILE_PATH=./traces/alibaba-cluster-trace-v2018/alibaba_random_50_dags.pkl
EXECUTION_MODE=replay
WORKER_CONFIG=alibaba_cluster
RELEASE_POLICY=fixed
CSV_FILES=""
CONF_FILES=""
SCHEDULER_LABELS=""
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
    SCHEDULER_NAME=$SCHEDULER
    if [[ ${SCHEDULER} == TetriSched_DYN ]];
    SCHEDULER_NAME=TetriSched
    fi

    echo "[x] Initiating the execution of ${LOG_BASE}"
    if [ ! -f "${LOG_DIR}/${LOG_BASE}/${LOG_BASE}.csv" ]; then
	MYCONF="\
    --log_file_name=${LOG_DIR}/${LOG_BASE}/${LOG_BASE}.log
    --csv_file_name=${LOG_DIR}/${LOG_BASE}/${LOG_BASE}.csv
    --log_level=${LOG_LEVEL}
    --execution_mode=${EXECUTION_MODE}
    --replay_trace=${REPLAY_TRACE}
    --max_deadline_variance=${DEADLINE_VAR}
    --min_deadline_variance=${NUM_INVOCATIONS}
    --workload_profile_path=${WORKLOAD_PROFILE_PATH}
    --override_num_invocations=1
    --override_arrival_period=10
    --randomize_start_time_max=100
    --worker_profile_path=profiles/workers/${WORKER_CONFIG}.yaml
    --scheduler_runtime=${SCHEDULER_RUNTIME}
    --scheduler=${SCHEDULER_NAME}"
    if [[ ${SCHEDULER} != EDF ]]; then
        MYCONF+="
        --enforce_deadlines
        --retract_schedules
        --drop_skipped_tasks
        --release_taskgraphs
        --scheduler_log_times=${SCHEDULER_LOG_TIMES}
        "
        if [[ ${SCHEDULER} != TetriSched_DYN ]]; then
            MYCONF+="
            --scheduler_time_discretization=${SCHEDULER_TIME_DISCRETIZATION}
            "
        fi
        else
            MYCONF+="
            --scheduler_time_discretization=1
            --scheduler_adaptive_discretization
            --scheduler_max_time_discretization=${SCHEDULER_MAX_DISCRETIZATION}
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
            if [[ ${SCHEDULER} == TetriSched_DYN && "${SCHEDULER_TIME_DISCRETIZATION}" -ne "${SCHEDULER_TIME_DISCRETIZATIONS[0]}"  ]]; then
                continue
            fi

            LOG_BASE=${REPLAY_TRACE}_scheduler_${SCHEDULER}_release_policy_${RELEASE_POLICY}_deadline_var_${DEADLINE_VAR}
            LABEL=${SCHEDULER}_deadline_${DEADLINE_VAR}

            if [[ ${SCHEDULER} != EDF ] && [ ${SCHEDULER} != TetriSched_DYN ]]; then
             LOG_BASE+="_scheduler_discretization_${SCHEDULER_TIME_DISCRETIZATION}"
             LABEL+="_discr_${SCHEDULER_TIME_DISCRETIZATION}"
            fi
            if [[ ${CSV_FILES} == "" ]]; then
            CSV_FILES+="${LOG_DIR}/${LOG_BASE}/${LOG_BASE}.csv"
            CONF_FILES+="${LOG_DIR}/${LOG_BASE}/${LOG_BASE}.conf"
            SCHEDULER_LABELS+="${LABEL}"
            else
            CSV_FILES+=",${LOG_DIR}/${LOG_BASE}/${LOG_BASE}.csv"
            CONF_FILES+=",${LOG_DIR}/${LOG_BASE}/${LOG_BASE}.conf"
            SCHEDULER_LABELS+=",${LABEL}"
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
echo "[x] Analzying experiments. ${CSV_FILES} , ${SCHEDULER_LABELS}"

CSV_FILES=""
SCHEDULER_LABELS=""
CONF_FILES=""
for DEADLINE_VAR in ${DEADLINE_VARIANCES[@]}; do
    for SCHEDULER in ${SCHEDULERS[@]}; do
        for SCHEDULER_TIME_DISCRETIZATION in ${SCHEDULER_TIME_DISCRETIZATIONS[@]}; do
        if [[ ${SCHEDULER} == EDF && "${SCHEDULER_TIME_DISCRETIZATION}" -ne "${SCHEDULER_TIME_DISCRETIZATIONS[0]}"  ]]; then
                continue
        fi
        if [[ ${SCHEDULER} == TetriSched_DYN && "${SCHEDULER_TIME_DISCRETIZATION}" -ne "${SCHEDULER_TIME_DISCRETIZATIONS[0]}"  ]]; then
                continue
        fi

        LABEL=${SCHEDULER}
        LOG_BASE=${REPLAY_TRACE}_scheduler_${SCHEDULER}_release_policy_${RELEASE_POLICY}_deadline_var_${DEADLINE_VAR}

        if [[ ${SCHEDULER} != EDF ]]; then
            LOG_BASE+="_scheduler_discretization_${SCHEDULER_TIME_DISCRETIZATION}"
            LABEL+="_d_${SCHEDULER_TIME_DISCRETIZATION}"
        fi
        if [[ ${CSV_FILES} == "" ]]; then
            CSV_FILES+="${LOG_DIR}/${LOG_BASE}/${LOG_BASE}.csv"
            CONF_FILES+="${LOG_DIR}/${LOG_BASE}/${LOG_BASE}.conf"
            SCHEDULER_LABELS+="${LABEL}"
            else
            CSV_FILES+=",${LOG_DIR}/${LOG_BASE}/${LOG_BASE}.csv"
            CONF_FILES+=",${LOG_DIR}/${LOG_BASE}/${LOG_BASE}.conf"
            SCHEDULER_LABELS+=",${LABEL}"
            fi
        done
    done
    if ! python3 analyze.py \
	--csv_files=${CSV_FILES} \
	--csv_labels=${SCHEDULER_LABELS} \
	--plot_goodput_graph \
    --goodput_plot_name="deadline_${DEADLINE_VAR}.png" \
    --goodput_plot_title="Deadline ${DEADLINE_VAR}" \
	--output_dir=${LOG_DIR}; then
	echo "[x] Failed in analyzing. Exiting."
	exit 4
    fi
    CSV_FILES=""
    SCHEDULER_LABELS=""
    CONF_FILES=""
done


 