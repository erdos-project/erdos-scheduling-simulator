#!/bin/bash
# $1 directory where to save the logs.

# Scheduler runtimes in us.TetriSched
SCHEDULERS=(EDF TetriSched)
# MAX_DEADLINE_VARIANCES=(15 25 50 100 200)
# MAX_DEADLINE_VARIANCES=(200 400 800)
MAX_DEADLINE_VARIANCES=(300 400 500)
SCHEDULER_TIME_DISCRETIZATIONS=(1 10 20)
# RELEASE_POLICIES=(fixed poisson gamma)
RELEASE_POLICIES=(gamma)
# POISSON_ARRIVAL_RATES=(0.2 0.5 1 2)
POISSON_ARRIVAL_RATES=(0.015 0.02)
GAMMA_COEFFICIENTS=(1 2 4) #cv2
DAG_AWARENESS=(0 1) # False True
TASK_CPU_DIVISOR=20

ERDOS_SIMULATOR_DIR="." # Change this to the directory where the simulator is located.
MIN_DEADLINE_VARIANCE=10
NUM_INVOCATIONS=300
SCHEDULER_LOG_TIMES=10
SCHEDULER_RUNTIME=0
LOG_LEVEL=info
REPLAY_TRACE=alibaba
WORKLOAD_PROFILE_PATH=./traces/alibaba-cluster-trace-v2018/alibaba_set_0_6600_dags.pkl
EXECUTION_MODE=replay
WORKER_CONFIG=alibaba_cluster

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
# --log_dir=${LOG_DIR}/${LOG_BASE}
# --scheduler_log_to_file
    MYCONF="\
--log_file_name=${LOG_DIR}/${LOG_BASE}/${LOG_BASE}.log
--csv_file_name=${LOG_DIR}/${LOG_BASE}/${LOG_BASE}.csv
--log_level=${LOG_LEVEL}
--execution_mode=${EXECUTION_MODE}
--replay_trace=${REPLAY_TRACE}
--max_deadline_variance=${MAX_DEADLINE_VARIANCE}
--min_deadline_variance=${MIN_DEADLINE_VARIANCE}
--workload_profile_path=${WORKLOAD_PROFILE_PATH}
--override_num_invocations=${NUM_INVOCATIONS}
--randomize_start_time_max=100
--worker_profile_path=profiles/workers/${WORKER_CONFIG}.yaml
--scheduler_runtime=${SCHEDULER_RUNTIME}
--override_release_policy=${RELEASE_POLICY}
--scheduler=${SCHEDULER}
--alibaba_loader_task_cpu_divisor=${TASK_CPU_DIVISOR}
"
        if [[ ${RELEASE_POLICY} == fixed ]]; then
            MYCONF+="--override_arrival_period=10
"
        else
            MYCONF+="--override_poisson_arrival_rate=${POISSON_ARRIVAL_RATE}
"
            if [[ ${RELEASE_POLICY} == gamma ]]; then
                MYCONF+="--override_gamma_coefficient=${GAMMA_COEFFICIENT}
"
            fi
        fi

        if [[ ${DAG_AWARE} == 1 ]]; then
            MYCONF+="--release_taskgraphs
"
        fi

        if [[ ${SCHEDULER} != EDF ]]; then
            MYCONF+="
--enforce_deadlines
--retract_schedules
--drop_skipped_tasks
--scheduler_log_times=${SCHEDULER_LOG_TIMES}
--scheduler_time_discretization=${SCHEDULER_TIME_DISCRETIZATION}
    "
        fi 
        echo "${MYCONF}" > ${LOG_DIR}/${LOG_BASE}/${LOG_BASE}.conf
        if ! time python3 main.py --flagfile=${LOG_DIR}/${LOG_BASE}/${LOG_BASE}.conf > ${LOG_DIR}/${LOG_BASE}/${LOG_BASE}.output; then
            echo "[x] Failed in the execution of ${LOG_BASE}. Exiting."
            exit 3
        fi
    else
	    echo "[x] ${LOG_DIR}/${LOG_BASE}/${LOG_BASE}.csv already exists."
    fi
    echo "[x] Finished execution of ${LOG_BASE}."
}

for MAX_DEADLINE_VARIANCE in ${MAX_DEADLINE_VARIANCES[@]}; do
    for SCHEDULER in ${SCHEDULERS[@]}; do
        for RELEASE_POLICY in ${RELEASE_POLICIES[@]}; do
            for POISSON_ARRIVAL_RATE in ${POISSON_ARRIVAL_RATES[@]}; do
                for GAMMA_COEFFICIENT in ${GAMMA_COEFFICIENTS[@]}; do
                    for SCHEDULER_TIME_DISCRETIZATION in ${SCHEDULER_TIME_DISCRETIZATIONS[@]}; do
                        for DAG_AWARE in ${DAG_AWARENESS[@]}; do
                            if [[ ${SCHEDULER} == EDF && ( "${SCHEDULER_TIME_DISCRETIZATION}" -ne "${SCHEDULER_TIME_DISCRETIZATIONS[0]}" || "${DAG_AWARE}" -ne "${DAG_AWARENESS[0]}" )  ]]; then
                                continue
                            fi

                            LOG_BASE=${REPLAY_TRACE}_scheduler_${SCHEDULER}_release_policy_${RELEASE_POLICY}_max_deadline_var_${MAX_DEADLINE_VARIANCE}_dag_aware_${DAG_AWARE}_poisson_arrival_rate_${POISSON_ARRIVAL_RATE}_gamma_coefficient_${GAMMA_COEFFICIENT}

                            if [[ ${SCHEDULER} != EDF ]]; then
                                LOG_BASE+="_scheduler_discretization_${SCHEDULER_TIME_DISCRETIZATION}"
                            fi

                            mkdir -p ${LOG_DIR}/${LOG_BASE}
                            execute_experiment ${LOG_DIR} ${LOG_BASE} &
                            # sleep 0.5
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