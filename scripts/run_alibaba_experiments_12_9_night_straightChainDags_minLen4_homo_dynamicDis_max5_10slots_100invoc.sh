#!/bin/bash
# $1 directory where to save the logs.

# Scheduler runtimes in us.TetriSched
SCHEDULERS=(EDF TetriSched)
MAX_DEADLINE_VARIANCES=(25 200 50 100) # Keep deadline tight. Don't change this
SCHEDULER_TIME_DISCRETIZATIONS=(1)
GAMMA_COEFFICIENTS=(1 2 4) #cv2 don't change this
RELEASE_POLICIES=(fixed_gamma)
POISSON_ARRIVAL_RATES=(0.06 0.08) # Tune this
BASE_ARRIVAL_RATES=(0.03 0.04) # Tune this
DAG_AWARENESS=(1) # False True
TASK_CPU_DIVISOR=25

DYNAMIC_DISCRETIZATION=1
HETEROGENEOUS_RESOURCE=0
WORKER_CONFIG=alibaba_cluster

ERDOS_SIMULATOR_DIR="." # Change this to the directory where the simulator is located.
MIN_DEADLINE_VARIANCE=10
NUM_INVOCATIONS=100
SCHEDULER_LOG_TIMES=10
SCHEDULER_RUNTIME=0
LOG_LEVEL=debug
REPLAY_TRACE=alibaba
WORKLOAD_PROFILE_PATH=./traces/alibaba-cluster-trace-v2018/alibaba_straight_chain_7K_dags_minLen4_set1.pkl.pkl
EXECUTION_MODE=replay

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


    export TETRISCHED_LOGGING_DIR="${LOG_DIR}/${LOG_BASE}/"
    MYCONF="\
--log_dir=${LOG_DIR}/${LOG_BASE}
--log_file_name=${LOG_BASE}.log
--csv_file_name=${LOG_BASE}.csv
--scheduler_log_to_file
--log_graphs
--log_level=${LOG_LEVEL}
--execution_mode=${EXECUTION_MODE}
--replay_trace=${REPLAY_TRACE}
--max_deadline_variance=${MAX_DEADLINE_VARIANCE}
--min_deadline_variance=${MIN_DEADLINE_VARIANCE}
--workload_profile_path=${WORKLOAD_PROFILE_PATH}
--override_num_invocations=${NUM_INVOCATIONS}
--override_base_arrival_rate=${BASE_ARRIVAL_RATE}
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
            if [[ ${RELEASE_POLICY} == gamma || ${RELEASE_POLICY} == fixed_gamma ]]; then
                MYCONF+="--override_gamma_coefficient=${GAMMA_COEFFICIENT}
"
            fi
        fi

        if [[ ${DAG_AWARE} == 1 ]]; then
            MYCONF+="--release_taskgraphs
"
        fi

        if [[ ${HETEROGENEOUS_RESOURCE} == 1 ]]; then
            MYCONF+="--alibaba_enable_heterogeneous_resource_type
"
        fi

        if [[ ${OPTIMIZATION_PASS} == 1 ]]; then
            MYCONF+="--scheduler_enable_optimization_pass
"
        fi

        if [[ ${SCHEDULER} != EDF ]]; then
            MYCONF+="
--enforce_deadlines
--retract_schedules
--drop_skipped_tasks
--scheduler_log_times=${SCHEDULER_LOG_TIMES}
--scheduler_time_discretization=${SCHEDULER_TIME_DISCRETIZATION}
--scheduler_dynamic_discretization
--scheduler_max_time_discretization=5
--scheduler_max_occupancy_threshold=0.7
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


for POISSON_ARRIVAL_RATE in ${POISSON_ARRIVAL_RATES[@]}; do
    for SCHEDULER in ${SCHEDULERS[@]}; do
        for RELEASE_POLICY in ${RELEASE_POLICIES[@]}; do
            for GAMMA_COEFFICIENT in ${GAMMA_COEFFICIENTS[@]}; do
                for MAX_DEADLINE_VARIANCE in ${MAX_DEADLINE_VARIANCES[@]}; do
                    for SCHEDULER_TIME_DISCRETIZATION in ${SCHEDULER_TIME_DISCRETIZATIONS[@]}; do
                        for DAG_AWARE in ${DAG_AWARENESS[@]}; do
                            if [[ ${SCHEDULER} == EDF ]]; then
                                if [[ "${SCHEDULER_TIME_DISCRETIZATION}" -ne "${SCHEDULER_TIME_DISCRETIZATIONS[0]}" ]]; then
                                    continue
                                fi
                                DAG_AWARE=0
                            fi                            

                            # TODO: Make this more elegant.
                            if [[ ${POISSON_ARRIVAL_RATE} == ${POISSON_ARRIVAL_RATES[0]} ]]; then 
                                BASE_ARRIVAL_RATE=${BASE_ARRIVAL_RATES[0]}
                            fi
                            if [[ ${POISSON_ARRIVAL_RATE} == ${POISSON_ARRIVAL_RATES[1]} ]]; then 
                                BASE_ARRIVAL_RATE=${BASE_ARRIVAL_RATES[1]}
                            fi
                            if [[ ${POISSON_ARRIVAL_RATE} == ${POISSON_ARRIVAL_RATES[2]} ]]; then 
                                BASE_ARRIVAL_RATE=${BASE_ARRIVAL_RATES[2]}
                            fi

                            LOG_BASE=${REPLAY_TRACE}_scheduler_${SCHEDULER}_release_policy_${RELEASE_POLICY}_max_deadline_var_${MAX_DEADLINE_VARIANCE}_dag_aware_${DAG_AWARE}_poisson_arrival_rate_${POISSON_ARRIVAL_RATE}_gamma_coefficient_${GAMMA_COEFFICIENT}_base_arrival_rate_${BASE_ARRIVAL_RATE}
                            if [[ ${SCHEDULER} == TetriSched ]]; then
                                LOG_BASE+="_scheduler_discretization_${SCHEDULER_TIME_DISCRETIZATION}"
                                if [[ ${DYNAMIC_DISCRETIZATION} == 1 ]]; then
                                    LOG_BASE+="_dynamic_max_occupancy_threshold_0.7"
                                fi
                            fi

                            if [ -f "${LOG_DIR}/${LOG_BASE}/${LOG_BASE}.csv" ]; then
                                echo "[x] ${LOG_DIR}/${LOG_BASE}/${LOG_BASE}.csv already exists."
                                continue
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
