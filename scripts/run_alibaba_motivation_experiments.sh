#!/bin/bash
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


# Random seeds.
# We use different seeds so that we can run with a different set of TaskGraphs
# being chosen from the trace, along with different arrival patterns.
RANDOM_SEEDS=(420665456 9165261 106432512 95498947 105937176 66362485 416681780)

# Schedulers
# We use the following baseline schedulers to compare the performance of DAGSched with.
SCHEDULERS=(EDF TetriSched GraphenePrime DAGSched)

# Poisson arrival rates.
# We use the following arrival rates for the Poisson arrival process.
MEDIUM_ARRIVAL_RATES=(0.0075 0.005 0.0046 0.004 0.0036)
HARD_ARRIVAL_RATES=(0.0025 0.0025 0.002 0.002 0.0015)

execute_experiment () {
  SCHEDULER=$1
  RANDOM_SEED=$2
  LOG_DIR=$3
  LOG_BASE="run_${RANDOM_SEED}"
  MEDIUM_ARRIVAL_RATE=$4
  HARD_ARRIVAL_RATE=$5

  EXPERIMENT_DIR="${LOG_DIR}/${SCHEDULER}/${LOG_BASE}/arrival_rate_${MEDIUM_ARRIVAL_RATE}_${HARD_ARRIVAL_RATE}"
  mkdir -p ${EXPERIMENT_DIR}

  if [ -f "${EXPERIMENT_DIR}/alibaba_trace_replay.csv" ]; then
    echo "[x] The experiment for ${SCHEDULER} with random seed ${RANDOM_SEED} has already been run."
    return
  fi

  # Build the baseline configuration for the experiment.
  EXPERIMENT_CONF="\
  # Output configuration.
  --log_dir=${EXPERIMENT_DIR}
  --log_file_name=alibaba_trace_replay.log
  --csv_file_name=alibaba_trace_replay.csv
  --log_level=debug
  "

  EXPERIMENT_CONF+="
  # Worker configuration.
  --worker_profile_path=profiles/workers/alibaba_cluster_30_slots.yaml
  "

  EXPERIMENT_CONF+="
  # Workload configuration.
  --execution_mode=replay
  --replay_trace=alibaba
  --workload_profile_paths=traces/alibaba-cluster-trace-v2018/easy_dag_sukrit_10k.pkl,traces/alibaba-cluster-trace-v2018/medium_dag_sukrit_10k.pkl,traces/alibaba-cluster-trace-v2018/hard_dag_sukrit_10k.pkl
  --workload_profile_path_labels=easy,medium,hard
  --override_release_policies=poisson,poisson,poisson
  --override_num_invocations=0,200,100
  --override_poisson_arrival_rates=0.0075,${MEDIUM_ARRIVAL_RATE},${HARD_ARRIVAL_RATE}
  --randomize_start_time_max=50
  --min_deadline=5
  --max_deadline=500
  --min_deadline_variances=25,50,10
  --max_deadline_variances=50,100,25
  
  # Loader configuration.
  --alibaba_loader_task_cpu_divisor=10
  --alibaba_loader_min_critical_path_runtimes=200,500,600
  --alibaba_loader_max_critical_path_runtimes=500,1000,1000
  "

  if [[ ${SCHEDULER} == "EDF" ]]; then
    EXPERIMENT_CONF+="
    # Scheduler configuration.
    --scheduler=EDF
    --enforce_deadlines
    "
  elif [[ ${SCHEDULER} == "TetriSched" ]]; then
    EXPERIMENT_CONF+="
    # Scheduler configuration.
    --scheduler=TetriSched
    --enforce_deadlines
    --scheduler_time_discretization=1
    --scheduler_enable_optimization_pass
    --retract_schedules
    --scheduler_time_limit=120
    "
  elif [[ ${SCHEDULER} == "GraphenePrime" ]]; then
    EXPERIMENT_CONF+="
    # Scheduler configuration.
    --scheduler=GraphenePrime
    --scheduler_selective_rescheduling
    --scheduler_selective_rescheduling_sample_size=2
    --scheduler_time_discretization=2
    --scheduler_enable_optimization_pass
    --scheduler_plan_ahead=1000
    --retract_schedules
    --scheduler_time_limit=120
    "
  elif [[ ${SCHEDULER} == "DAGSched" ]]; then
    EXPERIMENT_CONF+="
    # Scheduler configuration.
    --scheduler=TetriSched
    --release_taskgraphs
    --enforce_deadlines
    --scheduler_time_discretization=1
    --scheduler_enable_optimization_pass
    --scheduler_reconsideration_period=0.2
    --retract_schedules
    --scheduler_time_limit=120
    "
  else
    echo "[x] ERROR: Unknown scheduler ${SCHEDULER}"
    exit 1
  fi

  EXPERIMENT_CONF+="\
  --scheduler_runtime=0
  --random_seed=${RANDOM_SEED}"

  echo "${EXPERIMENT_CONF}" | sed -e 's/^[ \t]*//' > ${EXPERIMENT_DIR}/alibaba_trace_replay.conf

  echo "[x] Constructed configuration for ${EXPERIMENT_DIR}. Beginning experiment"

  if ! python3 main.py --flagfile=${EXPERIMENT_DIR}/alibaba_trace_replay.conf > ${EXPERIMENT_DIR}/alibaba_trace_replay.output; then
      echo "[x] Failed in the execution of ${LOG_BASE}. Exiting."
      exit 3
  fi

  echo "[x] Finished execution of ${EXPERIMENT_DIR}."
}

if [ ${#MEDIUM_ARRIVAL_RATES[@]} -ne ${#HARD_ARRIVAL_RATES[@]} ]; then
  echo "[x] ERROR: The number of medium and hard arrival rates must be the same."
  exit 1
fi

for SCHEDULER in ${SCHEDULERS[@]}; do
  for RANDOM_SEED in ${RANDOM_SEEDS[@]}; do
    for ((i=0; i<${#MEDIUM_ARRIVAL_RATES[@]}; i++)); do
      MEDIUM_ARRIVAL_RATE=${MEDIUM_ARRIVAL_RATES[$i]}
      HARD_ARRIVAL_RATE=${HARD_ARRIVAL_RATES[$i]}
      echo "[x] Running ${SCHEDULER} with random seed ${RANDOM_SEED} and arrival rates ${MEDIUM_ARRIVAL_RATE} and ${HARD_ARRIVAL_RATE}"
      execute_experiment ${SCHEDULER} ${RANDOM_SEED} ${LOG_DIR} ${MEDIUM_ARRIVAL_RATE} ${HARD_ARRIVAL_RATE}
    done
  done
done