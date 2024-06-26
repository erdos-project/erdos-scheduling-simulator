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
#RANDOM_SEEDS=(420665456 6785649879 1232434 243243453453 3785432875 8984928429 4295429857 99854278957 32542345235 67676 1979879073895 1)
#RANDOM_SEEDS=(1232434 42066545 6785649879 32434234353 106432512)
#RANDOM_SEEDS=(1232434 42066545 106432512)
RANDOM_SEEDS=(1232434)

# Schedulers
# We use the following baseline schedulers to compare the performance of DAGSched with.
#SCHEDULERS=(EDF DAGSched_Dyn)
#SCHEDULERS=(EDF DAGSched_Dyn)
#SCHEDULERS=(DAGSched_Dyn TetriSched_1 TetriSched_5)
#SCHEDULERS=(EDF TetriSched_0 DAGSched_Dyn Graphene)
SCHEDULERS=(FIFO)

# Poisson arrival rates.
# We use the following arrival rates for the Poisson arrival process.
#MEDIUM_ARRIVAL_RATES=(0.0175 0.0175 0.02 0.02 0.0225 0.0225 0.0225)
#HARD_ARRIVAL_RATES=(0.0125 0.015 0.015 0.0175 0.0175 0.02 0.015)
#MEDIUM_ARRIVAL_RATES=(0.035 0.038 0.04 0.042 0.04 0.05 0.045 0.045 0.075 0.06 0.055)
#HARD_ARRIVAL_RATES=(0.022 0.025 0.0275 0.028 0.028 0.035 0.03 0.025 0.055 0.04 0.038)
#MEDIUM_ARRIVAL_RATES=(0.035 0.032 0.03 0.038 0.04 0.0425 0.045 0.034 0.033 0.032 0.035 0.035 0.034 0.034 0.028 0.03)
#HARD_ARRIVAL_RATES=(0.022 0.02 0.0175 0.025 0.0275 0.029 0.031 0.021 0.02 0.019 0.02 0.021 0.022 0.021 0.024 0.021)
#MEDIUM_ARRIVAL_RATES=(0.034 0.033 0.032 0.04 0.029 0.029 0.01)
#HARD_ARRIVAL_RATES=(0.022 0.021 0.022 0.015 0.02 0.018 0.06)
#MEDIUM_ARRIVAL_RATES=(0.01 0.01 0.01 0.01 0.01  0.01 0.01    0.01   0.01   0.01    0.01   0.01)
#HARD_ARRIVAL_RATES=(  0.06 0.05 0.04 0.03 0.045 0.055 0.0425 0.0525 0.0535 0.0545  0.053  0.054)
#MEDIUM_ARRIVAL_RATES=(0.01 0.01)
#HARD_ARRIVAL_RATES=(  0.06 0.05)
#MEDIUM_ARRIVAL_RATES=(0.01 0.01  0.01    0.01   0.01   0.01    0.01   0.01 0.01 0.01)
#HARD_ARRIVAL_RATES=(  0.06 0.05 0.055 0.0525 0.0535 0.0545  0.053  0.054 0.0475 0.045)
#MEDIUM_ARRIVAL_RATES=( 0.032 0.022 0.015 0.01  0.0075 0.005 0.0025 0.0025 0.0025  0.0025  0.0025)
#HARD_ARRIVAL_RATES=(   0.045 0.045 0.045 0.045  0.045 0.045  0.045 0.04   0.035   0.03    0.025)
#MEDIUM_ARRIVAL_RATES=( 0.032 0.022 0.015 0.01  0.0075 0.005 0.0025 0.0025)
#HARD_ARRIVAL_RATES=(   0.045 0.045 0.045 0.045  0.045 0.045  0.045 0.04)
#MEDIUM_ARRIVAL_RATES=( 0.032 0.022 0.032 0.042)
#HARD_ARRIVAL_RATES=(   0.045 0.045 0.05  0.05)
#MEDIUM_ARRIVAL_RATES=( 0.032 0.032 0.042 0.04 0.03 0.025 0.02 0.015 0.01)
#HARD_ARRIVAL_RATES=(   0.045 0.05  0.05  0.06 0.04 0.035 0.03 0.03  0.02)
#MEDIUM_ARRIVAL_RATES=( 0.03 0.025 0.02 0.015 0.01 0.012 0.014 0.015 0.016 0.015 0.018)
#HARD_ARRIVAL_RATES=(   0.04 0.035 0.03 0.03  0.02 0.021 0.022 0.024 0.025 0.028 0.025)

#MEDIUM_ARRIVAL_RATES=( 0.01  0.01 0.008 0.01 0.025 0.02 0.012 0.014 0.015 0.016 0.01 0.01  0.006 0.008  0.01   0.01)
#HARD_ARRIVAL_RATES=(  0.018 0.015  0.02 0.02 0.035 0.03 0.021 0.022 0.024 0.025 0.05 0.024 0.023 0.0225 0.0235 0.021)
#MEDIUM_ARRIVAL_RATES=(  0.01   0.01   0.01    0.01    0.01  0.015  0.012  0.0135  0.0135  0.013  0.014  0.0145  0.016   0.01    0.01    0.01)
#HARD_ARRIVAL_RATES=(   0.012  0.014  0.015  0.0165  0.0175  0.014   0.02    0.02   0.022  0.025  0.025   0.026  0.026  0.033  0.0345  0.0385)
MEDIUM_ARRIVAL_RATES=(  0.01)
HARD_ARRIVAL_RATES=(   0.012)
#MEDIUM_ARRIVAL_RATES=(  0.01 0.02  0.03)
#HARD_ARRIVAL_RATES=(   0.045 0.05  0.05)
# Parallel Factor
# The number of experiments to run in parallel.
PARALLEL_FACTOR=3


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
  --worker_profile_path=profiles/workers/alibaba_cluster_25k.yaml
  "

  EXPERIMENT_CONF+="
  # Workload configuration.
  --execution_mode=replay
  --replay_trace=alibaba
  --workload_profile_paths=traces/alibaba-cluster-trace-v2018/alibaba_filtered_new_alind_easy_dags_till_30k_cpu_usage.pkl,traces/alibaba-cluster-trace-v2018/medium_filtered.pkl,traces/alibaba-cluster-trace-v2018/hard_filtered.pkl
  --workload_profile_path_labels=easy,medium,hard
  --override_release_policies=poisson,poisson,poisson
  --override_num_invocations=0,350,650
  --override_poisson_arrival_rates=0.0075,${MEDIUM_ARRIVAL_RATE},${HARD_ARRIVAL_RATE}
  --randomize_start_time_max=50
  --min_deadline=5
  --max_deadline=500
  --min_deadline_variances=25,50,10
  --max_deadline_variances=50,100,25

  # Loader configuration.
  --alibaba_loader_task_cpu_usage_random
  --alibaba_loader_task_cpu_multiplier=1
  --alibaba_loader_task_cpu_usage_min=120
  --alibaba_loader_task_cpu_usage_max=1500
  --alibaba_loader_min_critical_path_runtimes=200,500,600
  --alibaba_loader_max_critical_path_runtimes=500,1000,1000
  "

  if [[ ${SCHEDULER} == "EDF" ]]; then
    EXPERIMENT_CONF+="
    # Scheduler configuration.
    --scheduler=EDF
    --enforce_deadlines
    "
  elif [[ ${SCHEDULER} == "FIFO" ]]; then
    EXPERIMENT_CONF+="
    # Scheduler configuration.
    --scheduler=FIFO
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
  elif [[ ${SCHEDULER} == "Graphene" ]]; then
    EXPERIMENT_CONF+="
    # Scheduler configuration.
    --scheduler=Graphene
    --scheduler_time_discretization=1
    --scheduler_enable_optimization_pass
    --retract_schedules
    --scheduler_plan_ahead=0
    --scheduler_time_limit=60
    "
  elif [[ ${SCHEDULER} == "DAGSched" ]]; then
    EXPERIMENT_CONF+="
    # Scheduler configuration.
    --scheduler=TetriSched
    --release_taskgraphs
    --enforce_deadlines
    --scheduler_time_discretization=1
    --scheduler_enable_optimization_pass
    --scheduler_reconsideration_period=0.6
    --retract_schedules
    "
  elif [[ ${SCHEDULER} == "DAGSched_2" ]]; then
    EXPERIMENT_CONF+="
    # Scheduler configuration.
    --scheduler=TetriSched
    --release_taskgraphs
    --enforce_deadlines
    --scheduler_time_discretization=2
    --scheduler_enable_optimization_pass
    --scheduler_reconsideration_period=0.6
    --retract_schedules
    "
  elif [[ ${SCHEDULER} == "DAGSched_4" ]]; then
    EXPERIMENT_CONF+="
    # Scheduler configuration.
    --scheduler=TetriSched
    --release_taskgraphs
    --enforce_deadlines
    --scheduler_time_discretization=4
    --scheduler_enable_optimization_pass
    --scheduler_reconsideration_period=0.6
    --retract_schedules
    "
  elif [[ ${SCHEDULER} == "DAGSched_8" ]]; then
    EXPERIMENT_CONF+="
    # Scheduler configuration.
    --scheduler=TetriSched
    --release_taskgraphs
    --enforce_deadlines
    --scheduler_time_discretization=8
    --scheduler_enable_optimization_pass
    --scheduler_reconsideration_period=0.6
    --retract_schedules
    "
  elif [[ ${SCHEDULER} == "DAGSched_16" ]]; then
    EXPERIMENT_CONF+="
    # Scheduler configuration.
    --scheduler=TetriSched
    --release_taskgraphs
    --enforce_deadlines
    --scheduler_time_discretization=16
    --scheduler_enable_optimization_pass
    --scheduler_reconsideration_period=0.6
    --retract_schedules
    "
  elif [[ ${SCHEDULER} == "DAGSched_32" ]]; then
    EXPERIMENT_CONF+="
    # Scheduler configuration.
    --scheduler=TetriSched
    --release_taskgraphs
    --enforce_deadlines
    --scheduler_time_discretization=32
    --scheduler_enable_optimization_pass
    --scheduler_reconsideration_period=0.6
    --retract_schedules
    "
  elif [[ ${SCHEDULER} == "DAGSched_Dyn" ]]; then
    EXPERIMENT_CONF+="
    # Scheduler configuration.
    --scheduler=TetriSched
    --release_taskgraphs
    --enforce_deadlines
    --scheduler_time_discretization=1
    --scheduler_enable_optimization_pass
    --retract_schedules
    --scheduler_dynamic_discretization
    --scheduler_max_time_discretization=8
    --scheduler_max_occupancy_threshold=0.999
    --finer_discretization_at_prev_solution
    --finer_discretization_window=4
    --scheduler_selective_rescheduling
    --scheduler_reconsideration_period=0.6
    "
  elif [[ ${SCHEDULER} == "TetriSched_1" ]]; then
    EXPERIMENT_CONF+="
    # Scheduler configuration.
    --scheduler=TetriSched
    --enforce_deadlines
    --scheduler_time_discretization=1
    --scheduler_enable_optimization_pass
    --retract_schedules
    --scheduler_plan_ahead=1
    "
  elif [[ ${SCHEDULER} == "TetriSched_5" ]]; then
    EXPERIMENT_CONF+="
    # Scheduler configuration.
    --scheduler=TetriSched
    --enforce_deadlines
    --scheduler_time_discretization=1
    --scheduler_enable_optimization_pass
    --retract_schedules
    --scheduler_plan_ahead=5
    "
  elif [[ ${SCHEDULER} == "TetriSched_0" ]]; then
    EXPERIMENT_CONF+="
    # Scheduler configuration.
    --scheduler=TetriSched
    --enforce_deadlines
    --scheduler_time_discretization=1
    --scheduler_enable_optimization_pass
    --retract_schedules
    --scheduler_plan_ahead=0
    "
  elif [[ ${SCHEDULER} == "TetriSched_10" ]]; then
    EXPERIMENT_CONF+="
    # Scheduler configuration.
    --scheduler=TetriSched
    --enforce_deadlines
    --scheduler_time_discretization=1
    --scheduler_enable_optimization_pass
    --retract_schedules
    --scheduler_plan_ahead=10
    "
  elif [[ ${SCHEDULER} == "TetriSched_20" ]]; then
    EXPERIMENT_CONF+="
    # Scheduler configuration.
    --scheduler=TetriSched
    --enforce_deadlines
    --scheduler_time_discretization=1
    --scheduler_enable_optimization_pass
    --retract_schedules
    --scheduler_plan_ahead=20
    "
  elif [[ ${SCHEDULER} == "TetriSched_1_n" ]]; then
    EXPERIMENT_CONF+="
    # Scheduler configuration.
    --scheduler=TetriSched
    --enforce_deadlines
    --scheduler_time_discretization=1
    --scheduler_enable_optimization_pass
    --retract_schedules
    --scheduler_plan_ahead=1
    --ilp_goal=min_placement_delay
    "
  elif [[ ${SCHEDULER} == "TetriSched_5_n" ]]; then
    EXPERIMENT_CONF+="
    # Scheduler configuration.
    --scheduler=TetriSched
    --enforce_deadlines
    --scheduler_time_discretization=1
    --scheduler_enable_optimization_pass
    --retract_schedules
    --scheduler_plan_ahead=5
    --ilp_goal=min_placement_delay
    "
  elif [[ ${SCHEDULER} == "TetriSched_10_n" ]]; then
    EXPERIMENT_CONF+="
    # Scheduler configuration.
    --scheduler=TetriSched
    --enforce_deadlines
    --scheduler_time_discretization=1
    --scheduler_enable_optimization_pass
    --retract_schedules
    --scheduler_plan_ahead=10
    --ilp_goal=min_placement_delay
    "
  elif [[ ${SCHEDULER} == "TetriSched_20_n" ]]; then
    EXPERIMENT_CONF+="
    # Scheduler configuration.
    --scheduler=TetriSched
    --enforce_deadlines
    --scheduler_time_discretization=1
    --scheduler_enable_optimization_pass
    --retract_schedules
    --scheduler_plan_ahead=20
    --ilp_goal=min_placement_delay
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

  if ! TETRISCHED_LOGGING_DIR=${EXPERIMENT_DIR} python3 main.py --flagfile=${EXPERIMENT_DIR}/alibaba_trace_replay.conf > ${EXPERIMENT_DIR}/alibaba_trace_replay.output; then
      echo "[x] Failed in the execution of ${LOG_BASE}. Exiting."
      exit 3
  fi

  echo "[x] Finished execution of ${EXPERIMENT_DIR}."
}

if [ ${#MEDIUM_ARRIVAL_RATES[@]} -ne ${#HARD_ARRIVAL_RATES[@]} ]; then
  echo "[x] ERROR: The number of medium and hard arrival rates must be the same."
  exit 1
fi

for RANDOM_SEED in ${RANDOM_SEEDS[@]}; do
  for ((i=0; i<${#MEDIUM_ARRIVAL_RATES[@]}; i++)); do
    for SCHEDULER in ${SCHEDULERS[@]}; do
      MEDIUM_ARRIVAL_RATE=${MEDIUM_ARRIVAL_RATES[$i]}
      HARD_ARRIVAL_RATE=${HARD_ARRIVAL_RATES[$i]}
      echo "[x] Running ${SCHEDULER} with random seed ${RANDOM_SEED} and arrival rates ${MEDIUM_ARRIVAL_RATE} and ${HARD_ARRIVAL_RATE}"
      execute_experiment ${SCHEDULER} ${RANDOM_SEED} ${LOG_DIR} ${MEDIUM_ARRIVAL_RATE} ${HARD_ARRIVAL_RATE} &

      if [[ $(jobs -r -p | wc -l) -ge $PARALLEL_FACTOR ]]; then
          echo "[x] Waiting for a job to terminate because $PARALLEL_FACTOR jobs are running."
          wait -n
      fi
    done
  done
done

wait
echo "[x] Finished all experiments."
