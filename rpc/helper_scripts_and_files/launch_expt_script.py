import argparse
import os
import random
import subprocess
import sys
import time
import numpy as np

# Make simulator modules accessible
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
two_level_parent_dir = os.path.dirname(parent_dir)
sys.path.append(two_level_parent_dir)

from workload import JobGraph
from utils import EventTime


# Function to parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate a workload of queries based on distribution type.")
    parser.add_argument("--distribution", choices=["periodic", "fixed", "poisson", "gamma", "closed_loop", "fixed_gamma"], default="gamma",
                        help="Type of distribution for query inter-arrival times (default: gamma)")
    parser.add_argument("--mean_qps", type=float, default=0.04,
                        help="Mean query per second ingest rate (default: 0.04)")
    parser.add_argument("--cv_squared", type=float, default=1,
                        help="Coefficient of variation squared for gamma distribution (default: 1)")
    parser.add_argument("--num_queries", type=int, default=50,
                        help="Number of queries to generate (default: 50)")
    parser.add_argument("--dataset_size", choices=["50", "100", "250", "500"], default="50",
                        help="Dataset size per query in GB (default: 50)")
    parser.add_argument("--max_cores", type=int, choices=[50, 75, 100, 200], default=50,
                        help="Maximum executor cores (default: 50)")
    parser.add_argument("--release_generator", type=str, choices=["simulator_release_impl", "launch_script_impl"], default="simulator_release_impl",
                        help="Choose taskgraph release generator implementation between simulator and launch script (default: simulator)")
    
    # Simulator release generator specific arguments
    parser.add_argument("--period", type=int, default=25,
                        help="Releases a DAG after period time has elapsed (default: 25)")
    parser.add_argument("--variable_arrival_rate", type=float, default=1.0,
                        help="Variable arrival rate for poisson and gamma distributions (default: 1.0)")
    parser.add_argument("--coefficient", type=float, default=1.0,
                        help="Coefficient for poisson and gamma distributions (default: 1.0)")
    parser.add_argument("--base_arrival_rate", type=float, default=1.0,
                        help="Base arrival rate for fixed_gamma distribution (default: 1.0)")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Defines the number of concurrent DAGs to execute in closed_loop setting (default: 1)")
    parser.add_argument("--rng_seed", type=int, default=1234,
                        help="RNG seed for generating inter-arrival periods and picking DAGs (default: 1234)")
    
    return parser.parse_args()


# Function to map dataset size (in GB) to deadline in seconds
def map_dataset_to_deadline(dataset_size):
    # 50gb => 2mins, 100gb => 6mins, 250gb => 12mins, 500gb => 24mins
    mapping = {"50": 120, "100": 360, "250": 720, "500": 1440}
    return mapping.get(dataset_size, 120) # Default to 120s if dataset size is NA


# Launch the query
def launch_query(query_number, dataset_size, max_cores):
    deadline = map_dataset_to_deadline(dataset_size)
    spark_submit_command = f"/serenity/scratch/dgarg/anaconda3/condabin/conda run -n spark_feb_9 && /home/dgarg39/spark_feb_9/spark_mirror/bin/spark-submit --deploy-mode cluster --master spark://130.207.125.81:7077 --conf 'spark.port.maxRetries=132' --conf 'spark.eventLog.enabled=true' --conf 'spark.eventLog.dir=/serenity/scratch/dgarg/spark_logs/event_log' --conf 'spark.sql.adaptive.enabled=false' --conf 'spark.sql.adaptive.coalescePartitions.enabled=false' --conf 'spark.sql.autoBroadcastJoinThreshold=-1' --conf 'spark.sql.shuffle.partitions=1' --conf 'spark.sql.files.minPartitionNum=1' --conf 'spark.sql.files.maxPartitionNum=1' --conf 'spark.app.deadline={deadline}' --class 'main.scala.TpchQuery' target/scala-2.13/spark-tpc-h-queries_2.13-1.0.jar {query_number} {dataset_size} {max_cores}"
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Launching Query: {query_number}, "
          f"dataset: {dataset_size}GB, deadline: {deadline}s, maxCores: {max_cores}")
    try:
        subprocess.Popen(spark_submit_command, shell=True)
        print("Query launched successfully.")
    except Exception as e:
        print(f"Error launching query: {e}")


# Function to generate inter-arrival times for point-based distribution
def generate_point_inter_arrival_times(num_queries, mean_qps):
    # Calculate the inter-arrival time based on the mean query per second rate
    inter_arrival_time = 1 / mean_qps
    # Return a list with the same inter-arrival time for each query
    return [inter_arrival_time] * num_queries


# Main function
def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Set the seed for reproducibility for both random and numpy
    rng = random.Random(args.rng_seed)
    
    if args.release_generator == "launch_script_impl":
        np.random.seed(1234)

        # Generate inter-arrival times based on the distribution type
        if args.distribution == "gamma":
            # Function to sample from gamma distribution
            def sample_gamma(mean, cv_squared, seed=None):
                shape = (1 / cv_squared)
                scale = cv_squared / mean
                return np.random.gamma(shape, scale)

            # Generate inter-arrival times based on gamma distribution
            inter_arrival_times = [sample_gamma(args.mean_qps, args.cv_squared) for _ in range(args.num_queries)]
        elif args.distribution == "point":
            # Generate inter-arrival times for point-based distribution
            inter_arrival_times = generate_point_inter_arrival_times(args.num_queries, args.mean_qps)
        else:
            raise ValueError("Distribution type not supported for launch_script_impl generator.")

        # Verify inter-arrival times
        print("Inter-arrival times generated from launch_script_impl:", inter_arrival_times)
        
    elif args.release_generator == "simulator_release_impl":
        # set chosen policy based on the input arg
        chosen_polilcy = None
        if args.distribution == "periodic":
            chosen_polilcy = JobGraph.ReleasePolicyType.PERIODIC
        elif args.distribution == "fixed":
            chosen_polilcy = JobGraph.ReleasePolicyType.FIXED
        elif args.distribution == "poisson":
            chosen_polilcy = JobGraph.ReleasePolicyType.POISSON
        elif args.distribution == "gamma":
            chosen_polilcy = JobGraph.ReleasePolicyType.GAMMA
        elif args.distribution == "closed_loop":
            chosen_polilcy = JobGraph.ReleasePolicyType.CLOSED_LOOP
        elif args.distribution == "fixed_gamma":
            chosen_polilcy = JobGraph.ReleasePolicyType.FIXED_AND_GAMMA
            
        # Use ReleasePolicy in JobGraph to get the release times
        # Create an instance of the release policy
        release_policy = JobGraph.ReleasePolicy(
            policy_type=chosen_polilcy,
            period=EventTime(args.period, EventTime.Unit.US),
            fixed_invocation_nums=args.num_queries,
            variable_arrival_rate=args.variable_arrival_rate,
            coefficient=args.coefficient,
            concurrency=args.concurrency,
            start=EventTime(0, EventTime.Unit.US),
            base_arrival_rate=args.base_arrival_rate,
            rng_seed=args.rng_seed
        )
        
        # Get release times from the simulator using the release
        # policy
        release_times = release_policy.get_release_times(
            completion_time=EventTime(
                        sys.maxsize, EventTime.Unit.US
                    )
        )

        # Verify inter-arrival times
        print("Release times generated from simulator_release_impl:", release_times)

    # Launch queries after waiting for the required inter-arrival
    # times
    if args.release_generator == "launch_script_impl":
        for i, inter_arrival_time in enumerate(inter_arrival_times):
            time.sleep(inter_arrival_time)  # Wait for the inter-arrival time
            query_number = rng.randint(1, 22)
            # launch_query(query_number, args.dataset_size,
            # args.max_cores)
            print("Current time: ", time.strftime('%Y-%m-%d %H:%M:%S'), " launching query: ", query_number)
    elif args.release_generator == "simulator_release_impl":
        # set the current time. Releases will happen relative to
        # it.
        release_start_ts = time.time()
        for i, release_time in enumerate(release_times):
            next_release_ts = release_start_ts + release_time.time
            while time.time() < next_release_ts:
                time.sleep(0.1)
            query_number = rng.randint(1, 22)
            print("Current time: ", time.strftime('%Y-%m-%d %H:%M:%S'), ", time_elapsed: ", str(release_time.time), ", launching query: ", query_number)

if __name__ == "__main__":
    main()