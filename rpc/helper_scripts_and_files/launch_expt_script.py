import argparse
import random
import subprocess
import time

import numpy as np


# Function to parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate a workload of queries based on distribution type.")
    parser.add_argument("--distribution", choices=["gamma", "point"], default="gamma",
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
    random.seed(1234)
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

    # Verify inter-arrival times
    print("Inter-arrival times:", inter_arrival_times)

    # Launch queries after waiting for the required inter-arrival times
    for i, inter_arrival_time in enumerate(inter_arrival_times):
        time.sleep(inter_arrival_time)  # Wait for the inter-arrival time
        query_number = random.randint(1, 22)
        launch_query(query_number, args.dataset_size, args.max_cores)

if __name__ == "__main__":
    main()