import argparse
import os
import random
import subprocess
import sys
import time
import numpy as np

from pathlib import Path

from workload import JobGraph
from utils import EventTime
from data.tpch_loader import make_release_policy
from rpc import erdos_scheduler_pb2
from rpc import erdos_scheduler_pb2_grpc

import grpc


def map_dataset_to_deadline(dataset_size):
    # 50gb => 2mins, 100gb => 6mins, 250gb => 12mins, 500gb => 24mins
    mapping = {"50": 120, "100": 360, "250": 720, "500": 1440}
    return mapping.get(dataset_size, 120)  # Default to 120s if dataset size is NA


def launch_query(query_number, index, args):
    deadline = map_dataset_to_deadline(args.dataset_size)

    cmd = [
        f"{args.spark_mirror_path.resolve()}/bin/spark-submit",
        *("--deploy-mode", "cluster"),
        *("--master", f"spark://{args.spark_master_ip}:7077"),
        *("--conf", "'spark.port.maxRetries=132'"),
        *("--conf", "'spark.eventLog.enabled=true'"),
        *("--conf", f"'spark.eventLog.dir={args.spark_eventlog_dir.resolve()}'"),
        *("--conf", "'spark.sql.adaptive.enabled=false'"),
        *("--conf", "'spark.sql.adaptive.coalescePartitions.enabled=false'"),
        *("--conf", "'spark.sql.autoBroadcastJoinThreshold=-1'"),
        *("--conf", "'spark.sql.shuffle.partitions=1'"),
        *("--conf", "'spark.sql.files.minPartitionNum=1'"),
        *("--conf", "'spark.sql.files.maxPartitionNum=1'"),
        *("--conf", f"'spark.app.deadline={deadline}'"),
        *("--class", "'main.scala.TpchQuery'"),
        f"{args.tpch_spark_path.resolve()}/target/scala-2.13/spark-tpc-h-queries_2.13-1.0.jar",
        f"{query_number}",
        f"{index}",
        f"{args.dataset_size}",
        f"{args.max_cores}",
    ]

    # print(
    #     f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Launching Query: {query_number}, "
    #     f"dataset: {args.dataset_size}GB, deadline: {deadline}s, maxCores: {args.max_cores}"
    # )

    try:
        cmd = " ".join(cmd)
        print("Launching:", cmd)
        p = subprocess.Popen(
            cmd,
            shell=True,
        )
        print("Query launched successfully.")
        return p
    except Exception as e:
        print(f"Error launching query: {e}")


def generate_release_times(rng, args):
    if args.distribution == "periodic":
        release_policy_args = {
            "period": EventTime(args.period, EventTime.Unit.US),
        }
    elif args.distribution == "fixed":
        release_policy_args = {
            "period": EventTime(args.period, EventTime.Unit.US),
            "num_invocations": args.num_queries,
        }
    elif args.distribution == "poisson":
        release_policy_args = {
            "rate": args.variable_arrival_rate,
            "num_invocations": args.num_queries,
        }
    elif args.distribution == "gamma":
        release_policy_args = {
            "rate": args.variable_arrival_rate,
            "num_invocations": args.num_queries,
            "coefficient": args.coefficient,
        }
    elif args.distribution == "fixed_gamma":
        release_policy_args = {
            "variable_arrival_rate": args.variable_arrival_rate,
            "base_arrival_rate": args.base_arrival_rate,
            "num_invocations": args.num_queries,
            "coefficient": args.coefficient,
        }
    else:
        raise NotImplementedError(
            f"Release policy {args.distribution} not implemented."
        )

    release_policy = make_release_policy(
        args.distribution,
        release_policy_args,
        rng,
        args.rng_seed,
        (args.randomize_start_time_min, args.randomize_start_time_max),
    )

    release_times = release_policy.get_release_times(
        completion_time=EventTime(sys.maxsize, EventTime.Unit.US)
    )

    return release_times


def main():
    parser = argparse.ArgumentParser(
        description="Generate a workload of queries based on distribution type."
    )
    parser.add_argument(
        "--spark-mirror-path",
        type=Path,
        required=True,
        help="Path to spark-mirror repository",
    )
    parser.add_argument(
        "--spark-master-ip",
        type=str,
        required=True,
        help="IP address of node running Spark master",
    )
    parser.add_argument(
        "--tpch-spark-path",
        type=Path,
        required=True,
        help="Path to TPC-H Spark repository",
    )
    parser.add_argument(
        "--spark-eventlog-dir",
        default=Path(os.getcwd()) / "spark-eventlog",
        type=Path,
        help="Path to directory in which to Spark event logs will be dumped",
    )
    parser.add_argument(
        "--distribution",
        choices=["periodic", "fixed", "poisson", "gamma", "closed_loop", "fixed_gamma"],
        default="gamma",
        help="Type of distribution for query inter-arrival times (default: gamma)",
    )
    parser.add_argument(
        "--num_queries",
        type=int,
        default=50,
        help="Number of queries to generate (default: 50)",
    )
    parser.add_argument(
        "--dataset_size",
        choices=["50", "100", "250", "500"],
        default="50",
        help="Dataset size per query in GB (default: 50)",
    )
    parser.add_argument(
        "--max_cores",
        type=int,
        choices=[50, 75, 100, 200],
        default=50,
        help="Maximum executor cores (default: 50)",
    )
    parser.add_argument(
        "--period",
        type=int,
        default=25,
        help="Releases a DAG after period time has elapsed",
    )
    parser.add_argument(
        "--variable_arrival_rate",
        type=float,
        default=1.0,
        help="Variable arrival rate for poisson and gamma distributions",
    )
    parser.add_argument(
        "--coefficient",
        type=float,
        default=1.0,
        help="Coefficient for poisson and gamma distributions",
    )
    parser.add_argument(
        "--base_arrival_rate",
        type=float,
        default=1.0,
        help="Base arrival rate for fixed_gamma distribution",
    )
    parser.add_argument("--randomize_start_time_min", type=int, default=0)
    parser.add_argument("--randomize_start_time_max", type=int, default=0)
    parser.add_argument(
        "--rng_seed",
        type=int,
        default=1234,
        help="RNG seed for generating inter-arrival periods and picking DAGs (default: 1234)",
    )
    parser.add_argument(
        "--queries", type=int, nargs="+", help="Launch specific queries"
    )

    args = parser.parse_args()

    if not args.spark_eventlog_dir.exists():
        args.spark_eventlog_dir.mkdir(parents=True)

    os.environ["TPCH_INPUT_DATA_DIR"] = str(args.tpch_spark_path.resolve() / "dbgen")

    if args.queries:
        assert len(queries) == args.num_queries

    rng = random.Random(args.rng_seed)

    # Generate release times
    release_times = generate_release_times(rng, args)
    print("Release times:", release_times)

    # Launch queries
    ps = []
    inter_arrival_times = [release_times[0].time]
    for i in range(len(release_times) - 1):
        inter_arrival_times.append(release_times[i + 1].time - release_times[i].time)
    for i, inter_arrival_time in enumerate(inter_arrival_times):
        time.sleep(inter_arrival_time)
        if args.queries:
            query_number = args.queries[i]
        else:
            query_number = rng.randint(1, 22)
        ps.append(launch_query(query_number, i, args))
        print(
            f"({i+1}/{len(release_times)})",
            "Current time: ",
            time.strftime("%Y-%m-%d %H:%M:%S"),
            " launching query: ",
            query_number,
        )

    for p in ps:
        p.wait()

    # Wait for some time before sending the shutdown signal
    time.sleep(20)

    channel = grpc.insecure_channel("localhost:50051")
    stub = erdos_scheduler_pb2_grpc.SchedulerServiceStub(channel)
    response = stub.Shutdown(erdos_scheduler_pb2.Empty())
    channel.close()
    print("Sent shutdown signal to the service")


if __name__ == "__main__":
    main()
