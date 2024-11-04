ERDOS RPC Service
=============================

The package provides support for connecting frameworks to the ERDOS Simulator, with a goal of trying out different backend algorithms.

This code is being tested with Apache Spark v3.5.0 (with additional instrumentation outlined in [this](https://github.com/dhruvsgarg/spark_mirror/tree/erdos-spark-integration) repository)

To get the RPC service setup, from the ERDOS root directory, install the required packages using:

```bash
pip install -r rpc/requirements.txt
```

Then, run protoc to generate the service and message definitions using:

```bash
python -m grpc_tools.protoc -I./rpc/protos --python_out=. --grpc_python_out=. ./rpc/protos/rpc/erdos_scheduler.proto
```

and run the service using:

```bash
python -m rpc.service
```

You can also find the supported flags by the service, by running

```bash
python -m rpc.service --help
```
