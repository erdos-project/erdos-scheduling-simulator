ERDOS RPC Service
=============================

The package provides support for connecting frameworks to the ERDOS Simulator, with a goal of trying out different backend algorithms.

This code is being tested with Apache Spark v3.5.0 (with additional instrumentation outlined in [this](https://github.com/dhruvsgarg/spark_mirror/tree/erdos-spark-integration) repository)

To get the RPC service setup, first install the required packages using:

```bash
pip install -r requirements.txt
```

Then, run protoc to generate the service and message definitions using:

```bash
python -m grpc_tools.protoc -I./protos --python_out=./ --grpc_python_out=./ ./protos/erdos_scheduler.proto
```

and run the service using:

```bash
python service.py
```

You can also find the supported flags by the service, by running

```bash
python service.py --help
```
