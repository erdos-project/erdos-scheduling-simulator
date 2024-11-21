# Setup Instructions for Spark Mirror and ERDOS

This README provides step-by-step instructions to set up the environment, compile the Spark Mirror, and build the ERDOS scheduling simulator.

## Prerequisites
- Conda
- Git
- [Java Development Kit (JDK) 17.0.9](https://openjdk.org/)

---

## Step 0: Create Conda Environment

```bash
conda create -n <env_name> python=3.10
```

### Activate the environment:
```bash
conda activate <env_name>
```

### If jdk17.0.9 isn't installed, install it for <env_name>
```bash
conda install -c conda-forge openjdk=17.0.9
```


## Step 1: Setup `spark-mirror`
Clone the repository with submodules
```bash
git clone https://github.com/dhruvsgarg/spark_mirror.git --recursive
```

### Verify branch
Verify or set current branch `erdos-spark-integration`

### Verify env variable `SPARK_HOME`
Verify or set `SPARK_HOME` to point to the correct location of `spark-mirror`.

### Verify env variable `JAVA_HOME`
NOTE: `JAVA_HOME` should automatically get set to `/path/to/anaconda3/envs/<env_name>/lib/jvm`

### For first time compilation (entire package)
```bash
./build/sbt package
```

### For subsequent, quicker iterations
Start the interactive shell
```bash
./build/sbt
```

Switch to project spark-core
```bash
project core
```

Compile and then package
```bash
compile
package
```

### Fix guava versions for ERDOS-Spark integration
Fresh compile+package of spark adds `guava-14.0.1.jar` under `/path/to/spark_mirror/assembly/target/scala-2.13/jars/`.
This jar interferes with gRPC which requires a `guava-31` jar. To fix:
- Remove existing `guava-14` jar: `rm assembly/target/scala-2.13/jars/guava-14.0.1.jar`
- Run `./sbin/patch-erdos.sh`
- Verify `guava-31.0.1-jre.jar` exists under `assembly/target/scala-2.13/jars/`



## Step 2: Compile ERDOS
NOTE: The `erdos-scheduling-simulator` in Step 2 refers to the seperately cloned repository. It is not the `erdos-scheduling-simulator` submodule within
the `spark-mirror` repository.

### Clone repo
```bash
git clone https://github.com/erdos-project/erdos-scheduling-simulator.git --recursive
```

### Install requirements for the package
```bash
pip install -r requirements.txt
```

### Set `GUROBI_DIR`
```bash
export GUROBI_DIR=/serenity/scratch/dgarg/gurobi/gurobi1003/linux64
```

### Build inside schedulers/tetrisched/build/
```bash
export CMAKE_INSTALL_MODE=ABS_SYMLINK

cmake .. -DINSTALL_GTEST=OFF -DTBB_INSTALL=OFF
```

* Verify that python bindings are written to the new `<env_name>` conda env and not some old env

### Run make
```bash
make -j install
```

### Test that simulator works with `simple_av_workload`
NOTE: Might need to create `experiments` sub-directory if it doesnt already exist
```bash
python3 main.py --flagfile=configs/simple_av_workload.conf > experiments/simple_av_workload_test.output
```
The TaskGraph should complete and meet its deadline.


## Step 3: Spark-Erdos service functionality test
NOTE: As in step 2, the `erdos-scheduling-simulator` here also refers to the seperately cloned repository.

From the base directory:

### Install the requirements
```bash
pip install -r rpc/requirements.txt
```

### Run protoc to generate the service and message definitions using
```bash
python -m grpc_tools.protoc -I./rpc/protos --python_out=. --grpc_python_out=. ./rpc/protos/rpc/erdos_scheduler.proto
```

### Run the service using
```bash
python -m rpc.service
```

### Run local tests for the erdos-spark service
Note: Verify that `pytest` is installed in the `<env_name>`. Else first do `pip install pytest`. Once installed, run the tests using:
```bash
pytest tests/test_service.py
```

## Step 4: Running ERDOS with Spark backend

### Start the service
```bash
python -m rpc.service
```

### Start all components of the spark cluster
Run the following commands from the root directory of the `spark-mirror` repository.

Also, verify that environment variable `SPARK_HOME` is set correctly to point to the path of `spark_mirror`

* Start Spark Master
```bash
./sbin/start-master.sh --host <HOST_IP> --properties-file /path/to/spark_mirror/conf/<CONFIG_FILE_NAME>.conf
```

* Start Spark Worker
```bash
./sbin/start-worker.sh spark://<HOST_IP>:7077 --properties-file /path/to/spark_mirror/conf/<CONFIG_FILE_NAME>.conf
```

* Start Spark History Server
```bash
./sbin/start-history-server.sh --properties-file /path/to/spark_mirror/conf/<CONFIG_FILE_NAME>.conf
```

At this point, the spark framework should be registered with the erdos-service.

### Viewing spark cluster status
TBD

### Shutdown cluster

* To stop all spark services after the experiment concludes
```bash
./sbin/stop-all.sh
```