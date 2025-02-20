# Setup Instructions for Spark Mirror and ERDOS

This README provides step-by-step instructions to set up the environment, compile the Spark Mirror, and build the ERDOS scheduling simulator.

## Prerequisites
- Conda
- Git
- [Java Development Kit (JDK) 17.0.9](https://openjdk.org/)

---

## Step 0A: Create Conda Environment
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

## Step 0B: Setup TPCH (dataset, jar) workload
Build the dataset
```bash
cd /path/to/tpch-spark/dbgen

make

./dbgen
```

Running `./dbgen` above creates a dataset of scale factor `s` of `1` (default) i.e. 1GB.

> NOTE: Had updated the scala version to 2.13.0 in tpch.sbt. The sbt version used was `1.9.7`.

Next, we build the target for `tpch-spark`:
```bash
sbt package
```

> NOTE: In case of errors in building the target, check `openjdk` version. It should be `17` and not `21`.


## Step 1: Setup `spark-mirror`
Clone the repository with submodules
```bash
git clone https://github.com/dhruvsgarg/spark_mirror.git --recursive
```

> NOTE: If the submodule was cloned earlier but has been updated since, `git fetch --all` will not be able to track those changes. To pull in updates
> from submodule's parent, run `git submodule update --init --recursive`.

### Verify branch
Verify or set current branch `erdos-spark-integration`

### Verify env variable `SPARK_HOME`
Verify or set `SPARK_HOME` to point to the correct location of `spark-mirror`.

### Verify env variable `JAVA_HOME`
> NOTE: `JAVA_HOME` should automatically get set to `/path/to/anaconda3/envs/<env_name>/lib/jvm`

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

### Update `PATH` with spark bin files
```bash
export PATH=$PATH:/path/to/spark_mirror/bin
```

## Step 2: Compile ERDOS
> NOTE: The `erdos-scheduling-simulator` in Step 2 refers to the seperately cloned repository. It is not the `erdos-scheduling-simulator` submodule within
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
> NOTE: Might need to create `experiments` sub-directory if it doesnt already exist
```bash
python3 main.py --flagfile=configs/simple_av_workload.conf > experiments/simple_av_workload_test.output
```
The TaskGraph should complete and meet its deadline.


## Step 3: Spark-Erdos service functionality test
> NOTE: As in step 2, the `erdos-scheduling-simulator` here also refers to the seperately cloned repository.

From the base directory:

### Install the requirements
```bash
pip install -r rpc/requirements.txt
```

### Run protoc to generate the service and message definitions using
```bash
python -m grpc_tools.protoc -I./rpc/protos --python_out=. --grpc_python_out=. ./rpc/protos/rpc/erdos_scheduler.proto
```

### Run the service
```bash
python -m rpc.service --enforce_deadlines --scheduler_runtime=0 
```

The above command uses the default argument values from the `service.py` and `main.py`. The default scheduler is `EDF`. Other options available for the
service are `FIFO` and `TetriSched`. The DSched scheduler is a specific instantiation of the `TetriSched` scheduler. The other schedulers can be run
as follows:

#### To instantiate FIFO scheduler for the service:
```bash
python -m rpc.service --scheduler=FIFO --enforce_deadlines --scheduler_runtime=0 
```

#### To instantiate DSched scheduler for the service:
```bash
python -m rpc.service --scheduler=TetriSched --enforce_deadlines --scheduler_runtime=0 --release_taskgraphs --opt_passes=CRITICAL_PATH_PASS --opt_passes=CAPACITY_CONSTRAINT_PURGE_PASS --oppasses=DYNAMIC_DISCRETIZATION_PASS --retract_schedules --scheduler_max_occupancy_threshold=0.999 --finer_discretization_at_prev_solution --scheduler_selective_rescheduling --scheduler_reconsideration_period=0.6  --scheduler_time_discretization=1 --scheduler_max_time_discretization=5 --finer_discretization_window=5 --scheduler_log_to_file  
```

### Run local tests for the erdos-spark service
> NOTE: Verify that `pytest` is installed in the `<env_name>`. Else first do `pip install pytest`. Once installed, run the tests using:
```bash
pytest tests/test_service.py
```

## Step 4: Running ERDOS with Spark backend

### Start the service
```bash
python -m rpc.service
```
Refer to the above section to instantiate different schedulers for the service. 

> NOTE: Since we emulate a 20-node spark cluster on a single system, an additional flag `--override_worker_cpu_count` needs to be passed in the
> service launch command.

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
Start a ssh tunnel to the node hosting the spark cluster and access port `18080` using the command:
```bash
ssh -L 18080:<HOST_IP>:18080 <username>@<node_ip>
```

Once this command succeeds, you can view the History Server on your laptop's browser at URL: `localhost:18080`

> NOTE: Same process needs to be repeated to view Master and Worker UIs. They run on ports `8080` and `8081` respectively.

### Submitting a test spark application
To be submitted from within the `tpch-spark` repo:
```bash
/path/to/spark_mirror/bin/spark-submit --deploy-mode cluster --master spark://<NODE_IP>:7077 --conf 'spark.port.maxRetries=132' --conf 'spark.eventLog.enabled=true' --conf 'spark.eventLog.dir=/path/to/event_log' --conf 'spark.sql.adaptive.enabled=false' --conf 'spark.sql.adaptive.coalescePartitions.enabled=false' --conf 'spark.sql.autoBroadcastJoinThreshold=-1' --conf 'spark.sql.shuffle.partitions=1' --conf 'spark.sql.files.minPartitionNum=1' --conf 'spark.sql.files.maxPartitionNum=1' --conf 'spark.app.deadline=120' --class 'main.scala.TpchQuery' target/scala-2.13/spark-tpc-h-queries_2.13-1.0.jar "4" "50" "50"
```

The above job submission is parameterized by `(DEADLINE, QUERY_NUM, DATASET_SIZE, MAX_CORES)`. An example input value for this tuple is 
`(120, 4, 50, 50)`.
> Refer to `launch_expt_script.py` in `tpch-spark` for more details on eligible values for these parameters and how they are used.

> NOTE: By default, env variable `TPCH_INPUT_DATA_DIR` will look for `dbgen` inside the current working directory. While it works for `spark-submit`
> issued from inside the `tpch-spark` repository, it needs to be explicitly set otherwise. 

Once submitted, review the application's runtime status on the Spark Web UI.

### Shutdown cluster
* To stop the master and worker(s) after the experiment concludes, run:
```bash
./sbin/stop-all.sh
```

> NOTE: This command does not terminate the History Server process.