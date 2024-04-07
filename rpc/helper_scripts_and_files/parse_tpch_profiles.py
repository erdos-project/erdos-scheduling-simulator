import json
import os
import re
import subprocess


def decompress_file(compressed_file_path):
    # Modify filename to replace "events_1_app" with "decomp_events_1_app"
    directory, filename = os.path.split(compressed_file_path)
    filename_parts = filename.split("_")
    filename_parts = [part if part != "events" else "decomp_events" for part in filename_parts]
    decompressed_file_path = os.path.join(directory, "_".join(filename_parts))

    # Remove the .zstd extension
    decompressed_file_path = os.path.splitext(decompressed_file_path)[0]

    decompress_command = f'zstd -d {compressed_file_path} -o {decompressed_file_path}'
    subprocess.run(decompress_command, shell=True)
    return decompressed_file_path

def parse_event_log(event_log_path):
    stages = []
    stage_info = {}
    tasks_info = {}
    spark_app_name = None
    with open(event_log_path, 'r') as file:
        for line in file:
            # Parse relevant events
            if '"Event":"SparkListenerEnvironmentUpdate"' in line:
                match = re.search(r'"spark.app.name":"(.*?)"', line)
                if match:
                    spark_app_name = match.group(1)
            elif '"Event":"SparkListenerStageSubmitted"' in line:
                stage_info = re.search(r'"Stage ID":(\d+).*?"Stage Name":"(.*?)"'
                                       r'.*?"Number of Tasks":(\d+)', line)
                if stage_info:
                    stage_id = int(stage_info.group(1))
                    stage_name = stage_info.group(2)
                    num_tasks = int(stage_info.group(3))
                    tasks_info[stage_id] = {"num_tasks": num_tasks, "total_runtime": 0, "completed_tasks": 0}
            elif '"Event":"SparkListenerTaskStart"' in line:
                task_info = re.search(r'"Stage ID":(\d+).*?"Task ID":(\d+).*?"Launch Time":(\d+)', line)
                if task_info:
                    stage_id = int(task_info.group(1))
                    task_id = int(task_info.group(2))
                    launch_time = int(task_info.group(3))
                    tasks_info[stage_id][task_id] = {"launch_time": launch_time}
            elif '"Event":"SparkListenerTaskEnd"' in line:
                task_info = re.search(r'"Stage ID":(\d+).*?"Task ID":(\d+).*?"Finish Time":(\d+)', line)
                if task_info:
                    stage_id = int(task_info.group(1))
                    task_id = int(task_info.group(2))
                    finish_time = int(task_info.group(3))
                    launch_time = tasks_info[stage_id][task_id]["launch_time"]
                    runtime = finish_time - launch_time
                    tasks_info[stage_id]["total_runtime"] += runtime
                    tasks_info[stage_id]["completed_tasks"] += 1
                    if tasks_info[stage_id]["completed_tasks"] == tasks_info[stage_id]["num_tasks"]:
                        average_runtime = tasks_info[stage_id]["total_runtime"] / tasks_info[stage_id]["num_tasks"]
                        stages.append({"stage_id": stage_id, "stage_name": stage_name, "num_tasks": tasks_info[stage_id]["num_tasks"], "average_runtime_ms": average_runtime})
    return spark_app_name, stages

def generate_json_for_event_logs(logs_directory):
    job_data = {}
    for root, _, filenames in os.walk(logs_directory):
        for filename in filenames:
            if filename.endswith(".zstd"):  
                event_log_path = os.path.join(root, filename)
                decompressed_file_path = decompress_file(event_log_path)
                spark_app_name, stages = parse_event_log(decompressed_file_path)
                job_data[spark_app_name] = stages

    with open("spark_jobs_data.json", "w") as json_file:
        json.dump(job_data, json_file, indent=4)

def main():
    logs_directory = "/serenity/scratch/dgarg/spark_logs/event_log/temp_event_logs"
    generate_json_for_event_logs(logs_directory)

if __name__ == "__main__":
    main()
