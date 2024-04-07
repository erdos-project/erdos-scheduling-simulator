import json
import os
import re


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
            if filename.startswith("app-"):  
                event_log_path = os.path.join(root, filename)
                print(f"Parsing event log for: {event_log_path}")
                spark_app_name, stages = parse_event_log(event_log_path)
                job_data[spark_app_name] = stages

    with open("spark_jobs_data.json", "w") as json_file:
        json.dump(job_data, json_file, indent=4)

def main():
    logs_directory = "/home/dgarg39/spark_profiles_cloudlab_09_10Mar24/spark_logs/event_log/"
    generate_json_for_event_logs(logs_directory)

if __name__ == "__main__":
    main()
