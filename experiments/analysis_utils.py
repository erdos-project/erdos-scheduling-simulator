import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import shutil

def calculate_arrival_rate_and_cv2(release_time: list[int]):
    release_time.sort()
    inter_arrival_times = np.diff(release_time)
    avg_inter_arrival_time = np.mean(inter_arrival_times)
    std_inter_arrival_time = np.std(inter_arrival_times)
    cv2 = (std_inter_arrival_time/avg_inter_arrival_time) ** 2
    return 1/avg_inter_arrival_time, cv2

def find_all_file_paths(path, ends_with=".csv"):
    csv_file_paths = []
    if os.path.isdir(path):
        for filename in os.listdir(path):
            if filename.endswith(ends_with):
                csv_file_paths.append(os.path.join(path, filename))
            else:
                csv_file_paths += find_all_file_paths(os.path.join(path, filename), ends_with)
    return csv_file_paths

def extract_variables_from_filename(filename):
    # Split the filename by underscores
    parts = filename.split('_')

    # Extract the variables based on your format
    replay_trace = parts[0]
    scheduler = parts[2]
    release_policy = parts[5]
    deadline_var = int(parts[9])
    dag_aware = parts[12] == "1"
    
    try:
        arrival_rate = float(parts[16])
        cv2 = int(parts[19].split('.')[0])  # Assuming the file extension is .csv
    except:
        # Before 11/28 afternoon, I used a different format for the filename and didn't include the arrival rate and CV2
        arrival_rate = 10
        cv2 = 2

    if scheduler == "TetriSched":
        scheduler_time_discretization = int(parts[-1].split('.')[0])
        scheduler = f"TetriSched_time_dis_{scheduler_time_discretization}" + ("_DAG_aware" if dag_aware else "")
    else:
        scheduler_time_discretization = None

    # Create a dictionary to store the extracted variables
    variables = {
        'trace': replay_trace,
        'release_policy': release_policy,
        'max_deadline_variance': deadline_var,
        'scheduler': scheduler,
        'DAG_aware': dag_aware,
        'scheduler_time_discretization': scheduler_time_discretization,
        "arrival_rate": arrival_rate,
        "cv2": cv2,
    }

    return variables


def extract_experiments_result(base_dir: str) -> pd.DataFrame:
    rows = []
    # Loop through each folder and process the CSV file
    for csv_file_path in find_all_file_paths(base_dir):
        file_name = csv_file_path.split(os.sep)[-1]
        try:
            # Open the CSV file and read the last line
            with open(csv_file_path, 'r') as file:
                lines = file.readlines()
                last_line = lines[-1]
            
            end_time, _, finished_tasks, cancelled_tasks, missed_task_deadlines, finished_task_graphs, cancelled_task_graphs, missed_task_graph_deadlines = last_line.split(",")
            row = extract_variables_from_filename(file_name)
            # Analyze SLO attainment and goodput
            slo_attainment = (int(finished_task_graphs) - int(missed_task_graph_deadlines)) / (int(cancelled_task_graphs) + int(finished_task_graphs))
            row["slo_attainment"] = slo_attainment
            row["goodput"] = int(finished_tasks)
            row["csv_file_path"] = csv_file_path
            
            # Calculate the arrival rate and cv2
            release_times = []
            for line in lines:
                if "TASK_RELEASE" not in line:
                    continue
                # event_time should be the actual release time
                event_time, _, task_name, _, task_intended_release_time, task_release_time, task_deadline, task_id, task_graph = line.strip().split(",")
                release_times.append(int(task_release_time))
            
            actual_arrival_rate, actual_cv2 = calculate_arrival_rate_and_cv2(release_times)
            row["actual_arrival_rate"] = actual_arrival_rate
            row["actual_cv2"] = actual_cv2
            
            rows.append(row)
        except FileNotFoundError:
            print(f"File not found: {csv_file_path}")
        except Exception as e:
            print(f"An error occurred while processing {csv_file_path}: {str(e)}")
            # I want to remove the parent folder of the CSV file
            # print(f"Removing {os.path.dirname(csv_file_path)}")
            # shutil.rmtree(os.path.dirname(csv_file_path)) 

    return pd.DataFrame(rows)


def plot_slo_attainments(data: pd.DataFrame):
    # Define your unique values for the grid
    cv2_values = sorted(data["cv2"].unique())
    arrival_rate_values = sorted(data["arrival_rate"].unique())
    scheduler_values = ["TetriSched_time_dis_20", "TetriSched_time_dis_20_DAG_aware", "TetriSched_time_dis_10", 
                        "TetriSched_time_dis_10_DAG_aware", "TetriSched_time_dis_1",  "TetriSched_time_dis_1_DAG_aware", "EDF"]

    # Number of schedulers
    n_schedulers = len(scheduler_values)

    # Create a subplot grid
    fig, axes = plt.subplots(len(arrival_rate_values), len(cv2_values), figsize=(20, 15), sharey=True)

    # Define the width of each bar and the spacing between them
    bar_width = 0.20
    spacing = 0.05
    group_width_factor = 2  # Increase this factor to widen the distance between groups

    # Collect handles and labels for the legend
    handles, labels = [], []

    # Iterate over each subplot and plot the data
    for i, arrival_rate in enumerate(arrival_rate_values):
        for j, cv2 in enumerate(cv2_values):
            ax = axes[i][j]
            subset = data[(data['arrival_rate'] == arrival_rate) & (data['cv2'] == cv2)]

            # Get unique deadline variances
            deadline_vars = sorted(subset['max_deadline_variance'].unique())
            x = np.arange(len(deadline_vars)) * group_width_factor  # Adjust x positions

            for k, scheduler in enumerate(scheduler_values):
                scheduler_data = subset[subset['scheduler'] == scheduler]
                # Calculate the position of each bar
                bar_positions = x - (n_schedulers * bar_width / 2) + (k * bar_width) + (spacing * k)
                # Some bars may not exist for some schedulers
                slo_attainments = []
                for deadline_var in deadline_vars:
                    if len(scheduler_data[scheduler_data['max_deadline_variance'] == deadline_var]['slo_attainment']) == 0:
                        slo_attainments.append(0)
                    else:
                        slo_attainments.append(scheduler_data[scheduler_data['max_deadline_variance'] == deadline_var]['slo_attainment'].item())
                
                ax.bar(bar_positions, slo_attainments, width=bar_width, label=scheduler)

            for c in ax.containers:
                labels = [f'{(v.get_height() * 100):.1f}' for v in c]
                ax.bar_label(c, labels=labels, label_type='edge', rotation=45, size=8)

            ax.set_xticks(x)
            ax.set_xticklabels(deadline_vars)
            
            # This is the actual "task" arrival rate and cv2
            # ax.set_title(f"Arrival Rate: {subset['actual_arrival_rate'].mean():.2f}, CV2: {subset['actual_cv2'].mean():.2f}")
            
            # This is the arrival rate and cv2 we specified for "task graph" in the config file
            ax.set_title(f"Arrival Rate: {arrival_rate}, CV2: {cv2}")
            
            ax.set_xlabel('Max Deadline Variance')
            ax.set_ylabel('SLO Attainment')

    # Adjust layout and add a super title
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust the bottom parameter to make space for the legend

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=len(labels))

    plt.suptitle('SLO Attainment Comparison (min_deadline_var=10, num_invocation=400) 11_29_2023', size=16)

    # Show the plot
    plt.show()