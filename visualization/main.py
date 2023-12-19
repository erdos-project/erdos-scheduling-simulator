import streamlit as st

from component.utils import *

st.set_page_config(layout="wide")

st.title("Individual Run Analysis")

csv_file_path = st.text_input(
    "CSV file path relative to visualization/main.py",
    "../experiments/ray_osdi_12_12_night_sysml_02_homo_poisson_chain_dag_random_duration_clip_deadline_5/alibaba_scheduler_TetriSched_release_policy_gamma_max_deadline_var_50_dag_aware_1_poisson_arrival_rate_0.012_gamma_coefficient_1_base_arrival_rate_0_scheduler_discretization_1/alibaba_scheduler_TetriSched_release_policy_gamma_max_deadline_var_50_dag_aware_1_poisson_arrival_rate_0.012_gamma_coefficient_1_base_arrival_rate_0_scheduler_discretization_1.csv",
)
workload_file_path = st.text_input(
    "Workload file path relative to visualization/main.py",
    "../traces/alibaba-cluster-trace-v2018/alibaba_straight_chain_7K_dags_minLen4_set1.pkl",
)

csv_data = get_csv_data(csv_file_path)

# Separate Dataframe for completed, cancelled, miss deadline task graphs
df_meet_deadline_task_graphs = csv_data.df_task_graphs[
    (csv_data.df_task_graphs["cancelled"] == False)
    & (csv_data.df_task_graphs["deadline_miss_detected_at"].isnull())
]
df_cancelled_task_graphs = csv_data.df_task_graphs[
    (csv_data.df_task_graphs["cancelled"] == True)
]
df_miss_deadline_task_graphs = csv_data.df_task_graphs[
    (csv_data.df_task_graphs["cancelled"] == False)
    & (csv_data.df_task_graphs["deadline_miss_detected_at"].notnull())
]
length_total_task_graphs = (
    len(df_meet_deadline_task_graphs)
    + len(df_cancelled_task_graphs)
    + len(df_miss_deadline_task_graphs)
)

# Separate Dataframe for completed, cancelled, miss deadline tasks
df_meet_deadline_tasks = csv_data.df_tasks[
    (csv_data.df_tasks["cancelled"] == False)
    & (csv_data.df_tasks["deadline_miss_detected_at"].isnull())
]
df_cancelled_tasks = csv_data.df_tasks[(csv_data.df_tasks["cancelled"] == True)]
df_miss_deadline_tasks = csv_data.df_tasks[
    (csv_data.df_tasks["cancelled"] == False)
    & (csv_data.df_tasks["deadline_miss_detected_at"].notnull())
]

########## Metrics Row ##########
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
col1.metric(
    "%SLO",
    str((len(df_meet_deadline_task_graphs) / length_total_task_graphs) * 100) + "%",
)
col2.metric(
    "Meet Deadline Task Graphs",
    str((len(df_meet_deadline_task_graphs) / length_total_task_graphs) * 100) + "%",
)
col3.metric(
    "Canceled Task Graphs",
    str((len(df_cancelled_task_graphs) / length_total_task_graphs) * 100) + "%",
)
col4.metric(
    "Missed Deadline Task Graphs",
    str((len(df_miss_deadline_task_graphs) / length_total_task_graphs) * 100) + "%",
)
col5.metric(
    "Time spent meet deadline task graphs",
    csv_data.completed_task_graph_run_time - csv_data.miss_deadline_task_graph_run_time,
)
col6.metric("Time spent canceled task graphs", csv_data.canceled_task_graph_run_time)
col7.metric(
    "Time spent missed deadline task graphs", csv_data.miss_deadline_task_graph_run_time
)
########## Metrics Row ##########

smoothing_window_size = st.slider(
    "Smoothing Window Size For Resource Utilization", 0, 20, 10
)
plot_resource_utilization_timeline_chart(
    csv_data.worker_pool_stats, smoothing_window_size
)

st.write("### Worker Pools")
st.dataframe(csv_data.df_worker_pools.drop(["utilizations"], axis=1))

st.write("### Task Graphs Stats")
tab1, tab2, tab3, tab4 = st.tabs(["All", "Meet Deadline", "Cancelled", "Miss Deadline"])
with tab1:
    st.write(csv_data.df_task_graphs.describe(include="all").fillna("").astype("str"))
with tab2:
    st.write(
        df_meet_deadline_task_graphs.describe(include="all").fillna("").astype("str")
    )
with tab3:
    st.write(df_cancelled_task_graphs.describe(include="all").fillna("").astype("str"))
with tab4:
    st.write(
        df_miss_deadline_task_graphs.describe(include="all").fillna("").astype("str")
    )

st.write("### Tasks Stats")
df_tasks_with_cols_useful_for_stats = csv_data.df_tasks.drop(
    ["timestamp", "task_id", "name", "task_graph"], axis=1
)
tab1, tab2, tab3, tab4 = st.tabs(["All", "Meet Deadline", "Cancelled", "Miss Deadline"])
with tab1:
    st.write(
        df_tasks_with_cols_useful_for_stats.describe(include="all")
        .fillna("")
        .astype("str")
    )
with tab2:
    st.write(
        df_meet_deadline_tasks.drop(
            ["timestamp", "task_id", "name", "task_graph"], axis=1
        )
        .describe(include="all")
        .fillna("")
        .astype("str")
    )
with tab3:
    st.write(
        df_cancelled_tasks.drop(["timestamp", "task_id", "name", "task_graph"], axis=1)
        .describe(include="all")
        .fillna("")
        .astype("str")
    )
with tab4:
    st.write(
        df_miss_deadline_tasks.drop(
            ["timestamp", "task_id", "name", "task_graph"], axis=1
        )
        .describe(include="all")
        .fillna("")
        .astype("str")
    )

st.write("### Task Graphs Table")
tab1, tab2, tab3, tab4 = st.tabs(["All", "Meet Deadline", "Cancelled", "Miss Deadline"])
with tab1:
    st.write(csv_data.df_task_graphs.fillna("").astype("str"))
with tab2:
    st.write(df_meet_deadline_task_graphs.fillna("").astype("str"))
with tab3:
    st.write(df_cancelled_task_graphs.fillna("").astype("str"))
with tab4:
    st.write(df_miss_deadline_task_graphs.fillna("").astype("str"))

st.write("### Tasks Table")
tab1, tab2, tab3, tab4 = st.tabs(["All", "Meet Deadline", "Cancelled", "Miss Deadline"])
with tab1:
    st.write(csv_data.df_tasks.fillna("").astype("str"))
with tab2:
    st.write(df_meet_deadline_tasks.fillna("").astype("str"))
with tab3:
    st.write(df_cancelled_tasks.fillna("").astype("str"))
with tab4:
    st.write(df_miss_deadline_tasks.fillna("").astype("str"))

plot_task_placement_timeline_chart(csv_data.df_worker_pools, csv_data.df_tasks)

########## Visualize DAG ##########
trace_data = get_original_trace_data(workload_file_path)

task_graph_id = st.text_input(
    "Enter the Task Graph ID you want to lookup (e.g. j_523717@1)"
)
if task_graph_id != "":
    visualize_task_graph(task_graph_id, csv_data.df_tasks, trace_data)
########## Visualize DAG ##########
