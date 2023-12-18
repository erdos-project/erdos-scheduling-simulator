import streamlit as st

from component.utils import *

st.set_page_config(layout="wide")

st.title("Individual Run Analysis")

csv_file_path = st.text_input(
    "CSV file path",
    "../experiments/ray_osdi_12_12_night_sysml_02_homo_poisson_chain_dag_random_duration_clip_deadline_5/alibaba_scheduler_TetriSched_release_policy_gamma_max_deadline_var_50_dag_aware_1_poisson_arrival_rate_0.012_gamma_coefficient_1_base_arrival_rate_0_scheduler_discretization_1/alibaba_scheduler_TetriSched_release_policy_gamma_max_deadline_var_50_dag_aware_1_poisson_arrival_rate_0.012_gamma_coefficient_1_base_arrival_rate_0_scheduler_discretization_1.csv",
)
workload_file_path = st.text_input(
    "Workload file path",
    "../traces/alibaba-cluster-trace-v2018/alibaba_straight_chain_7K_dags_minLen4_set1.pkl",
)

csv_data = get_csv_data(csv_file_path)

col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
col1.metric(
    "%SLO",
    str(
        len(
            csv_data.df_task_graphs[
                (csv_data.df_task_graphs["cancelled"] == False)
                & (csv_data.df_task_graphs["deadline_miss_detected_at"].isnull())
            ]
        )
        / len(csv_data.df_task_graphs)
        * 100
    )
    + "%",
)
col2.metric(
    "Finished Task Graphs",
    len(csv_data.df_task_graphs[(csv_data.df_task_graphs["cancelled"] == False)]),
)
col3.metric(
    "Canceled Task Graphs",
    len(csv_data.df_task_graphs[(csv_data.df_task_graphs["cancelled"] == True)]),
)
col4.metric(
    "Missed Deadline Task Graphs",
    len(
        csv_data.df_task_graphs[
            (csv_data.df_task_graphs["deadline_miss_detected_at"].notnull())
        ]
    ),
)
col5.metric("Time spent completed task graphs", csv_data.completed_task_graph_run_time)
col6.metric("Time spent canceled task graphs", csv_data.canceled_task_graph_run_time)
col7.metric(
    "Time spent missed deadline task graphs", csv_data.miss_deadline_task_graph_run_time
)

st.write("### Worker Pools")
st.dataframe(csv_data.df_worker_pools.drop(["utilizations"], axis=1))

st.write("### Task Graphs Stats")
tab1, tab2, tab3 = st.tabs(["All", "Canceled", "Completed"])
with tab1:
    st.write(csv_data.df_task_graphs.describe(include="all").fillna("").astype("str"))
with tab2:
    st.write(
        csv_data.df_task_graphs[csv_data.df_task_graphs["cancelled"] == True]
        .describe(include="all")
        .fillna("")
        .astype("str")
    )
with tab3:
    st.write(
        csv_data.df_task_graphs[csv_data.df_task_graphs["cancelled"] == False]
        .describe(include="all")
        .fillna("")
        .astype("str")
    )

st.write("### Tasks Stats")
df_tasks_with_cols_useful_for_stats = csv_data.df_tasks.drop(
    ["timestamp", "task_id", "name", "task_graph"], axis=1
)
tab1, tab2, tab3 = st.tabs(["All", "Canceled", "Completed"])
with tab1:
    st.write(
        df_tasks_with_cols_useful_for_stats.describe(include="all")
        .fillna("")
        .astype("str")
    )
with tab2:
    st.write(
        df_tasks_with_cols_useful_for_stats[
            df_tasks_with_cols_useful_for_stats["cancelled"] == True
        ]
        .describe(include="all")
        .fillna("")
        .astype("str")
    )
with tab3:
    st.write(
        df_tasks_with_cols_useful_for_stats[
            df_tasks_with_cols_useful_for_stats["cancelled"] == False
        ]
        .describe(include="all")
        .fillna("")
        .astype("str")
    )


st.write("### Task Graphs Table")
tab1, tab2, tab3 = st.tabs(["All", "Canceled", "Completed"])
with tab1:
    csv_data.df_task_graphs
with tab2:
    csv_data.df_task_graphs[csv_data.df_task_graphs["cancelled"] == True]
with tab3:
    csv_data.df_task_graphs[csv_data.df_task_graphs["cancelled"] == False]

st.write("### Tasks Table")
tab1, tab2, tab3 = st.tabs(["All", "Canceled", "Completed"])
with tab1:
    csv_data.df_tasks
with tab2:
    csv_data.df_tasks[csv_data.df_tasks["cancelled"] == True]
with tab3:
    csv_data.df_tasks[csv_data.df_tasks["cancelled"] == False]

plot_task_placement_timeline_chart(csv_data.df_worker_pools, csv_data.df_tasks)
########## Visualize DAG ##########

trace_data = get_original_trace_data(workload_file_path)

task_graph_id = st.text_input(
    "Enter the Task Graph ID you want to lookup (e.g. j_523717@1)"
)
if task_graph_id != "":
    visualize_task_graph(task_graph_id, csv_data.df_tasks, trace_data)
########## Visualize DAG ##########

plot_resource_utilization_timeline_chart(csv_data.worker_pool_stats)
