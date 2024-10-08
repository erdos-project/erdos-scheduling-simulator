#!/bin/bash

# cancelled() method takes a CSV as input and outputs the number of
# TaskGraphs that have been cancelled.
cancelled() {
    if [[ "$1" == *.csv ]]; then
        cat $1 | grep "TASK_CANCEL" | awk -F, '{print $6}' | sort | uniq | wc -l
    else
	echo "cancelled() takes a CSV file as input."
    fi
}

# released() method takes a CSV as input and outputs the number of
# TaskGraphs that have been released.
released() {
    if [[ "$1" == *.csv ]]; then
        cat $1 | grep "TASK_GRAPH_RELEASE" | wc -l
    else
	echo "released() takes a CSV file as input."
    fi
}

# finished() method takes a CSV as input and outputs the number of
# TaskGraphs that have been released.
finished() {
    if [[ "$1" == *.csv ]]; then
	cat $1 | grep "TASK_GRAPH_FINISHED" | wc -l
    else
	echo "finished() takes a CSV file as input."
    fi
}
