#!/bin/bash
# $1 is the directory where the logs of the experiments are stored.
# The rest of the arguments are passed directly to the analyze.py script.

RESULT_DIR=$1

if [[ -z $RESULT_DIR ]]; then
    echo "[x] ERROR: Please provide a directory where the results are being stored."
    exit 1
fi

CSV_FILES=""
CONF_FILES=""

for DIR in $(ls -d $RESULT_DIR/*); do
    EXPERIMENT=$(basename $DIR)
    CSV_FILE=$DIR/$EXPERIMENT.csv
    CONF_FILE=$DIR/$EXPERIMENT.conf

    if [[ ! -f $CSV_FILE ]] || [[ ! -f $CONF_FILE ]]; then
        echo "[x] Could not find config or CSV file for: $EXPERIMENT. Skipping."
    else
        echo "[x] Including experiment $EXPERIMENT."
        if [[ -z $CSV_FILES ]]; then
            CSV_FILES=$CSV_FILE
        else
            CSV_FILES="$CSV_FILES,$CSV_FILE"
        fi

        if [[ -z $CONF_FILES ]]; then
            CONF_FILES=$CONF_FILE
        else
            CONF_FILES="$CONF_FILES,$CONF_FILE"
        fi
    fi
done

echo "[x] Analyzing experiments."
python3 analyze.py --csv_files=$CSV_FILES --conf_files=$CONF_FILES --aggregate_stats ${@:2}
