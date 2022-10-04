#!/bin/bash
# $1 is the directory where the results are stored.

RESULTS_DIR=${1%/}
if [[ -z ${RESULTS_DIR} ]]; then
    echo "[x] ERROR: Please provide the directory where the results are stored."
    exit 1
fi

for MODEL in $RESULTS_DIR/*; do
    MODEL_NAME=${MODEL##*/}
    echo "[x] Results for ${MODEL_NAME}:"
    python analyze.py --run_file=$MODEL/${MODEL_NAME}_RUN --cpu_file=$MODEL/${MODEL_NAME}_CPU --gpu_file=$MODEL/${MODEL_NAME}_GPU
done
