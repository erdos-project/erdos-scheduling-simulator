#!/bin/bash

GPU_ID=0

MODEL_DIR=${1%/}
if [[ -z ${MODEL_DIR} ]]; then
    echo "[x] ERROR: Please provide the directory where the models are stored."
    exit 1
fi

OUTPUT_DIR=${2%/}
if [[ -z ${OUTPUT_DIR} ]]; then
    echo "[x] ERROR: Please provide the directory where the outputs are to be stored."
    exit 2
fi

for MODEL in $MODEL_DIR/*; do
    MODEL_NAME=${MODEL##*/}
    OUTPUT_PATH=$OUTPUT_DIR/$MODEL_NAME
    echo "[x] Profiling $MODEL_NAME from $MODEL."
    mkdir -p $OUTPUT_PATH
    CUDA_VISIBLE_DEVICES=$GPU_ID python run_inference.py --model_dir=$MODEL --image_path=./image.jpg 2>/dev/null 1>$OUTPUT_PATH/${MODEL_NAME}_RUN & 
    INF_PID=$!
    ./usage.sh $INF_PID >> $OUTPUT_PATH/${MODEL_NAME}_CPU &
    CPU_USAGE_PID=$!
    nvidia-smi --format=csv --query-gpu=timestamp,memory.used,utilization.gpu -lms 100 -i $GPU_ID >> $OUTPUT_PATH/${MODEL_NAME}_GPU &
    GPU_USAGE_PID=$!
    echo "[x] Waiting for inference job to finish."
    wait $INF_PID
    kill $CPU_USAGE_PID
    kill $GPU_USAGE_PID
    break
done
