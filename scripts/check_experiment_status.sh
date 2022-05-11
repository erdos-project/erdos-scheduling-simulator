#!/bin/bash
# $1 is the directory where the logs of the experiments are being stored.

RESULT_DIR=$1

if [[ -z ${RESULT_DIR} ]]; then
    echo "[x] ERROR: Please provide a directory where the results are being stored."
    exit 1
fi

OPTIONS=$2
SKIP_FINISHED=false
if [[ OPTIONS == "--only-running" ]]; then
    SKIP_FINISHED=true
fi

for DIR in $(ls -d $RESULT_DIR/*); do
    EXPERIMENT=$(basename $DIR)
    if [[ ! -f ${DIR}/task_placement_delay.png ]]; then
	echo -e "[x] Experiment: \e[33m $EXPERIMENT \e[0m"
        echo -e "    STATUS: \e[31m RUNNING \e[0m"
	RECENT_UPDATE=$(tail -n 1 ${DIR}/${EXPERIMENT}.csv)
	IFS=', ' read -r -a CSV_SPLIT <<< "${RECENT_UPDATE}"
	echo -e "      TIME: \e[31m ${CSV_SPLIT[0]} \e[0m"
    elif [ "$SKIP_FINISHED" = true ]; then
	echo -e "[x] Experiment: \e[33m $EXPERIMENT \e[0m"
        echo -e "    STATUS: \e[32m FINISHED \e[0m"
    fi
done
