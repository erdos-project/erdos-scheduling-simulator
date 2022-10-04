#!/bin/bash
#$1 is the ID of the process.
#Output is a space-seperated file of Timestamp,CPU%,MEM%.

PROC_ID=$1

adddate() {
    while IFS="\n" read -r line; do
	cpu_mem=`awk -F' ' '{print $9,$10}' <<< "$line"`
        echo "$(date '+%Y/%m/%d %T.%N') $cpu_mem"
    done
}

top -p $PROC_ID -d 0.1 -b | grep --line-buffered $PROC_ID | adddate
