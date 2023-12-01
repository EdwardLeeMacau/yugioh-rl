#!/bin/bash

# Script to repeat 'python main.py' 10000 times
# Prints the current iteration number to the console
# Prints the time taken since the script started to the console

MAX=10000
START=$(date +"%s")

secs_to_human() {
    if [[ -z ${1} || ${1} -lt 60 ]] ;then
        min=0 ; secs="${1}"
    else
        time_mins=$(echo "scale=2; ${1}/60" | bc)
        min=$(echo ${time_mins} | cut -d'.' -f1)
        secs="0.$(echo ${time_mins} | cut -d'.' -f2)"
        secs=$(echo ${secs}*60|bc|awk '{print int($1+0.5)}')
    fi
    echo "Time Elapsed : ${min} minutes and ${secs} seconds."
}

for i in $(seq 1 $MAX);
do
    echo "$i / $MAX";

    secs_to_human $(($(date +%s) - $START));
    python main.py
done
