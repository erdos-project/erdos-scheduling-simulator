#!/usr/bin/env bash
scheduler=${1:-Clockwork}

SLOS=(    10  25  50  100 250 500) # In ms
GOODPUTS=(400 500 575 675 725 800)
PARALLEL_FACTOR=8

for i in ${!SLOS[@]}
do
  slo=${SLOS[$i]}
  goodput=${GOODPUTS[$i]}
  echo "Running Clockwork microbenchmark with $scheduler, SLO $slo, goodput $goodput"
  request_rate=$(bc -l <<< "$goodput / (15 * 1000000)")
  filename="clockwork-microbenchmark-$scheduler-slo-$slo-goodput-$goodput"
  slo=$((slo * 1000)) # Convert to us
  python3 main.py \
      --flagfile=configs/clockwork_resnet50_microbenchmark.conf \
      --override_poisson_arrival_rate=$request_rate \
      --override_slo=$slo \
      --log="$filename.log" \
      --csv="$filename.csv" &
  if [[ $(jobs -r -p | wc -l) -ge $PARALLEL_FACTOR ]]; then
    echo "[x] Waiting for a job to terminate because $PARALLEL_FACTOR jobs are running."
    wait -n 
  fi
done
wait
echo "[x] Finished executing all experiments."
