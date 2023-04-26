#!/usr/bin/env bash
scheduler=${1:-Clockwork}

SLOS=(    10  25  50  100 250 500) # In ms
GOODPUTS=(400 500 575 675 725 800)

for i in ${!SLOS[@]}
do
  slo=${SLOS[$i]}
  goodput=${GOODPUTS[$i]}
  echo "Running Clockwork microbenchmark with $scheduler, SLO $slo, goodput $goodput"
  slo=$((slo * 1000)) # Convert to us
  request_rate=$(bc -l <<< "$goodput / (15 * 1000000)")
  filename="clockwork-microbenchmark-$scheduler-slo-$slo-goodput-$goodput"
  python3 main.py \
      --flagfile=configs/clockwork_resnet50_microbenchmark.conf \
      --override_poisson_arrival_rate=$request_rate \
      --override_slo=$slo \
      --log="$filename.log" \
      --csv="$filename.csv"
done
