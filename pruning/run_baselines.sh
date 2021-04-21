#!/bin/sh
export KMP_BLOCKTIME=1
export OMP_NUM_THREADS=10
export KMP_SETTINGS=TRUE
export KMP_AFFINITY=granularity=fine

echo > working.sh

for i in $(seq 1 10)
do
  echo python MNIST_Baselines.py $i >> working.sh
done

parallel -j 2 --tmuxpane --fg < working.sh
rm working.sh