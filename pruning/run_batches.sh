#!/bin/sh
# export KMP_BLOCKTIME=1
# export OMP_NUM_THREADS=48
# export KMP_SETTINGS=TRUE
# export KMP_AFFINITY=granularity=fine,compact

working_file=working3.sh

echo > $working_file


for i in $(seq 1 8)
do
#   echo python MNIST_PruneDBM_HeuristicFI.py 20 50 6 2 $i >> working.sh
   echo python MNIST_PruneDBM_HeuristicFI.py 10 25 12 $i 10 --no-constant >> $working_file 
#  echo python MNIST_PruneDBM_VarianceFI.py 10 25 12 $i 10 --no-constant >> working2.sh
##  echo python MNIST_PruneDBM_VarianceFI.py 5 25 12 $i 10 --no-constant >> working.sh
#   echo python MNIST_PruneDBM_VarianceFI_allzero.py 10 25 6 $i 10 >> working.sh
#   echo python MNIST_PruneDBM_W.py 10 25 12 $i 10 >> working3.sh
#   echo python MNIST_PruneDBM_AntiFI.py 10 25 12 $i 10 >> working3.sh
#   echo python MNIST_PruneDBM_HeuristicFI.py 10 20 12 2 $i >> working.sh
done

parallel -j 2 --tmuxpane --fg < $working_file

# rm working.sh
