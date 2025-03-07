#!/bin/sh

for i in {0..63}
do
  sbatch --job-name "gen_$i" --output "datasets/single/1D_triv/out_$i.txt" ./slurm_script.sh "$i"
done
