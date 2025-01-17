#!/bin/sh

for i in {0..99}
do
  sbatch --job-name "gen_$i" --output "out_multi_$i.txt" ./slurm_script.sh "$i"
done
