#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --partition=singlenode
#SBATCH --job-name=generation
#SBATCH --time=15:00:00
#SBATCH --export=NONE
#SBATCH --output=out_gen3d.txt

unset SLURM_EXPORT_ENV
module load python

venv gen
source gen/bin/activate

pip install python==3.12.7
pip install dill
pip install hebo

python generation_trigonometric.py multi_$1