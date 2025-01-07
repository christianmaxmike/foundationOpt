#!/bin/bash -l
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --job-name=generation
#SBATCH --time=12:00:00
#SBATCH --export=NONE
#SBATCH --output=out_generationSolver.txt

unset SLURM_EXPORT_ENV
module load python

venv gen
source gen/bin/activate

pip install python==3.12.7
pip install dill
pip install hebo

python generation.py