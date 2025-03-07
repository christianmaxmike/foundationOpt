#!/bin/bash -l
##SBATCH --nodes=1
#SBATCH --partition=singlenode
##SBATCH --gres=gpu:a100:1
##SBATCH --partition=a100
##SBATCH --gres=gpu:h100:1
##SBATCH --partition=h100
#SBATCH --job-name=foundOpt
#SBATCH --time=23:59:00
#SBATCH --export=NONE
#SBATCH --output=out.txt

unset SLURM_EXPORT_ENV
module load python

#export http_proxy=http://proxy:80
#export https_proxy=http://proxy:80
#export HTTP_PROXY=http://proxy:80
#export HTTPS_PROXY=http://proxy:80
export http_proxy=http://proxy.rrze.uni-erlangen.de:80
export https_proxy=http://proxy.rrze.uni-erlangen.de:80
export HTTP_PROXY=http://proxy.rrze.uni-erlangen.de:80
export HTTPS_PROXY=http://proxy.rrze.uni-erlangen.de:80

source $HOME/venvs/foundOpt/bin/activate

#python generation.py --id $1
#python SineCosGenerator.py --id $1
python generate_data.py --type "SimpleTrig" --id $1 --minComponents 1 --maxComponents 2
#python train_model_main.py 