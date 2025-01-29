#!/bin/sh
#
#SBATCH --partition=compute
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=10:00:00


export PYTHONPATH=$PYTHONPATH:$(pwd)
export OMP_NUM_THREADS=10
srun --partition=compute --ntasks=10 --cpus-per-task=1 --mem-per-cpu=4GB --time=10:00:00 python3 auxiliary/playground2.py