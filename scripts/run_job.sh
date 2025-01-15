#!/bin/sh

export PYTHONPATH=$PYTHONPATH:$(pwd)
export OMP_NUM_THREADS=10
srun --partition=compute --ntasks=10 --cpus-per-task=1 --mem-per-cpu=1GB --time=00:10:00 mpirun python3 auxiliary/playground2.py