#!/bin/sh

srun --partition=compute --ntasks=1 --cpus-per-task=1 --mem-per-cpu=1GB --time=00:01:00 python3 auxiliary/ga_playground.py


