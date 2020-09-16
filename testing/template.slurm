#!/bin/bash
#SBATCH --time=0:39:00
# nodes=min-max
#SBATCH --nodes=12-12
#SBATCH --mem=32G 
#SBATCH =seqalkway 
# You need to load a compiler before openmpi.

module load gcc/8.3.0
module load openmpi/3.1.4 

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mpicxx -std=c++14 -fopenmp -o a test.cpp -O3
mpirun a < xulin.dat
