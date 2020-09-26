#!/bin/bash
#SBATCH --time=0:10:00
# nodes=min-max
#SBATCH --nodes=12-12
#SBATCH --mem=32G 
# You need to load a compiler before openmpi.

module load gcc/8.3.0
module load openmpi/3.1.4 

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mpicxx -std=c++14 -O3 -fopenmp -o xuliny-seqalkway xuliny-seqalkway.cpp
mpirun xuliny-seqalkway < xulin_test.txt
