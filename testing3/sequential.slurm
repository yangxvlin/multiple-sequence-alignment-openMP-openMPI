#!/bin/bash
#SBATCH --time=0:39:00
# nodes=min-max
#SBATCH --nodes=12-12
#SBATCH --mem=32G 
--output=sequential.txt    
--job-name=sequential
--cpus-per-task=1 
--ntasks-per-node=1
--partition=physical    
# You need to load a compiler before openmpi.

module load gcc/8.3.0
module load openmpi/3.1.4 

mpicxx -std=c++14 -fopenmp -o b seqalign-mpi-skeleton.cpp -O3
mpirun b < mseq-big13-example.txt
