#!/bin/bash
#SBATCH --error=seqalkway-xuliny-%j.err
#SBATCH --output=seqalkway-xuliny-%j.out
#SBATCH --partition=snowy
#SBATCH --time=0:40:00
# nodes=min-max
#SBATCH --nodes=12-12
#SBATCH --cpus-per-task=<cpus-per-task>
#SBATCH --ntasks-per-node=<ntasks-per-node> 
#SBATCH --mem=32G 
#SBATCH --job-name=seqalkway 
# You need to load a compiler before openmpi.

module load gcc/8.3.0
module load openmpi/3.1.4 


