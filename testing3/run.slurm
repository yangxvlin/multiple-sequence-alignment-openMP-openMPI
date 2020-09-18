#!/bin/bash
#SBATCH --output=run.out
#SBATCH --time=0:01:00

# <cpus-per-task> <ntasks-per-node> 
#  1  1-16


# <cpus-per-task> = 1
for i in {1..16}
do
   sbatch --output=12node-1-cpt-${i}-npn-physical.txt    --job-name=1-cpt-${i}-npn --cpus-per-task=1 --partition=physical    --ntasks-per-node=${i} "template.slurm"
done
