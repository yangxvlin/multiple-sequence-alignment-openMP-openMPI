#!/bin/bash
#SBATCH --output=test1.out
#SBATCH --time=0:01:00

# <cpus-per-task> <ntasks-per-node> 
#  1  1-16
#  2  1-8
#  3  1-5
#  4  1-4
#  5  1-3
#  6  1-2
#  7  1-2
#  8  1-2
#  9  1
# 10  1
# ... 
# 16  1

# <cpus-per-task> = 1
for i in {9..11}
do
   sbatch --output=12node-1-cpt-${i}-npn-snowy.txt    --job-name=1-cpt-${i}-npn --cpus-per-task=1 --partition=snowy    --ntasks-per-node=${i} "template.slurm"
done
