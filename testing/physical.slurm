#!/bin/bash
#SBATCH --output=run.out
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
array=( 1 2 4 8 16 )
for i in "${array[@]}"
do
   sbatch --output=12node-1-cpt-${i}-npn-physical.txt    --job-name=physical --cpus-per-task=1 --partition=physical    --ntasks-per-node=${i} "template.slurm"
done

# <cpus-per-task> = 2
array=( 1 2 4 8 )
for i in "${array[@]}"
do
   sbatch --output=12node-2-cpt-${i}-npn-physical.txt    --job-name=physical --cpus-per-task=2 --partition=physical    --ntasks-per-node=${i} "template.slurm"
done

# <cpus-per-task> = 4
array=( 1 2 4 )
for i in "${array[@]}"
do
   sbatch --output=12node-4-cpt-${i}-npn-physical.txt    --job-name=physical --cpus-per-task=4 --partition=physical    --ntasks-per-node=${i} "template.slurm"
done


# <cpus-per-task> = 6-8
array=( 6 8 )
for i in {1..2}
do
    for j in "${array[@]}"
    do
        sbatch --output=12node-${j}-cpt-${i}-npn-physical.txt    --job-name=physical}-npn --cpus-per-task=${j} --partition=physical    --ntasks-per-node=${i} "template.slurm"
    done
done

# <cpus-per-task> = 9-16
array=( 9 10 12 14 16 )
for j in "${array[@]}"
do
    sbatch --output=12node-${j}-cpt-1-npn-physical.txt    --job-name=physical --cpus-per-task=${j} --partition=physical    --ntasks-per-node=1 "template.slurm"
done
