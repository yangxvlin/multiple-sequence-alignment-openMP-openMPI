#!/bin/bash
#SBATCH --output=run.out

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
for i in {1..16}
do
   sbatch --output=12node-1-cpt-${i}-npn-snowy.txt    --job-name=1-cpt-${i}-npn --cpus-per-task=1 --partition=snowy    --ntasks-per-node=${i} "template.slurm"
done

# <cpus-per-task> = 2
for i in {1..8}
do
   sbatch --output=12node-2-cpt-${i}-npn-snowy.txt    --job-name=2-cpt-${i}-npn --cpus-per-task=2 --partition=snowy    --ntasks-per-node=${i} "template.slurm"
done

# <cpus-per-task> = 3
for i in {1..5}
do
   sbatch --output=12node-3-cpt-${i}-npn-snowy.txt    --job-name=3-cpt-${i}-npn --cpus-per-task=3 --partition=snowy    --ntasks-per-node=${i} "template.slurm"
done

# <cpus-per-task> = 4
for i in {1..4}
do
   sbatch --output=12node-4-cpt-${i}-npn-snowy.txt    --job-name=4-cpt-${i}-npn --cpus-per-task=4 --partition=snowy    --ntasks-per-node=${i} "template.slurm"
done

# <cpus-per-task> = 5
for i in {1..5}
do
   sbatch --output=12node-5-cpt-${i}-npn-snowy.txt    --job-name=5-cpt-${i}-npn --cpus-per-task=5 --partition=snowy    --ntasks-per-node=${i} "template.slurm"
done

# <cpus-per-task> = 6-8
for i in {1..2}
do
    for j in {6..8}
    do
        sbatch --output=12node-${j}-cpt-${i}-npn-snowy.txt    --job-name=${j}-cpt-${i}-npn --cpus-per-task=${j} --partition=snowy    --ntasks-per-node=${i} "template.slurm"
    done
done

# <cpus-per-task> = 9-16
for j in {9..16}
do
    sbatch --output=12node-${j}-cpt-1-npn-snowy.txt    --job-name=${j}-cpt-1-npn --cpus-per-task=${j} --partition=snowy    --ntasks-per-node=1 "template.slurm"
done
