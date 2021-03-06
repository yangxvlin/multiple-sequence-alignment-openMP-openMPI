#!/bin/bash
#SBATCH --output=run.out
#SBATCH --time=0:01:00

# <cpus-per-task> <ntasks-per-node> 
#  1  1-16


# <cpus-per-task> = 1
for i in {1..16}
do
   sbatch --output=12node-1-cpt-${i}-npn-physical.out  --job-name=1-cpt-${i}-npn --cpus-per-task=1 --partition=physical    --ntasks-per-node=${i} "template.slurm"
done

# sbatch --output=12node-1-cpt-1-npn-physical.out  --job-name=1-cpt-1-npn --cpus-per-task=1 --partition=physical    --ntasks-per-node=1 "template.slurm"
# sbatch --output=12node-2-cpt-1-npn-physical.out  --job-name=2-cpt-1-npn --cpus-per-task=2 --partition=physical    --ntasks-per-node=1 "template.slurm"
# sbatch --output=12node-4-cpt-1-npn-physical.out  --job-name=4-cpt-1-npn --cpus-per-task=4 --partition=physical    --ntasks-per-node=1 "template.slurm"
# sbatch --output=12node-6-cpt-1-npn-physical.out  --job-name=6-cpt-1-npn --cpus-per-task=6 --partition=physical    --ntasks-per-node=1 "template.slurm"
# sbatch --output=12node-8-cpt-1-npn-physical.out  --job-name=8-cpt-1-npn --cpus-per-task=8 --partition=physical    --ntasks-per-node=1 "template.slurm"
# sbatch --output=12node-9-cpt-1-npn-physical.out  --job-name=9-cpt-1-npn --cpus-per-task=9 --partition=physical    --ntasks-per-node=1 "template.slurm"
# sbatch --output=12node-10-cpt-1-npn-physical.out  --job-name=10-cpt-1-npn --cpus-per-task=10 --partition=physical    --ntasks-per-node=1 "template.slurm"
# sbatch --output=12node-12-cpt-1-npn-physical.out  --job-name=12-cpt-1-npn --cpus-per-task=12 --partition=physical    --ntasks-per-node=1 "template.slurm"
# sbatch --output=12node-14-cpt-1-npn-physical.out  --job-name=14-cpt-1-npn --cpus-per-task=14 --partition=physical    --ntasks-per-node=1 "template.slurm"
# sbatch --output=12node-15-cpt-1-npn-physical.out  --job-name=15-cpt-1-npn --cpus-per-task=15 --partition=physical    --ntasks-per-node=1 "template.slurm"
# sbatch --output=12node-16-cpt-1-npn-physical.out  --job-name=16-cpt-1-npn --cpus-per-task=16 --partition=physical    --ntasks-per-node=1 "template.slurm"

# sbatch --output=10node-2-cpt-1-npn-physical.out  --job-name=2-cpt-1-npn --cpus-per-task=2 --partition=physical    --ntasks-per-node=1 --nodes=10 "template.slurm"

# sbatch --output=12node-1-cpt-1-npn-snowy.out  --job-name=1-cpt-1-npn --cpus-per-task=1 --partition=snowy    --ntasks-per-node=1 "template.slurm"
# sbatch --output=12node-2-cpt-1-npn-snowy.out  --job-name=2-cpt-1-npn --cpus-per-task=2 --partition=snowy    --ntasks-per-node=1 "template.slurm"
# sbatch --output=12node-4-cpt-1-npn-snowy.out  --job-name=4-cpt-1-npn --cpus-per-task=4 --partition=snowy    --ntasks-per-node=1 "template.slurm"
# sbatch --output=12node-6-cpt-1-npn-snowy.out  --job-name=6-cpt-1-npn --cpus-per-task=6 --partition=snowy    --ntasks-per-node=1 "template.slurm"
# sbatch --output=12node-8-cpt-1-npn-snowy.out  --job-name=8-cpt-1-npn --cpus-per-task=8 --partition=snowy    --ntasks-per-node=1 "template.slurm"
# sbatch --output=12node-9-cpt-1-npn-snowy.out  --job-name=9-cpt-1-npn --cpus-per-task=9 --partition=snowy    --ntasks-per-node=1 "template.slurm"
# sbatch --output=12node-10-cpt-1-npn-snowy.out  --job-name=10-cpt-1-npn --cpus-per-task=10 --partition=snowy    --ntasks-per-node=1 "template.slurm"
# sbatch --output=12node-12-cpt-1-npn-snowy.out  --job-name=12-cpt-1-npn --cpus-per-task=12 --partition=snowy    --ntasks-per-node=1 "template.slurm"
# sbatch --output=12node-14-cpt-1-npn-snowy.out  --job-name=14-cpt-1-npn --cpus-per-task=14 --partition=snowy    --ntasks-per-node=1 "template.slurm"
# sbatch --output=12node-15-cpt-1-npn-snowy.out  --job-name=15-cpt-1-npn --cpus-per-task=15 --partition=snowy    --ntasks-per-node=1 "template.slurm"
# sbatch --output=12node-16-cpt-1-npn-snowy.out  --job-name=16-cpt-1-npn --cpus-per-task=16 --partition=snowy    --ntasks-per-node=1 "template.slurm"

# sbatch --output=8node-4-cpt-1-npn-snowy.out  --job-name=4-cpt-1-npn --cpus-per-task=4 --partition=snowy    --ntasks-per-node=1 --nodes=8 "template.slurm"
# sbatch --output=10node-4-cpt-1-npn-snowy.out  --job-name=4-cpt-1-npn --cpus-per-task=4 --partition=snowy    --ntasks-per-node=1 --nodes=10 "template.slurm"

# sbatch --output=1node-16-cpt-1-npn-snowy2.out  --job-name=16-cpt-1-npn --cpus-per-task=16 --partition=snowy    --ntasks-per-node=1 --nodes=1 --time=0:50:00 "template.slurm"
