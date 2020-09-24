# multiple-sequence-alignment-openMP-openMPI
COMP90025 - Parallel and Multicore Computing - 2020S2 - Assignment2B

## how to submit
- ``` /data/gpfs/projects/punim0520/pmc/pub/bin/handin seqalkway xuliny-seqalkway.cpp <cpus-per-task> <ntasks-per-node>  ```
    - ``` <cpus-per-task> ```
      - cores per process
    - ``` <ntasks-per-node> ```
      - processes per node
    - product of both <= 16
    - ```
      For example, you may have:
      CPUs per task = 4
      Tasks per node = 4
      Nodes = 12

      That means you are using 4×4 cores per node, so 4×4×12 cores total.
      ```
## how to run on wsl
- ``` mpicxx -std=c++14 -fopenmp -o a test.cpp -O3 ```
- ``` mpirun -np 4 a < mseq.dat > mseq.out ```
- ``` mpirun -np 4 a < mseq1.dat > parallel1_mseq1.out ```
## how to test
- ``` cd testing ```
- ``` sbatch run.slurm ```
