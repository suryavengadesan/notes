# Slurm

## Official Guide
https://my.nersc.gov/script_generator.php 

- Man Pages
    - https://slurm.schedmd.com/man_index.html 
- Documentation
    - https://slurm.schedmd.com/ 

## NERSC Guide
https://docs.nersc.gov/jobs/

- Jobs 
    - Set nodes for a specific amount of time 
        - interactive vs. batch 

- Parallel Work
    - Job sets
        - Set of tasks 

- Login Node
    - Editing, compiling, preparing jobs 

- Submitting jobs 
    - sbatch
        - submit job for later execution
        - performs quota enforcement by checking file system
    - salloc 
        - allocate resources in realtime (interactive batch job)
            - executes srun to run parallel tasks 
    - srun
        - submit current job execution 
            - initiate job steps in real time
    
- Minimum job requirements 
    - number of nodes
    - time 
    - type of nodes (constraint)
    - quality of service (QOS)
    - number of GPUs (only PerlMutter)


- Job Script Generation 
    - https://my.nersc.gov/script_generator.php 

- Queue and QOS selection
    - https://docs.nersc.gov/jobs/policy/#selecting-a-queue
    - Quality of service <=> Queue
        - regular, debug, etc. 
# Ray Tune

- 6 key concepts 
    - define hyperparameters 
    - define search space
    - define trainable which specified objective
    - select search algorithm 
    - select scheduler to stop searches
- initialize Tuner
    - specify config, trainable, searc algo
    - runs experiments and create traials
        - investigate using analyses 

## Ray Tune Trainables

## Tune Search Spaces

## Tune Trials

## Tune Search Algorithms

## Tune Schedulers

## Rune Run Analyses

