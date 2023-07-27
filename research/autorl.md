# Algorithms

# PPO
# DDPG

# Concepts

# Sample Efficiency
# Data Efficiency
# Exploration-Exploitation Tradeoff

# Theses

# Towards Truely Open-ended Reinforcement Learning
https://ora.ox.ac.uk/objects/uuid:e20bd6be-2a47-4e0f-bb6f-30df290e58c8

# Papers
# Automated Reinforcement Learning: An Overview
https://arxiv.org/pdf/2201.05000.pdf


# Sample Efficient Automated Deep Reinforcement Learning
https://arxiv.org/pdf/2009.01555.pdf

# Tuning Mixed Input Hyperparameters on the Fly for Efficient Population Based AutoRL
https://arxiv.org/pdf/2106.15883.pdf

## Introduction

# Gray-box Gaussian Processes for Automated Reinforcement Learning
https://openreview.net/pdf?id=rmoMvptXK7M

## Abstract
- novel BO technique
    - enrich gaussian processes with reward curve estimations 
    - employs generalized logistic functions

# AutoRL-Bench 1.0
https://openreview.net/pdf?id=RyAl60VhTcG

## Abstract 
- Tabular dataset of reward curves (tabular benchmark)
    - 22 enviornments
    - 5 RL algorithms

# Hyperparameters in Reinforcement Learning and How to Tune Them
https://arxiv.org/pdf/2306.01324.pdf

## Abstract

## Hyperparameter Optimization Problem

## Related Work

## Hyperparameter Landscape of RL

## Tradeoffs in Hyperparameter Optimization in Practice

## Recommendations & Best Practices

## Conclusion

# AutoRL: Survey and Open Problems
https://arxiv.org/pdf/2201.03916.pdf

## Preliminaries
- non-stationarity --> hyperparameters change over time while training
	- e.g. learning rates value decreases when converging on global maximum or minimum 

- Inner loop optimization
- Outer loop optimization

# FLAML
https://arxiv.org/pdf/1911.04706.pdf 

## Abstract
- optimizes for low computation resource

## Introduction
- use low computation cost to search for learner and hyperparameter choices
- system needs to select hyperparameters frequently on different training data



# Akshay

- B-WE --> bandwith enforcer
- Espresso ~ Google vs and Facebook 
- Microsoft ~ cost optimization Cascarrea 
	- Systems of ML
		 
- traffic engineering
- traffic
- self-driving

# Block Contextual MDPs for Continual Learning
https://openreview.net/pdf?id=IRa5JCfqEMA

## 1. Introduction

## 2. Related Work 

## 3. Background & Motivation

## 4. Generalization Properties of Lipschitz BC-MDPs

## 5. Zero-Shot Adaptation to Unknown Systems 

## 6. Experiments

## 7. Limitations

## 8. Discussion

# TorchBeast
https://arxiv.org/pdf/1910.03552.pdf 


# PCC Vivace: Online-Learning Congestion Control
https://www.usenix.org/system/files/conference/nsdi18/nsdi18-dong.pdf

## Abstract


# MVFST-RL: An Asynchronous RL Framework for Congestion Control with Delayed Actions

https://arxiv.org/pdf/1910.04054.pdf

## Abstract

- under-utilization of bandwidth
	- network sender waiting on policy
	- artifact of current gym environments

- improved framework
	- asynchronous RL agent
	- uses delayed actions and off-policy corrections

- tested on pantheon benchmark framework

## 1. Introduction 

- MV-FST = training framework
	- addresses issues with non-blockign RL agent

- Existing frameworks
	- Iroko 

## 5. Experiments

- Training
	- 40 parallel actors 

- Results 
	- 

## 6. Conclusion and Future Work
- Challenging to deploy RL to datacenters 



# Deep Traffic 
https://arxiv.org/pdf/1801.02805.pdf

## Abstract

## 1. Introduction 

## 2. Related Work 

## 3. DeepTraffic Simulation and Competition

## 3.1 Simulation 

## 4. Network Performance and Exploration

## 5. Conclusion
- Human-based hyperparameter tuning

# Model Assertions for Debugging Machine Learning
https://ddkang.github.io/papers/2018/omg-nips-ws.pdf



# Distributed RL scheme for network routing
http://www.cs.cmu.edu/~jab/cv/pubs/littman.q-routing.pdf 


# Flow Types
## Elephant Flow
https://en.wikipedia.org/wiki/Elephant_flow

## Mice Flow 
https://en.wikipedia.org/wiki/Mouse_flow

# Deeproute

https://dl.acm.org/doi/abs/10.1007/978-3-030-45778-5_20

## 1. Introduction

## 2. Background 

## 3. Related Work 

## 4. Design of Deep Route

- Define network topology
	- unidirectional links 
		- varying bandwidth capacity and latency
	- flows
		- arrive at timestep t = 1
		- assigned path 

### 4.1 Deep Route in Simulation Model 
- State Space
- Action Space 


# HyperSched: Dynamic Resource Reallocation for Model Development on a Deadline

## 1. Introduction

## 2. Background and Motivation

## 3. Problem

## 4. Hypersched

## 5. System Implementation

## 6. Evaluation

## 7. Discussion

## 8. Conclusion

# Google's Research Philosophy
https://research.google/philosophy/ 

# NERSC Distributed Training
https://docs.nersc.gov/machinelearning/distributed-training/

# Hyperparameter Optimization: Foundation, Algorithms, Best Practices
https://arxiv.org/pdf/2107.05847.pdf


## 1. Introduction

## 8. Conclusion and Open Challenges


# Parallel Training of Deep Networks with Local Updates
https://arxiv.org/pdf/2012.03837.pdf 

# Stefan Wild 
https://wildsm.github.io/ 

# DeepHyper
https://ieeexplore.ieee.org/document/8638041

## 1. Introduction

- Key Contributions
	- collection of portable DNN hyperparameter search problem instances
	- generic interface between search methods and parallel task execution engines
	- scalable, asynchronous model-based search method
	- comparison to batch methods
	- workflow system at HPC scale with fault tolerance and parallel efficiency

## 2. Hyperparameter Search Problem 

## 3. Deephyper package
- Parts
	- Benchmarks
	- Search 
	- Evaluation

## 4. Experimental Results 

- Scaling DeepHyper to Theta Supercomputer
- Scaling AMBS to Cooley GPU Cluster

## 5. Related Work

- Includes worflow system integration
	- Hides complexities of HPC platforms 
	- Not found in Ray Tune

# On Scale-out DL Training for Cloud and HPC
https://mlsys.org/Conferences/2019/doc/2018/64.pdf

# Pytorch on HPC
- https://researchcomputing.princeton.edu/support/knowledge-base/pytorch 

# Embarrasingly Parallel Problems
- https://en.wikipedia.org/wiki/Embarrassingly_parallel
- https://en.wikipedia.org/wiki/Parallel_algorithm#Parallelizability

# Top 500
https://top500.org/ 

# Cloud vs. Grid
http://cloudscaling.com/blog/cloud-computing/grid-cloud-hpc-whats-the-diff/ 

- Scalability and Performance
    - Orthogonal 
- HSC vs. HPC
    - High Scalability Computing
    - High Performance Computing


# HPC and the Cloud 
https://d1.awsstatic.com/hpc/AWS%20NVIDIA%20and%20Hyperion%20Research%20Technical%20Spotlight.pdf 


# Cloud vs. HPC
https://wangzhezhe.github.io/2021/06/27/HPC-vs-CloudComputing/



# Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer 
https://openreview.net/pdf?id=Bx6qKuBM2AD

# Container-based workflow for Distributed Training of DL Algos in HPC Cluster
https://arxiv.org/pdf/2208.02498.pdf 

## 2.3 Containerization technologies 

### 2.3.1 Docker

- Implementation
    - Linux kernel
        - namespaces -> sets isolation layer
        - cgroups -> limits hardware usage 
- Advantages
    - docker hub connection
- Disadvantages
    - Requires root level process 

### 2.3.2 Singularlity 
- Advantages
    - non-root level execution

### 2.3.3. udocker 
- Advantages
	- does not require administrative privileges

### 2.3.2 Singularity

# 4. Methodology and Workflow

## 4.2 OpenMPI and Horovod Integration


# Deep RL Agent for Scheduling in HPC
https://arxiv.org/pdf/2102.06243.pdf



# Hyperparameter Tuning for Deep RL Applications
- https://arxiv.org/pdf/2201.11182.pdf

## 1. Intro 
- Existing Hyperparameter Search Libraries
	- Ray Tune 
	- Hyperband 
	- DeepHyper 
	- Optuna 

- Genetic Algorithm Success
	- Find optimial settings of 3-layer CNN 

- Goal
	- Evolutionary algorithms not employed for hyper parameter tuning in deep RL 
		- Modify the hyperparameters via evolutional methods 

- Pitfalls
	- Infinite compute cycles for trial and error episodes in genetic algorithms
	- DeepRL is still in infancy 

- HPS-RL
	- scalable deployment library
	- executes a genetic algorithm 

- Contributions
	- Automated multi-objective search using genetic algorithms
	- Leverage multiple threads and parallel processing to improve search time 


## 2. Identifying Hyperparameters
- Existing Methods 
	- Grid search 

## 3. Bayesian Optimization for Hyperparameter Search
- Bayesian Optimization
	- hyperparameter search on non-convex search spaces

- Fitness function approximated using gaussian kernels 
	- Use a function to determine next best point
		- expected maximization
		- maximum probability of maximization
		- upper confidence bound
		- entropy search 
	- Perform gradient based optimization

## 4. HPS-RL 
- Multi-objective optimization problem 

- Genetic Algorithm 
	- define genes 
	- define selection process 
	- evaluate genes in trials 
	- perform cross over and mutation

- Example
	- Genes = Hyperparameters
		- gamma, alpha, number of neurons, activiation functions, elpsion, number of layers
	- Evaluation = reward from 100 steps in gym
	- Selection = random distribution across sucessfull parents 
	- Cross Over = swap hyperparameters 
	- Mutation = replace hyperparameter with random value 
	- Code
		- https://github.com/esnet/hps-rl/blob/main/searchmethods/modularGA.py 
		- https://github.com/esnet/hps-rl/blob/main/searchmethods/GA4RL.py 
		- https://github.com/esnet/hps-rl/blob/main/main_search.py 

- Challenges
	- Agents trained on different RL models (e.g. DDPG vs. ACKTR)
		- have different types and number of hyperparameters 


## 5. Software
- 3 python packages
	- collection of genetic algorithm functions
	- benchmarks of gym environments
	- benchmark of deep RL algorithms with optimization functions

- Implementation
	- Submit jobs from single head node 

- Hardware
	- Intel i7-9750H CPU 
	- Nvidia GeForce RTX 2070 

- Packages
	- mpi4py

- Optimization Methods
	- Conjugate Gradient
		- Fast convergence but poor performance
	- Broyden-Fletcher-Golfarb-Shanno (BFGS)
		- Uses quasi-newtons methods to solve unconstrained optimization problems
	- Levenberg-Marquardt
		- Solves sum-of-square of non-linear functions
		- Behaves initially like gradient descent, then like the gauss-newton method

## 6. Experimental Results 

- Use three environments
	- Cartpole
	- Lunar Landing
	- Autonomous laser control 

- Better fitness is few episodes of training achieved in all environments

- Performing on multiple GPUs increase total time, which can be improved via parallelism in the future


## 7. Related Work 

- RL Applications
	- Integrate multiple RL algorithms to solve cartpole
- Hyperparameter Search 
	- Bayesian Optimization + Bandit Methods > Hyperband
- Additional Hyperparameter Tuning Libraries 
	- Hyperopt, Polyaxon

- Bayesian optimization methods performs poorly with large hyperparamter spaces

## 8. Conclusion

- Move towards genetic algorithms, and avoid bayesian optimization 
- Bayesian optimization (BO) requires large number of episodes to receive optimized results 
- Genetic algorithms (GA) provide more exploration, and therefore better optimization than BO 
- GA can be parallelized, while BO cannot 
- BO works well on continous hyperparameter surfaces, but early RL learning benefits more from GA's randomness

## Unresolved Questions
- Significance of figures 5a and 5b

## Research Ideas
- Incorporate hyperparameter search stops
- Better understand current hyperparameter tuning methods in deep RL
	- Explore population-based training methods
- Can I run CUDA on this? 

## References
- https://github.com/esnet/hps-rl 
	- Codebase 
- https://en.wikipedia.org/wiki/Slurm_Workload_Manager
	- Slurm
- https://docs.ray.io/en/latest/tune/index.html 
	- Ray Tune 
- https://keras.io/api/keras_tuner/tuners/hyperband/
	- Hyperband 
- https://optuna.org/
	- Optuna 
- https://lilianweng.github.io/posts/2019-09-05-evolution-strategies/#extension-ea-in-deep-learning
	- Lil'Log Blog
- https://docs.ray.io/en/latest/tune/examples/pbt_guide.html
	- Population Based Training

# Ray Tune: Research Platform for Distributed Model Selection and Training 

https://arxiv.org/pdf/1807.05118.pdf

## Abstract

## 1. Introduction 

## 2. Related Work 

- Existing open source systems 
	- HyperOpt, Spearmint, HPOLib, TuPAQ, MLBase
		- Implement random search and tree of parzen estimators 
		- Treat full triat execution as atomic unit 
			- Doesn't allow for intermediate control of train execution 
		- Doesn't support Hyperband 
	- Google Vizier
		- Closed-source service 
	- Mistique 
		- Focuses on debugging and memory footprint minimization 
	- Auto-SKLearn and Auto-WEKA
		- Focuse on execution level, not algorithm level

## 3. Requirement for API Generality

## 4. Tune API 
- User API
	- for users training models
	- modify hyperparameters mid training 
		- cooperative control model 

- Scheduling API
	- for researchers improving model search process

## 5. Implementation

- Existing distributed frameworks
	- MPI
	- Spark
	- Ray

- Tune uses trial schedulers
	- Best supported by Ray
		- Two-level scheduler
			- Spills over to cluster when local resources exhausted 
	- 

## 6. Conclusions and Future Work 

# System for Massively Parallel Hyperparameter Tuning [TODO]
https://proceedings.mlsys.org/paper/2020/file/f4b9ec30ad9f68f89b29639786cb62ef-Paper.pdf

## Abstract

## 1. Introduction 

- Asynchronous Successive Halving Algorithm (ASHA)
	- Inspired by SHA
		- Allocates more resources to promising configurations

- Partial Training Methods
	- SHA
	- Fabolas
	- Population Based Training
	- BOHB 

- System Design Decisions
	- (1) streamlining user interface to enhance usability 
	- (2) autoscaling parallel training
		- balance lower latency for individual training v.s high throughout total configuration evaluation 
	- (3) scheduling ML Jobs to optimize multi-tenant cluster utilization 
	- (4) tracking parallel hyperparameter tuning for reproducibity 
	

## 2. Related Work

## 3. ASHA Algorithm

- SHA Variants
	- Infinite Horizon
	- Finite Horizon 

## 4. Empirical Evaluation

## 5. Productionizing ASHA

## 6. Conclusion


# Hyperband
https://arxiv.org/pdf/1603.06560.pdf

## Abstract

- hyperparameter optimization
	- pure-exploration non-stochastic infinite-armed bandit problem 

## 1. Introduction 
- existing methods
	- random search 
	- grid search
	- bayesian optimizaiton 

- configuation evaluation 
	- allocating more resources for promising configs
	- 

## 2. Related Work 

## 3. Hyperband Algorithm

## 4. Hyperparameter Optimization Experiments

## 5. Theory 

# Deep Hyper [TODO]
https://ieeexplore.ieee.org/document/8638041
