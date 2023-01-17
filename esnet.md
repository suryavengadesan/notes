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

- Missing Link
	- Evolutionary algorithms not employed for hyper parameter tuning in deep RL 

- Possible Reason 
	- Lack of 

## 2. Hyperparameters

## 3. Bayesian Optimization for Hyperparameter Search

- H Function
	- expected maximizaiton
	- maximum probability of maximization
	- upper confidence bound
	- entropy search 

- Bayesian Optimization Process


## 4. HPS-RL 

- Multi-objective optimization problem 

- Genetic Algorithm 
	- define genes 
	- define selection process 
	- evaluate genes in trials 
	- perform cross over and mutation

- Example
	- Genes = Hyperparameters
		- gamma, alpha, number of neurons 
		- activiation functions, elpsion, number of layers
	- Evaluation = reward from 100 steps in gym
	- Selection = random distribution across sucessfull parents 
	- Cross Over = swap hyperparameters 
	- Mutation = replace hyperparameter with random value 

- Challenges
	- Agents trained on different RL models (e.g. DDPG vs. ACKTR)
		- have different types and number of hyperparameters 


## 5. Software

- 3 parts 
	- collection of genetic algorithm functions
	- benchmark gym environments
	- benchmark deep RL algorithms 

- Implementation
	- Submit jobs from single head node 

- Hardware
	- Intel i7-9750H CPU 
	- Nvidia GeForce RTX 2070 

- Packages
	- mpi4py 

## 6. Experimental Results 


## 7. Related Work 

- RL Applications
- Hyperparameter Search 
- Hyperparameter Tuning Libraries 

## 8. Conclusion


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

