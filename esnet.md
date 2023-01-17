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

- Grid search 

## 3. Bayesian Optimization for Hyperparameter Search

- Bayesian Optimization
	- hyperparameter search on non-convex search spaces

- Fitness function approximated using gaussian kernels 
	- Use a function to determine next best point
		- expected maximizaiton
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

- Bayesian optimization (BO) requires large number of episodes to receive optimized results 
- Genetic algorithms (GA) provide more exploration, and therefore better optimization than BO 
- GA can be parallelized, while BO cannot 
- BO works well on continous hyperparameter surfaces, but early RL learning benefits more from GA's randomness


## Unresolved Questions
- Significance of figures 5a and 5b

## Research Ideas
- Better understand current hyperparameter tuning methods in deep RL
	- Explore population-based training methods


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