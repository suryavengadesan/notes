# Cloud to Sky Computing 
https://sigops.org/s/conferences/hotos/2021/papers/hotos21-s02-stoica.pdf



# Learning symbolic rules for reasoning in quasi-natural language 
https://arxiv.org/pdf/2111.12038.pdf

# Intro

- challenges to transfer formal verification to informal domains 
	- such as common sense knowledge and natural language

- build a system that performs reasoning through explicit symbols 
	- also flexible to handle natural language input 
	- allows for better interpretability and verifiability 

- learn rules from data 
	- (1) determine symbols and manipulations allowed in system or rules 
	- (2) determine learning algorithm to induce rules from training data 

- MetaQNL = Quasi-natural language = formal symbolic system 
- MetaInduce = Learning algorithm = induces MetaQNL rules from data 

- MetaQNL
	- sentence = sequence of words and variables 
	- rule = multiple premises (sentences), one conclusion (sentence)

- variables are swapped with specific concrete instances in sentences 

- Solve Task to reach Goal from Assumptions
	- (1) Rule induction
	- (2) Theorem Proving 

- Meta Induce => Rule Induction
	- Discrete optimization problem
		- seeks minium set of rules consistent with training examples 
	- Search discrete, combinatorial search space 
	- Steps 
		- Encode problem as MAX-SAT problem (maximum-satisfiability)
			- (1) rule properose creates set of rules as candidates
			- (2) generate abstract rules from concrete rules 
				- use anti-unifcation procedure 
			- (3) encode proof paths in MAX-SAT and solve for subset of rules using MAX-SAT solver 

- Benchmarking
	- 3 tasks 
		- (1) learning compositional instructions 
			- miniSCAN and SCAN
		- (2) logical reasoning
			- RuleTaker 
		- (3) morphological analysis
			- real-world data 
	- Uses 2869 symbols, competitive with 11 billion parameters 

## 2. Related Work 


## 3. MetaQNL

- Sentences without variables are concrete sentences 
- Rules
- Substitutions
- Partial ordering amongst sentences and rules 
- Proof 
- Theorem Proving 

## 4. MetaInduce 

- Problem setup
- Loss function 

## 5. Soft Matching

## 6. Experiments

## 7. Limitations and Open Questions 

- Concluding Remarks 
	- Improve softmatching, anti-unification, and large-scale training 

# Generating Natural Language Proofs with Verifier-Guided Search

https://arxiv.org/pdf/2205.12443.pdf

## Abstract

## 1. Introduction

## 2. Related Work 

## 3. Generating Natural Proofs

## 4. Our Method: NLProofS

- Three components
	- (1) Stepwise Prover to generate candidate proof steps
	- (2) Verifier for scoring validity of proofs 
	- (3) Algorithm for searching proofs with high aggregated proof scores 

- 



## 5. Main Results

## 6. Analyses

## 7. Conclusion 

# Scallop: From Probabilistic Deductive Databases to Scalable Differentiable Reasoning

https://openreview.net/forum?id=ngdcA1tlDvj 

## Abstract

- Scallop
	- builds upon probabalistic deductive databases
	- develops a provenance framework of a tunable hyperparameter to specify reasoning granularity
		- (i) generalizes exact probabalistic reasoning
		- (ii) asymmtotically reduces computational cost 
		- (iii) provides relative accuracy guarantees 


## 1. Intro

## 2. Illustrative Overview

## 3. Background

- Datadog = logic programming language 

- Syntax
	- Datalog program = (set of facts, set of rules, query)
	- Atom = (predicate, list of argument terms)


- Semantics
	- Executioin of Datalog program
		- Set of all new derived facts using input facts and rules in a bottom-up evaluation strategy 

- Probability Extensions
	- f = probabilistic input fact
		- P(f) = 1, means deterministic fact
		- P(f) < 1, means input fact f has associated probability 
	- J = disjoint probabilities for a given fact f 
	- probabilistic database = set of all input facts and their disjoint probabilites 
	- probabilistic datalog program = probabilistic database along with all rules and queries

- Probability Calculations
	- 


## 4. Framework

## 5. Evaluation

## 6. Discussion and Limitaitons

## 7. Related Work

## 8. Conclusion and Future Work 

## Unresolved Questions
- What is a probabilistic database 

## Research Ideas


## References
- https://en.wikipedia.org/wiki/Probabilistic_database
	- Probabilistic databse 
- https://scallop-lang.github.io/ 	
	- Scallop Website 


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

# Exocompilation for Productive Programming of Hardware Accelerators

https://dl.acm.org/doi/pdf/10.1145/3519939.3523446

## Abstract 

## 1. Introduction

## 2. Example 

## 3. The Exo Language System 

## 4. Formal Core Language 

## 5. Effect Analysis & Transformation of Programs 

## 6. Contextual Analyses 

## 7. Case Studies 

## 8. Related Work 

## 9. Limitation & Future Work