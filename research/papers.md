# How to Build Research System
https://ratul.org/papers/ccr2010-howto.pdf

## Abstract
- inspired by Hal Varian on building economic models

## 1. Introduction 

## 2. Building a Research System
- Pick domain carefully
- Know problem well before building
- Debate several solution ideas
- Have core idea build behind the build
- Smart small when building
- Make it real

## 3. 

## 4. 

## 5. Idea Triage 

## 6. Building and Evaluating

## 7. Making it Real 

# Systems Research Advice by Lalith Suresh [TODO]

https://lalith.in/2020/09/27/Low-Level-Advice-For-Systems-Research/

# Systems Research Checklist [TODO]
https://obssr.od.nih.gov/sites/obssr/files/inline-files/best-practices-consensus-statement-FINAL-508.pdf


# How to do systems research [TODO]

https://www.cse.iitk.ac.in/users/biswap/HTDSR.pdf 

# Modularity in RL via Algorithmic Independence in Credit Assigment

https://arxiv.org/pdf/2106.14993.pdf

# DynamoDB [TODO]

https://assets.amazon.science/33/9d/b77f13fe49a798ece85cf3f9be6d/amazon-dynamodb-a-scalable-predictably-performant-and-fully-managed-nosql-database-service.pdf


# From Cloud Computing to Sky Computing 

https://sigops.org/s/conferences/hotos/2021/papers/hotos21-s02-stoica.pdf 

## 3. Lessons from the Intmernet 

## 4. Compatability Layer 

## 5. Intercloud Layer 

- Directory Service 
- Accounting and Charging 

# 6. Peering Between Clouds 

# Super Charging Distributed Computing Environments for High Performance Data Engineering 

https://arxiv.org/pdf/2301.07896.pdf 

# Gavel: Heterogeneity-Aware Cluster Scheduling Policies

https://www.usenix.org/system/files/osdi20-narayanan_deepak.pdf 

## Abstract

## 1. Introduction 

## 2. Background

## 3. System Overview

## 4. Scheduling Policies

## 5. Scheduling Mechanism

## 6. Implementation

## 7. Evaluation

## 8. Related Work and Discussion

## 9. Conclusion

## 10. Acknowledgements 


# Apache Mesos
https://people.eecs.berkeley.edu/~alig/papers/mesos.pdf

## Abstract
- resource offers
	- distributed two-level scheduler 
- mesos 
	- decides how many resources to offer each framework
- frameworks
	- which resources to accept and which computations to run 

## 1. Introduction

## 2. Target Environment

## 3. Architecture

## 4. Mesos Behavior

## 5. Implementation

- 10,000 lines of C++

## 6. Evaluation

## 7. Related Work

## 8. Conclusion

# RubberBand: Cloud-based Hyperparamater Tuning
https://dl.acm.org/doi/pdf/10.1145/3447786.3456245 

## Abstract

## 1. Introduction

## 2. Background

## 3. Cost-efficient Hyperparameter Tuning

## 4. Cloud-based Hyperparameter Tuning

## 5. Implementation

## 6. Evaluation

## 7. Discussion and Related Work

## 8. Conclusion

# DRL using GA for Parameter Optimization
https://arxiv.org/pdf/1905.04100.pdf

## Abstract

## 1. Introduction

## 2. Related Owrd

## 3. Background

## 4. DDPG + HER and GA

## 5. Experiment and Results

## 6. Discussion and Future Work 

# InstructGPT
https://arxiv.org/pdf/2203.02155.pdf

## 1. Introduction

## 2. Related Work

## 3. Methods and Experimental Data

## 4. Results

## 5. Discussion


# InstructGPT Blog
https://openai.com/blog/instruction-following/

## Results

## Methods

## Generalizing to Broader Preferences

## Limitations


# ChatGPT

https://openai.com/blog/chatgpt/

## Methods

- RL with Human Feedback 
	- i.e. InstructGPT

- Initial Model 
	- Supervised finetuning 
		- Human played both the user and AI assistant
	
- Reward Model
	- Collect Comparison Data
		- 2 >= responses ranked by quality 
		- Randomly select model message
			- Sample alternative completions 
			- AI trainers rank completions
	- Finetune PPO 
		- Use above reward defintion
	- Repeat 

## Limitations
- plausible-sounding by incorrect answers
- sensitive to input phrasing tweaks
	- claims doesn't know answers at first until rephrasing question
- over-optimization issues
	- verbose with repeated phrases
	- AI trainers perferred long answers
- lack of clarifying questions
	- guesses answer based on incomplete understanding of user's intention
- moderation API 
	- warn and blocks unsafe content 

## Iterative Deployment
- Iteratively improve safety flaws

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

# Decoupled Neural Interfaces Using Synthetic Gradients [TODO]
https://arxiv.org/pdf/1608.05343.pdf

# Dominant Resource Fairness [TODO]
https://cs.stanford.edu/~matei/papers/2011/nsdi_drf.pdf 

## Abstract

- Dominant Resource Fairness
	- generalization of max-min fairness

## 1. Introduction

## 2. Motivation

- 2000 Node Facebook cluster
	- Map-Reduce
		- Reduce is memory heavy
		- Map is CPU heavy 

- Fair Schedulers for Clusters
	- Quincy
	- Hadoop Fair Scheduler

## 3. Allocation Properties

- Four Properties for Fair Schedulers
	- Sharing Incentive
	- Strategy-Proofness
	- Envy-freeness
	- Pareto Efficiency 

- Four Nice To Have Properties
	- Single Resource Fairness
	- Bottleneck Fairness
	- Population Monotonicity
	- Resource Monotonicity

## 4. Dominant Resource Fairness (DRF)

## 5. Alternative Fair Allocation Policies

## 6. Analysis

## 7. Experimental Results

## 8. Related Work

## 9. Conclusion and Future Work 


# Waiting Game
https://lass.cs.umass.edu/papers/pdf/sc20-waiting.pdf

## Abstract

- Onprem cheaper for high utilization jobs 
- Waiting Policy
	- Dual of scheduling policy 
- 14M jobs run on 14.3k-core cluster

## 1. Introduction

- Scheduling Policy
	- determines which jobs to run when resources available
- Waiting Policy
	- which jobs wait for fixed resources when fixed resource not available
		- avoids spot-instance 

- Non-selective Waiting Policies
	- All jobs wait (AJW)
	- No jobs wait (NJW)
	- AJWT (AJW-T)
- Selective policies
	- Short Waits Wait (SWW)
	- Long Jobs Wait (LJW)
- Waiting Policy Models and Analysis
	- utilize two tools
		- marginal analysis from economics
		- queuing theory from operating systems 
- Implementation and Evaluation
	- Use trace driven job simulator

## 2. Background and Intuition
- Pricing Dynamics 
	- 

## 3. Non-selective Waiting Policies

## 4. Selective Waiting Policies

## 5. Implementation

## 6. Evaluation

## 7. Related Work 

# (Spark) Resilient Distributed Datasets: Fault-tolerant abstraction for in-memory cluster computing 

## Abstract
- distributed memory abstraction
	- allows programmers to perform in-memory computations on clusters
	- provides in-memory computation speedup 


## 1. Introduction
- existing methods
	- external storage (e.g. distributed filesystem) 
		- incurs data replicaiton, disk I/O, serialization

## 2. Resilient Distributed Datasets

- RDD abstraction 
	- read-only, partitioned collection of records

- spark programming interface 
	- 

## 3. Spark Programming Interface

## 4. Representing RDDs

## 5. Implementation

## 6. Evaluation

## 7. Discussion

## 8. Related Work 

## 9. Conclusion


# Ray: Distributed Framework for Emerging AI Applications

https://arxiv.org/pdf/1712.05889.pdf 

## Abstract

## 1. Introducion

## 2. Motivation and Requirements

- Training
	- all reduce or parameter server 
- Serving 
	- minimize latency for inference

- RL Needs
	- Fine-grained heterogenous computation
	- Flexible computation model 
	- Dynamic execution

## 3. Programming and Computation Model 
- Implements dynamic task graph computation model
	- application = graph of dependent tasks
		- evolves during execution of app
- Provides actor and task-parallel abstractions
- Different om CIEL, Orleans, Akks

## 3.1 Programming Model

## 3.2 Computation Model 

## 4. Architecture

## 4.1 Application Layer

## 4.2 System Layer

## 4.3 Putting Everything Together


## 5. Evaluation

## 6. Related Work 

- Dynamic Task Graphs
	- CIEL
		- lineage-based fault tolerance
	- Dask
		- fully integrates with python
- Dataflow Systems
	- MapReduce, Spark, Dryad
- Machine Learning Frameworks
	- Tensorflow, MXNet
- Actor Systems
- Global Control Store and Scheduling

## 7. Discussion and Experiences

## 8. Conclusion

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

# Dixin Tang Talk
https://people.eecs.berkeley.edu/~totemtang/

# Problems 

- Limited interactivity 
- Incorrect Insights 
- Engineering Cost
	- Data engineers using spark or ray 

# Research Question 
- Build a databse that minimizes resources 
	- Three Parts
		- Interactivity 
			- Transactional Panorama
		- Scalability
			- Modin
			- Taco
			- FormS
		- Resource Utilization 
			- CrocodileDB
	
# Transactional Panorama 
- Virtual Interface used to see data 
	- Examples 
		- Spread sheet systems
			- Each cells needs to be refreshed
		- Visualization Tools 
			- Tableau
- Panorama: Principled Approach 
	- VCM Properties
		- Visibility 
		- Coherence
		- Monotonicity 
	- Possible Property Combinations
	- New Options
- Performance Metrics -> Performance Trade-offs
- Concurrenc Between User and Dashboard
- Database Transactions to the Rescue 
	- Represents multiple reads and writes as a unit of work 

- Isolation
	- How and when changes made by one transactions are visible to other levels 

- Refresh Transaction 
	- Recompute results

- Invsibility 
- Staleness
- Performance Trade-offs 

- Datagram => tabular data 
- Challenges for Scaling Pandas Execution
	- Property 1
		- Output order requirements
	- Property 2
		- Flexible access patterns 

- Modin 
	- Core operators
	- Parallel Execution
		- Decomposition rules 
	- Metadata Management 
		- Lazy Execution

- Output Order Requiresments

- Adaptive Concurrency Control 
- Lux: Always-on Visualization Recommendations 



# Data Analytics Tools 

- Data Scientists, Data Analysts, Domain Experts 
	- User-centered analytical interfaces

# Arthur: A spark debugger [TODO]
https://ankurdave.com/dl/arthur-atc13.pdf

# Population Based Training of Neural Networks [TODO]
https://arxiv.org/pdf/1711.09846.pdf

# Provably Efficient Online Hyperparameter Optimization with Population-Based Bandits [TODO]
https://arxiv.org/pdf/2002.02518.pdf

# Human-level performance in first-person multiplayer games with population-based deep reinforcement learning [TODO]
https://arxiv.org/pdf/1807.01281.pdf


# PLB: Congestion Signals are Simple and Effective for Network Load Balancing [TODO]
https://dl.acm.org/doi/pdf/10.1145/3544216.3544226

## Abstract

- PLB = Protective Load Balancing 
	- transport protocol 
	- ECMP/WCMP reduce network hotspots 
- Link imbalance 

## Introduction 

## Reading Group 

# Cloud to Sky Computing [TODO]
https://sigops.org/s/conferences/hotos/2021/papers/hotos21-s02-stoica.pdf


# Learning symbolic rules for reasoning in quasi-natural language [TODO]
https://arxiv.org/pdf/2111.12038.pdf

## Introduction

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

# HYPPO 
https://arxiv.org/pdf/2110.01698.pdf

## 1. Introduction 

- HPO = hyperparameter optimization 

## 2. Motivation 

## 4. HYPPO Software Design 

- Feature 1: Uncertainty Quantification 

- Feature 2: Surrogate Modeling 


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

# Exocompilation for Productive Programming of Hardware Accelerators
https://dl.acm.org/doi/pdf/10.1145/3519939.3523446

## Abstract 

## 1. Introduction

## 2. Example 

- GEMM = general matrix multiply 
- Optimize GEMM 
	- (1) orchestrating data movement 
	- (2) selecting compute instructions

### 2.1 Procredures, Compilation, Scheduling

- exo function uses @proc decorator 
	- takes in type, size, memory arguments 

- Store loads and stores from custom accelerator memories 
	- Define customer memories, instructions, and configuration state 

- Write hardware library per accelerator 

- Exo compiles python to C 

- Use scheduling operations to rewrite python procedure 
	- split
	- reorder 

### 2.2 Memories

- define accumulator class 
	- alloc, free, and read functions 

### 2.3 Instructions 

- define accelerator instructions without expliciting modifying the compiler i.e. exocompilation 

- instr annotation 
	- annotation contains C code macro 
	- body of function contains exo code 
	- maps equivalence between C and exo 

## 3. The Exo Language System 

## 4. Formal Core Language 

## 5. Effect Analysis & Transformation of Programs 

## 6. Contextual Analyses 

## 7. Case Studies 

## 8. Related Work 

## 9. Limitation & Future Work

# SQIL: Imitation Learining 
https://arxiv.org/abs/1905.11108

## Abstract

## 1. Introduction 

- Standard learning or off-policy actor critic algorithm
	- Key modifications
		- Add reward for 1 for demonstrated actions in a state 
		- Add reward of 0 for all other actions in a state 

## 2. Soft Q Imitation Learning 

- 3 modifications 
	- (1) fills agent's experience replay buffer with demonstration set to reward +1 
	- (2) agents interacts with world and adds experiences with reward 0 
	- (3) 

## 3. Interpreting SQIL as Regularized Behavior Cloning 

## 4. Experimental Evaluation 

## References 
- https://spinningup.openai.com/en/latest/spinningup/keypapers.html
- Research Meeting Notes 
	- https://docs.google.com/document/d/1VRVjUm9qZACboFSpiTYnS-dbN535SXVsJsACUIx7oWk/edit