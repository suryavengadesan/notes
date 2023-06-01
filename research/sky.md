# Sky Retreat
# Introduction
## Ion's Talk

- Why Sky
	- Leverage best hardware
- How? 
	- Layered architectures
		- Portability layer
			- Azure ARC, Anthos
		- Complex portability layer
			- 10x more functionality than traditional OS 
		- Two-sided market
			- on side: existing services 
		- Compatability Set
			- Y-axis: CLouds, 3Parties
			- X-axis: Open source, proprietary
		- Four Quadrants
			- 3rd Party Open Source
			- Cloud Open Source
			- 3d party propreity
			- Cloud proprietary
		- First App Domain: ML
		- Why it will work? 
			- Start today with existing services
- Types of Projects
	- Exploration
		- New ideas 
	- Exploitation
		- Commit to devlop w/ positive feedback
- Example Project
	- SkyPilot

## Raluca's + (Natacha's) Talk
- Security Thrusts
	- security layer across clouds
	- distributed trust across clouds

## Joey's Talk
- Skyplane
	- applications
		- container registry, dataloaders
- Hydro
	- hydrologic
	- hydroflow
	- hydrodeploy
- LMSys 
	- Vicuna
- LLM Serving
	- Alpa
	- CacheLLM
- FrugalGPT
	- LLM Broker

## Koushik's Talk
- PL (programming languages/systems)
	- Distributed Computing Models
		- Hydro
	- Software Assistants
		- Code Scholar
		- Code Sense 
	- Testing Frameworks
		- ItyFuzz

# Sky Compute Session

## Sky Plane
- Cloud data challenges
	- partitions by region, provider, & services
- Locked in by data gravity 
	- egress fees
- Transfer broker
	- Optimizing cloud networking
	- NSDI paper
		- overlay routing
		- vm's per region

## Sky Container
- Improve container startup time
	- container technology underpinds cloud computing

## SkyPilot
- Jobs runs across many clouds based only on job configs 

## SkySpot
- Policy design for using spot VMs
- Recovery Policy
	- (1) Wait and retry
	- (2) Switch and retry between ondemand and spot
	- (3) Switch the zones

- Goal: minimize costs while finishing job by deadline
	- Scenario 1: job has no changeover delay
	- Scenario 2: Job has changeover delay

- Baseline policy
	- Use demand instance the entire time
- Optimal Policy*

- Derived rules
	- thrify rule
	- safety net rule 

# Sky Collaborators

## Samsung Cloud

## IBM Research
- Paul C Castro
	- castrop@us.ibm.com
- Diana Arroyo
	- https://researcher.watson.ibm.com/researcher/view.php?person=us-darroyo
		- https://github.com/project-codeflare/multi-cluster-app-dispatcher

# Sky Data

## Memory Optimization Methods in Model Training by Lily Liu
- Techniques
	- Tensor parallelism
	- Gradient checkpointing 

## SkyStorage by Shu Liu
- data across regions and providers
	- implements unified data store across clouds

## Hydro by Shadaj Laddad
- systems break when scaling app to multiple clouds
	- hydro logic
		- high level language 
	- hydro flow
		- rust kernel for dataflow
	- hydro deploy

## Transaction performance by Audrey
- Minimize makespan
	- time between start and end of schedule 


## Topolotree by Conor 
- Simplify graph problem into equivalent tree problem
	- Handling failures are further complicated when mapping to trees

## Starburst by Michael 
- Cost aware cloud scheduling

## BFTs by Neil 

## Vicuna by Lianmin Zheng
- 