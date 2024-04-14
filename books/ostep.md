# Part I: 
# 12. A Dialogue on Memory Virtualziation
# 13. The Abstraction: Address Spaces

## 13.1 Early Systems

## 13.2 Multiprogramming and Time Sharing
- protection
	- one process cannot read or write over other processes memory

## 13.3 Address Space

## 13.4 Goals

## 13.5 Summary

# 23. Complete Virtual Memory Systems
- How to build a complete VM system
	- VAX/VAS operating system
		- moderm virtual memory manager
	- linux operating system
		- VM system is flexible
			- runs on phones and multicore datacenters

## 23.1 VAS/VMS Virtual Memory
	- 32 bit virtual address space per process
	- Divided into 512 byte pages
	- Virtual address = 23 bit VPN, 9 bit offset

- Curse of generality
	- tasks to support broad class of applications and systems

### Memory Management Hardware
### A Read Address Space
### Page Replacement
### Other Neat Tricks

## 23.2 The Linux Virtual Memory System

### The Linux Address Space
### Page Table Structure
### Large Page Support
### The Page Cache
### Security and Buffer Overflows
### Security Problems: Meltdown & Spectre

## 23.3 Summary

# 24. Summary Dialogue on Memory Virtualization
- 

# Part II: Concurrency

## 25. Dialogue on Concurrency

## 26.Concurrency: An Introduction

- Thread
	- abstraction for a single running process

- Context Switch
	- moving form thread 1 to thread 2 on same processor

- Process Control Block
	- saves state of a process

- Thread Process Block
	- saves state of a thread

- Address Space Partition
	- Multiple stacks per thread 

- Race condition
- Indeterminate
- Mutual Exlcusion

### 26.1 Why Use Threads?

### 26.5 The Wish for Atomicity

### 26.6 One More Problem: Waiting for Another

### 26.7 Summary: Why in OS Class? 

## 27 Interlue: Thread API

### 27.1 Thread Creation

### 27.2 Thread Completion

## 28 Locks

### 28.1 Locks: The Basic Idea

## 29 Lock-based Concurrent Data Structures
- Thread safe
	- Adding locks to data structures to use threads

### 29.1 Concurrent Counters
- 

### 29.2 Concurrent Linked Lists

### 29.3 Concurrent Queues

### 29.4 Concurrent Hash Table

### 29.5 Summary

## 30 Condition Variables
- 
### 30.1 Definition and Routines
- Condition Variable 
	- Explicit queue that threads wait for state of execution
	- When state changes, CV wakes thread to continue by signaling

### 30.2 The Producer/Consumer (Bounded Buffer) Problem
- Mesa semantics
- Hoare semantics

- Single Buffer Producer/Consumer SOlution
	
### 30.3 Covering Conditions
- 

### 30.4 Summary

## 31 Semaphores
- synchronization primitive
	- used for both locks and condition variables

### 31.1 Semaphores: A Definition
- sem_wait()
- sem_post()

# Part III: Persistence

## 35 Dialogue on Persistence 

## 36 I/O Devices

## 37 Hard Disk Drives

## 38 Redundant Arrays of Inexpensive Disks (RAIDs)
- I/O is slow
	- we ant to disk to be larger and faster with more data
- We want disk to be reliable
	- If disk fails, we do not want data to be lost
- RAID looks like one disk
	- disk = groups of blocks one can read or write
- Using multiple disks in parallel
	1. performance - Greatly speeds up I/O times 
	2. capacity - more data
	3. reliability - spread data accross muleiple disks
		- redundancy -> can tolerate the loss of one disk and keep operating

- Note: transparency enables deployment
	- Can new functionality ensure no changes to rest of the system

- Increased deployability
	- Admins can use RAID without worries of software compatability

### 38.1 Interface and RAID Internals

### 38.2 Fault Model
- Fail-stop fault model
	- Disk can be in one of 2 states (working or failed)
		- Working - all blocks can be read or written
		- Failed - assume data is permanently lost
	- Assumptions
		- Assumse disk failure is easily detected
	- Exceptions
		- Silent Failures (e.g. disk corruptions)
		- Block becomes inacessible (e.g. Latent Sector Error)

### 38.3 How to Evaluate a RAID
- Three axes of evaluation
	- capacity
		- Given N disks with B blocks each, how much is useful capacity
			- N * B with no redundancy 
			- (N * B) / 2 with mirroring (each block copied once)
	- reliability
		- How many disk faults can the design tolerate
	- performance
		- depends on workload

### 38.4 RAID Level 0: Striping

- Striping 
	- No Redundancy
	- Upper Bound on Performance and Capacity

- Simplest Form
	- Stripes blocks across disks
		- Round-robin fashion 
		- Extract parallelism for calls to contigous chunks

Chunk Size
	- Number of blocks per disk before moving to next disk

#### Chunk Sizes
- Decreased chunk size
	- Increase real and write parallelism for single file
	- Positioning time increases
		- Time to acccess blocks across multiple disks
	- Total position time = maximum position time of requests across all drives
- Increased block size
	- Reuduces intra-file parallelism
	- Requires multipel concurrent requests to achieve high throughput
	- Reduces positioning time
		- Less number of disks and blocks 


### 38.5 RAID Level 1: Mirroring

### 38.6 RAID Level 4: Saving Space With Parity

### 38.7 RAID Level 5: Rotating Parity

### 38.8 RAID Comparison: A Summary


### 38.9

### 38.10

## 40. File System Implementation

## 40.2 Overall Organization

1
2
3
4
5
6

