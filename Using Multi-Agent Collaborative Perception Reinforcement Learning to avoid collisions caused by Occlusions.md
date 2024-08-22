# Using Multi-Agent Collaborative Perception Reinforcement Learning to avoid collisions caused by Occlusions.



## Related Works

### 1.  *Social Robot Navigation*

#### a. Model based methods

#### b.  RL and trajectories prediction method

**Pros**: Using DRL for dynamic planing, using advanced neural networks like Attntion or GCN for humans trajectories prediction, achieving excellent social navigation sucessful rate. 

**Cons**: assuming the robot has perfect observation and It is difficult to abstract all obstacles into circles in reality

​	**C. Chen, S. Hu, P. Nikdel, G. Mori and M. Savva, "Relational graph learning for crowd navigation",**

​	 **S. Liu, P. Chang, Z. Huang, N. Chakraborty, K. Hong, W. Liang, et al., "Intention aware robot crowd navigation with attention-based interaction graph"**

​	**Li,  et al. "Multi-Agent Dynamic Relational Reasoning for Social Robot Navigation." **

​	**SoNIC: Safe Social Navigation with Adaptive Conformal Inference and Constrained Reinforcement Learning**

​	



### 2. Multi-Agent Reinforcement Learning



​	**W. Wang, L. Mao, R. Wang and B. -C. Min, "Multi-Robot Cooperative Socially-Aware Navigation Using Multi-Agent Reinforcement Learning,"** 

​	**R. Han, S. Chen and Q. Hao, "Cooperative Multi-Robot Navigation in Dynamic Environment with Deep Reinforcement Learning,"** 

​	**Decentralized Multi-Robot Navigation for Autonomous Surface Vehicles  with Distributional Reinforcement Learning**



### 3. Occupancy grid map

#### a. cooperative OGM construction

​	Y. F. Chen、M. Liu、M. Everett and J. P.How，“Decentralized non-communication multiagent collision avoidance with deep reinforcement learning”



#### b. Ogm prediction

​	Xie, Zhanteng, and Philip Dames. "Stochastic Occupancy Grid Map Prediction in Dynamic Scenes."



## Problem Setup

static obstacle

ogm

multi agent

human policy：

## APPROACH

### 1.  MARL

Traditional RL:

Distributional RL:

#### Observation_space:

​	**robot_information:** global positon, goal position, 
​	**2-Channel-Ogm: occpancy rate, semantic label**

​	converted from 2d Lidar, label information from Camera 

​	**other robot informations**: robot in range

​		

#### Action Space:

​	holonomic: vx,vy

​	unicycle: V_angular, V_linear

​	Steering: [-1,1]	

#### Reward:

​	reach_goal:

​	collision: 

​	danger: to close to robots/humans

​	explore: 

#### Training Algorithm



### 2. Simulator

#### training based on CrowdNav Simulator



#### ogm generation:

pos,theta -> 2d-lidar -> local_ogm -> local_ogm_seq -> predicted_ogm -> rnn -> actor_critic -> action/value



### 3. cooperative OGM prediction

#### When to fusion？





## Experiment and Result

### 0. Experiment Setup

Training: 2d-simulator, randomly generated map

Test:

specifically designed map with corner/wall/crossroad……





### 1. cooperative perception can improve navigation performance

our simple network + CP	VS	our simple network without CP

baseline ogm plan method + CP	VS	baseline ogm plan method

our simple network + CP	?VS 	baseline ogm plan method



### 2.cooperative perception can improve OGM prediction performance



### 3. Real world experiments

turtle bot

## conclusion





