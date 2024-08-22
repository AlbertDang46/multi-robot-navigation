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

### a. cooperative OGM construction

​	Y. F. Chen、M. Liu、M. Everett and J. P.How，“Decentralized non-communication multiagent collision avoidance with deep reinforcement learning”



### b. OGM Prediction


##### Video Prediction Methods
- **Take OGM Prediction as Video Prediction, thus use Video Prediction method**
- **(Common baseline)1. ConvLSTM**
- **(Common baseline)2. PredNet Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning**
- **3. Disentangling Physical Dynamics from Unknown Factors for Unsupervised Video Prediction**
##### Bayesian Filtering and Dempster-Shafer Theory Methods
- **Only find python2 or c codebase**
- **1. 2015: Conditional Monte Carlo Dense Occupancy Tracker**
- **2. 2017: Dynamic Occupancy Grid Prediction for Urban Autonomous Driving: A Deep Learning Approach with Fully Automatic Labeling**
- **3. 2018: A Random Finite Set Approach for Dynamic Occupancy Grid Maps with Real-Time Application**
##### Learning Based Methods
- **Use RNN-based network to predict**
- **(Common baseline)1. 2016: Deep Tracking: Seeing Beyond Seeing Using Recurrent Neural Networks**
- **(Currently use)2. 2023: SOGMP++/SOGMP: Stochastic Occupancy Grid Map Prediction in Dynamic Scenes**
- **3. 2019: Multi-Step Prediction of Occupancy Grid Maps with Recurrent Neural Networks**
- **4. 2021: Double-Prong ConvLSTM for Spatiotemporal Occupancy Prediction in Dynamic Environments**
- **5. 2019: 2D Lidar Map Prediction via Estimating Motion Flow with GRU**
- **6. 2022: Learning Spatiotemporal Occupancy Grid Maps for Lifelong Navigation in Dynamic Scenes (3D)**
- **(Computer Vision for Dynamics/Static Segmentation) 7. 2017: Fully Convolutional Neural Networks for Dynamic Object Detection in Grid Maps**
- **(Computer Vision for Dynamics/Static Segmentation) 8. 2018: Dynamic Occupancy Grid Prediction for Urban Autonomous Driving: A Deep Learning Approach with Fully Automatic Labeling**



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





