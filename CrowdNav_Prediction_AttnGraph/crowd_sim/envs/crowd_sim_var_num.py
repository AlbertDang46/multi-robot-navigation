import gym
import numpy as np
from numpy.linalg import norm
import copy
import sys

import torch
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs import *
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.state import JointState
import os
import random
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import patches
from skimage.draw import line
from crowd_sim.envs.utils.lidar2d import Lidar2d,merge_ogm,merge_lidar

class CrowdSimVarNum(CrowdSim):
    """
    The environment for our model with no trajectory prediction, or the baseline models with no prediction
    The number of humans at each timestep can change within a range
    """
    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.
        """
        super().__init__()
        self.id_counter = None
        self.observed_human_ids = []
        self.pred_method = None
        self.frame_count = 0

        self.map_size = 32
        self.static_map_size = None
        self.original_map = None
        self.map_artists = []
        self.obst_directions = []
        self.lidar = None
        self.robots_connection_graph = None
        self.num_ray=90
        self.TIME=0
        


    # set the observation space and action space
    def set_space(self):
        # we set the max and min of action/observation space as inf
        # clip the action and observation as you need
        d={}
        # robot node: px, py, r, gx, gy, v_pref, theta,vx, vy
        d['robot_info'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,9,), dtype = np.float32)

        # occupancy grid map: (2,map_size,map_size), 0: occupancy map, 1: semantic label map
        d['occupancy_map'] = gym.spaces.Box(low=-np.inf, high=np.inf,shape=(2,self.map_size, self.map_size), dtype=np.float32)
        d['lidar'] = gym.spaces.Box(low=-np.inf, high=np.inf,shape=(self.num_ray,2), dtype=np.float32)
        # detected robots info: relative px, relative py, disp_x, disp_y, sorted by distance
        d['detected_robots_info'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.robot_num, 4), dtype=np.float32)


        # only consider all temporal edges (human_num+1) and spatial edges pointing to robot (human_num)
        d['temporal_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 2,), dtype=np.float32)
        d['spatial_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_human_num, 2), dtype=np.float32)
        # number of humans detected at each timestep
        d['detected_human_num'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, ), dtype=np.float32)
        # whether each human is visible to robot (ordered by human ID, should not be sorted)
        d['visible_masks'] = gym.spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.max_human_num,),
                                            dtype=bool)
        # occupancy grid map
        
        #self.observation_space.append(gym.spaces.Dict(d))
        self.observation_space=gym.spaces.Dict(d)

        high =  np.ones([1, ])
        #self.action_space.append(gym.spaces.Box(-high, high, dtype=np.float32)
        self.action_space=gym.spaces.Box(-high, high, dtype=np.float32)


    # configure the environment with the config file, and set the observation space and action space
    def configure(self, config):
        """ read the config to the environment variables """
        super(CrowdSimVarNum, self).configure(config)
        self.action_type=config.action_space.kinematics
        self.set_space()


    # set observation space and action space
    def set_robot(self,robot):
        #create a robot and add it to the list of robots
        self.robots.append(robot)

        # set the robot's nearest onstable position to train the robot avoid obstacles
        self.obst_directions.append(np.array([0.0, 0.0]))

    
    # set robot initial state and generate all humans for reset function
    # for crowd nav: human_num == self.human_num
    # for leader follower: human_num = self.human_num - 1
    def generate_robot_humans(self, phase, human_num=None):
        for robot in self.robots:
            #define human for several times
            if self.record:
                px, py = 0, 0
                gx, gy = 0, -1.5
                robot.set(px, py, gx, gy, 0, 0, np.pi / 2)
                # generate a dummy human
                for i in range(self.max_human_num):
                    human = Human(self.config, 'humans')
                    human.set(15, 15, 15, 15, 0, 0, 0)
                    human.isObstacle = True
                    self.humans.append(human)

            else:
                # for sim2real
                if robot.kinematics == 'unicycle':
                    while True: 
                        px, py, gx, gy = np.random.uniform(-self.arena_size-2, self.arena_size+2, 4)
                        grid_px = int(self.static_map_size / 2 + px / self.cell_length)
                        grid_py = int(self.static_map_size / 2 + py / self.cell_length)
                        # check if the robot is in the obstacle
                        while np.sum(self.original_map[grid_px-2:grid_px+3,grid_py-2:grid_py+3]) > 0:
                            px, py = np.random.uniform(-self.arena_size-2, self.arena_size+2, 2)
                            grid_px = int(self.static_map_size / 2 + px / self.cell_length)
                            grid_py = int(self.static_map_size / 2 + py / self.cell_length)
                        
                        grid_gx = int(self.static_map_size / 2 + np.floor(gx / self.cell_length))
                        grid_gy = int(self.static_map_size / 2 + np.floor(gy / self.cell_length))
                        # check if the goal is in the obstacle
                        while np.sum(self.original_map[grid_gx-2:grid_gx+3,grid_gy-2:grid_gy+3]) > 0:
                            gx, gy = np.random.uniform(-self.arena_size-2, self.arena_size+2, 2)
                            grid_gx = int(self.static_map_size / 2 + gx / self.cell_length)
                            grid_gy = int(self.static_map_size / 2 + gy / self.cell_length)
                        
                        # check if the robot and goal are too close
                        if np.linalg.norm([px - gx, py - gy]) >= 8: # 6
                            break
                    robot.set(px, py, gx, gy, 0, 0, np.random.uniform(0, 2 * np.pi))  # randomize init orientation
                    # 1 to 4 humans
                    self.human_num = np.random.randint(1, self.config.sim.human_num + self.human_num_range + 1)
                    # self.human_num = 4


                # for sim exp
                else:
                    # generate robot
                    while True: 
                        px, py, gx, gy = np.random.uniform(-self.arena_size-2, self.arena_size+2, 4)
                        grid_px = int(self.static_map_size / 2 + px / self.cell_length)
                        grid_py = int(self.static_map_size / 2 + py / self.cell_length)
                        # check if the robot is in the obstacle
                        while np.sum(self.original_map[grid_px-2:grid_px+3,grid_py-2:grid_py+3]) > 0:
                            px, py = np.random.uniform(-self.arena_size-2, self.arena_size+2, 2)
                            grid_px = int(self.static_map_size / 2 + px / self.cell_length)
                            grid_py = int(self.static_map_size / 2 + py / self.cell_length)
                        
                        grid_gx = int(self.static_map_size / 2 + np.floor(gx / self.cell_length))
                        grid_gy = int(self.static_map_size / 2 + np.floor(gy / self.cell_length))
                        # check if the goal is in the obstacle
                        while np.sum(self.original_map[grid_gx-2:grid_gx+3,grid_gy-2:grid_gy+3]) > 0:
                            gx, gy = np.random.uniform(-self.arena_size-2, self.arena_size+2, 2)
                            grid_gx = int(self.static_map_size / 2 + gx / self.cell_length)
                            grid_gy = int(self.static_map_size / 2 + gy / self.cell_length)
                        
                        # check if the robot and goal are too close
                        if np.linalg.norm([px - gx, py - gy]) >= 8: # 6
                            break
                        
                    robot.set(px, py, gx, gy, 0, 0, np.pi / 2)
                    robot.deactivated = False #  
        
        # generate humans
        self.human_num = np.random.randint(low=self.config.sim.human_num - self.human_num_range,
                                                    high=self.config.sim.human_num + self.human_num_range + 1)

        #enter crowd_sim/generate_random_human_position
        #enter crowd_sim_var_num/generate_circle_crossing_human
        self.generate_random_human_position(human_num=self.human_num)
        self.last_human_states = np.zeros((self.human_num, 5))
        # set human ids
        for i in range(self.human_num):
            self.humans[i].id = i


    # generate a human that starts on a circle, and its goal is on the opposite side of the circle
    def generate_circle_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            #enter here
            human.sample_random_attributes()
        
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            noise_range = 2
            px_noise = np.random.uniform(0, 1) * noise_range
            py_noise = np.random.uniform(0, 1) * noise_range
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False

            # check if the human's goal is in the obstacle
            x1 = int(self.static_map_size / 2 - px / self.cell_length)
            y1 = int(self.static_map_size / 2 - py / self.cell_length)
            if 0 <= x1 < self.static_map_size and 0 <= y1 < self.static_map_size and self.original_map[x1, y1] > 0:
                continue

            for i, agent in enumerate(self.robots + self.humans): #check further!
                # keep human at least 3 meters away from robot
                if i < self.robot_num:
                    if agent.kinematics == 'unicycle':
                        min_dist = self.circle_radius / 2 # Todo: if circle_radius <= 4, it will get stuck here
                if i >= self.robot_num or (i < self.robot_num and agent.kinematics != 'unicycle'):
                    min_dist = human.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            if not collide:
                break
                
        human.set(px, py, -px, -py, 0, 0, 0)
        return human


    # calculate the ground truth future trajectory of humans
    # if robot is visible: assume linear motion for robot
    # ret val: [self.predict_steps + 1, self.human_num, 4]
    # method: 'truth' or 'const_vel' or 'inferred'
    def calc_human_future_traj(self, method):
        # if the robot is invisible, it won't affect human motions
        # else it will
        human_num = self.human_num + 1 if self.robots[0].visible else self.human_num
        # buffer to store predicted future traj of all humans [px, py, vx, vy]
        # [time, human id, features]
        if method == 'truth':
            self.human_future_traj = np.zeros((self.buffer_len + 1, human_num, 4))
        elif method == 'const_vel':
            self.human_future_traj = np.zeros((self.predict_steps + 1, human_num, 4))
        else:
            raise NotImplementedError

        # initialize the 0-th position with current states
        for i in range(self.human_num):
            # use true states for now, to count for invisible humans' influence on visible humans
            # take px, py, vx, vy, remove radius
            self.human_future_traj[0, i] = np.array(self.humans[i].get_observable_state_list()[:-1])

        # if we are using constant velocity model, we need to use displacement to approximate velocity (pos_t - pos_t-1)
        # we shouldn't use true velocity for fair comparison with GST inferred pred
        if method == 'const_vel':
            self.human_future_traj[0, :, 2:4] = self.prev_human_pos[:, 2:4]

        # add robot to the end of the array
        if self.robots[0].visible:
            self.human_future_traj[0, -1] = np.array(self.robot.get_observable_state_list()[:-1])

        if method == 'truth':
            for i in range(1, self.buffer_len + 1):
                for j in range(self.human_num):
                    # prepare joint state for all humans
                    full_state = np.concatenate(
                        (self.human_future_traj[i - 1, j], self.humans[j].get_full_state_list()[4:]))
                    observable_states = []
                    for k in range(self.human_num):
                        if j == k:
                            continue
                        observable_states.append(
                            np.concatenate((self.human_future_traj[i - 1, k], [self.humans[k].radius])))

                    # use joint states to get actions from the states in the last step (i-1)
                    action = self.humans[j].act_joint_state(JointState(full_state, observable_states))

                    # step all humans with action
                    self.human_future_traj[i, j] = self.humans[j].one_step_lookahead(
                        self.human_future_traj[i - 1, j, :2], action)

                if self.robots[0].visible:
                    action = ActionXY(*self.human_future_traj[i - 1, -1, 2:])
                    # update px, py, vx, vy
                    self.human_future_traj[i, -1] = self.robot.one_step_lookahead(self.human_future_traj[i - 1, -1, :2],
                                                                                  action)
            # only take predictions every self.pred_interval steps
            self.human_future_traj = self.human_future_traj[::self.pred_interval]
        # for const vel model
        elif method == 'const_vel':
            # [self.pred_steps+1, human_num, 4]
            self.human_future_traj = np.tile(self.human_future_traj[0].reshape(1, human_num, 4), (self.predict_steps+1, 1, 1))
            # [self.pred_steps+1, human_num, 2]
            pred_timestep = np.tile(np.arange(0, self.predict_steps+1, dtype=float).reshape((self.predict_steps+1, 1, 1)) * self.time_step * self.pred_interval,
                                    [1, human_num, 2])
            pred_disp = pred_timestep * self.human_future_traj[:, :, 2:]
            self.human_future_traj[:, :, :2] = self.human_future_traj[:, :, :2] + pred_disp
        else:
            raise NotImplementedError

        # remove the robot if it is visible
        if self.robots[0].visible:
            self.human_future_traj = self.human_future_traj[:, :-1]


        # remove invisible humans #how to define invisible humans?
        self.human_future_traj[:, np.logical_not(self.human_visibility), :2] = 15
        self.human_future_traj[:, np.logical_not(self.human_visibility), 2:] = 0

        return self.human_future_traj


    # reset = True: reset calls this function; reset = False: step calls this function
    # sorted: sort all humans by distance to robot or not
    def generate_ob(self, robot_index,reset, sort=False):
        
        """Generate observation for reset and step functions"""    

        ob = {}
        # robot info: px, py, r, gx, gy, v_pref, theta
        ob['robot_info'] = self.robots[robot_index].get_full_state_list_noV()

        # occupancy grid map
        # 2*map_size*map_size, 0: occupancy map, 1: semantic label map
        robot_pos_x = int(self.static_map_size / 2 + np.floor(self.robots[robot_index].px / self.cell_length))
        robot_pos_y = int(self.static_map_size / 2 + np.floor(self.robots[robot_index].py / self.cell_length))
        ob['occupancy_map'] = self.lidar.convert_to_bitmap(self.lidar.get_raw_data(robot_pos_x,robot_pos_y,self.robots[robot_index].theta), self.map_size)
        # print(ob['occupancy_map'].shape)#(2,32,32)
        # print(self.lidar.get_raw_data(robot_pos_x,robot_pos_y,self.robots[robot_index].theta).shape)#(90,2)
        
        ob['lidar']=self.lidar.get_raw_data(robot_pos_x,robot_pos_y,self.robots[robot_index].theta)
        
        ob['detected_robots_info'] = np.array([[1e5, 1e5, 0, 0] for _ in range(self.robot_num)], dtype=np.float32)
        
         
        # spatial edges
        ob['temporal_edges'] = np.array([self.robots[robot_index].vx, self.robots[robot_index].vy]) 

        # nodes
        visible_humans, num_visibles, self.human_visibility = self.get_num_human_in_fov(robot_index) #enter crowd_sim/get_num_human_in_fov

        

        #prev_human_pos = copy.deepcopy(self.last_human_states)
        
        self.update_last_human_states(self.human_visibility, reset=reset)

        # edges
        

        # ([relative px, relative py, disp_x, disp_y], human id)
        all_spatial_edges = np.ones((self.max_human_num, 2)) * np.inf

        for i in range(self.human_num):
            if self.human_visibility[i]:
                # vector pointing from human i to robot
                relative_pos = np.array(
                    [self.last_human_states[i, 0] - self.robots[robot_index].px, self.last_human_states[i, 1] - self.robots[robot_index].py])
                all_spatial_edges[self.humans[i].id, :2] = relative_pos
        
        ob['visible_masks'] = np.zeros(self.max_human_num, dtype=bool)
        # sort all humans by distance (invisible humans will be in the end automatically)
        if sort:
            ob['spatial_edges'] = np.array(sorted(all_spatial_edges, key=lambda x: np.linalg.norm(x)))
            # after sorting, the visible humans must be in the front
            if num_visibles > 0:
                ob['visible_masks'][:num_visibles] = True
        else:
            ob['spatial_edges'] = all_spatial_edges
            ob['visible_masks'][:self.human_num] = self.human_visibility
        ob['spatial_edges'][np.isinf(ob['spatial_edges'])] = 15
        ob['detected_human_num'] = num_visibles
        # if no human is detected, assume there is one dummy human at (15, 15) to make the pack_padded_sequence work
        if ob['detected_human_num'] == 0:
            ob['detected_human_num'] = 1

        # update self.observed_human_ids
        self.observed_human_ids.append(np.where(self.human_visibility)[0])
        obst_positions = np.argwhere(ob['occupancy_map'][1][9: 23,9:23] > 0)

        if len(obst_positions) <= 8:
            self.obst_directions[robot_index] = np.array([0.0, 0.0])
        else:
            average_pos = np.mean(obst_positions, axis=0) - np.array([6.5, 6.5]) 
            if np.linalg.norm(average_pos) > 0:
                self.obst_directions[robot_index] = average_pos / np.linalg.norm(average_pos)
            else:
                self.obst_directions[robot_index] = np.array([0.0, 0.0])

        return ob


    # Update the specified human's end goals in the environment randomly
    def update_human_pos_goal(self, human):
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            v_pref = 1.0 if human.v_pref == 0 else human.v_pref
            gx_noise = (np.random.random() - 0.5) * v_pref
            gy_noise = (np.random.random() - 0.5) * v_pref
            gx = self.circle_radius * np.cos(angle) + gx_noise
            gy = self.circle_radius * np.sin(angle) + gy_noise
            collide = False

            if not collide:
                break

        # Give human new goal
        human.gx = gx
        human.gy = gy


    def reset_robot(self, robots):
        for robot in robots:            
            robot.deactivated = False


    def get_static_map(self,map_index):
        self.static_map_size = int(10 * self.map_size / self.robots[0].sensor_range)
        bitmap_file = os.path.join("bitmaps", f"bitmap_{map_index}", "bitmap.npy")
        self.original_map = np.load(bitmap_file)

        self.map_drawed = False
    
    
    # update the robots connection graph
    def update_robots_connection_graph(self):
        self.robots_connection_graph = np.array([[np.linalg.norm([robot.px - other_robot.px, robot.py - other_robot.py]) < self.config.robot.broadcast_range  for other_robot in self.robots] for robot in self.robots])
        #print(self.robots_connection_graph)

    # broadcast robot's observation to other robots which are in the broadcast range
    def broadcast(self,obs):
        self.update_robots_connection_graph()

        broadcasted_obs = copy.deepcopy(obs)
        for i in range(self.robot_num):
            for j in range(self.robot_num):
                if i == j or not self.robots_connection_graph[i, j]:
                    continue
                broadcasted_obs[i]['occupancy_map'] = merge_ogm(
                            broadcasted_obs[i]['occupancy_map'], obs[j]['occupancy_map'],
                            [self.robots[j].px-self.robots[i].px, self.robots[j].py-self.robots[i].py], 
                            self.robots[i].theta, self.robots[j].theta,
                            self.cell_length)
                broadcasted_obs[i]['detected_robots_info'][j] = np.array([(self.robots[j].px - self.robots[i].px)*np.cos(self.robots[i].theta) + (self.robots[j].py - self.robots[i].py)*np.sin(self.robots[i].theta),
                                                                          (self.robots[j].py - self.robots[i].py)*np.cos(self.robots[i].theta) - (self.robots[j].px - self.robots[i].px)*np.sin(self.robots[i].theta),
                                                                          (self.robots[j].vx - self.robots[i].vx)*np.cos(self.robots[i].theta) + (self.robots[j].vy - self.robots[i].vy)*np.sin(self.robots[i].theta), 
                                                                          (self.robots[j].vy - self.robots[i].vy)*np.cos(self.robots[i].theta) - (self.robots[j].vx - self.robots[i].vx)*np.sin(self.robots[i].theta)
                                                                          ])
                broadcasted_obs[i]['lidar'] = merge_lidar(broadcasted_obs[i]['lidar'], obs[j]['lidar'],self.robots[i].px,self.robots[j].px, self.robots[i].py,self.robots[j].py, self.robots[i].theta, self.robots[j].theta)
            # sort the detected robots by distance, the first one is the nearest robot
            broadcasted_obs[i]['detected_robots_info'] = np.array(sorted(broadcasted_obs[i]['detected_robots_info'], key=lambda x: x[0]**2 + x[1]**2))
        return broadcasted_obs
        

    def reset(self, phase='train', test_case=None):
        #self.static_map_size = int(10 * self.map_size / self.robots[0].sensor_range)
        #print('enter reset')
        """
        Reset the environment
        :return:
        """

        self.get_static_map(self.TIME%10)
        self.TIME=self.TIME+1
       
        if self.phase is not None:
            phase = self.phase
        if self.test_case is not None:
            test_case=self.test_case

        for robot in self.robots:
            if robot is None:
                raise AttributeError('robot has to be set!')
        
        assert phase in ['train', 'val', 'test']
        if test_case is not None:
            self.case_counter[phase] = test_case # test case is passed in to calculate specific seed to generate casenp.abs
        self.global_time = 0
        self.step_counter = 0
        self.id_counter = 0
        self.sobel_kernel = np.array([[-1, -1, -1],
                                    [-1,  8, -1],
                                    [-1, -1, -1]], dtype=np.float32) + 0.1* np.random.randn(3,3)

        self.humans = []
        cell_length = 2 * self.robots[0].sensor_range / self.map_size
        self.cell_length = cell_length
        
        # self.human_num = self.config.sim.human_num
        # initialize a list to store observed humans' IDs
        self.observed_human_ids = []

        
        # train, val, and test phase should start with different seed.
        # case capacity: the maximum number for train(max possible int -2000), val(1000), and test(1000)
        # val start from seed=0, test start from seed=case_capacity['val']=1000
        # train start from self.case_capacity['val'] + self.case_capacity['test']=2000
        counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                          'val': 0, 'test': self.case_capacity['val']}

        # here we use a counter to calculate seed. The seed=counter_offset + case_counter
        self.rand_seed = counter_offset[phase] + self.case_counter[phase] + self.thisSeed
        np.random.seed(self.rand_seed)

        self.generate_robot_humans(phase)
        self.reset_robot(self.robots)

        # use a graph to represent the connection between robots
        self.update_robots_connection_graph()
        
        
        # record px, py, r of each human, used for crowd_sim_pc env
        self.cur_human_states = np.zeros((self.max_human_num, 3))
        for i in range(self.human_num):
            self.cur_human_states[i] = np.array([self.humans[i].px, self.humans[i].py, self.humans[i].radius])

        # case size is used to make sure that the case_counter is always between 0 and case_size[phase]
        self.case_counter[phase] = (self.case_counter[phase] + int(1*self.nenv)) % self.case_size[phase]
        # print(self.potential)#None
        # print(self.angle)#not defined yet??
        self.potential=[]
        self.angle=[]
        # initialize potential and angular potential
        for robot in self.robots:
            rob_goal_vec = np.array([robot.gx, robot.gy]) - np.array([robot.px, robot.py])
            self.potent = -abs(np.linalg.norm(rob_goal_vec))
            self.angl = np.arctan2(rob_goal_vec[1], rob_goal_vec[0]) - robot.theta
            if self.angl > np.pi:
                # self.abs_angle = np.pi * 2 - self.abs_angle
                self.angl = self.angl - 2 * np.pi
            elif self.angl < -np.pi:
                self.angl = self.angl + 2 * np.pi
            self.potential.append(self.potent)
            self.angle.append(self.angl)
        
        self.lidar = Lidar2d(self.original_map, self.config.sim.num_ray, self.map_size, self.cell_length)
        


        # update human and robot states in lidar2d
        self.lidar.update_dynamic_map([[human.px, human.py] for human in self.humans], [[robot.px, robot.py] for robot in self.robots])
        # get robot observation
        
        obs=[self.generate_ob(i,reset=True, sort=self.config.args.sort_humans) for i in range(self.robot_num)]
        
        # broadcast robot's observation to other robots
        if self.config.robot.broadcast_open:
            obs = self.broadcast(obs)

        if self.config.sim.render:
            self.ob = obs[0]

        return obs


    def step(self, actions, update=True):
        """
        Step the environment forward for one timestep
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
        """
        # if self.robot.policy.name in ['ORCA', 'social_force']:
        #     # assemble observation for orca: px, py, vx, vy, r
        #     human_states = copy.deepcopy(self.last_human_states)
        #     # get orca action
        #     action = self.robot.act(human_states.tolist())
        # else:
        #     action = self.robot.policy.clip_action(action, self.robot.v_pref)

        # if self.robot.kinematics == 'unicycle':
        #     self.desiredVelocity[0] = np.clip(self.desiredVelocity[0] + action.v, -self.robot.v_pref, self.robot.v_pref)
        #     action = ActionRot(self.desiredVelocity[0], action.r)
        
        for i in range(self.robot_num):
            if self.robots[i].kinematics == 'holonomic':
                former_angle = np.array([self.robots[i].vx, self.robots[i].vy], dtype=np.float32)
                former_v = np.linalg.norm(former_angle)
                new_v = np.linalg.norm(i)
                if former_v == 0 or new_v == 0:
                    omega = 0
                else:
                    omega = np.arccos(np.clip(np.dot(former_angle, actions[i]) / (former_v * new_v), -1, 1)) / np.pi
                    assert 1>= omega >= 0
            else:
                omega = 0
            
            actions[i] = self.robots[i].policy.clip_action(actions[i], self.robots[i].v_pref * (1-omega))
            
        # get human actions        
        human_actions = self.get_human_actions()

        # need to update self.human_future_traj in testing to calculate number of intrusions
        self.phase = 'train'
        if self.phase == 'test': #why enter this in training?
            # use ground truth future positions of humans
            self.calc_human_future_traj(method='truth')

        # compute reward and episode info
        all_done = False
        all_rewards = 0
        rewards_list=[]
        info_list=[]

        for i in range(self.robot_num):
            
            reward, done, episode_info= self.calc_reward(i,actions[i], danger_zone='future')
            all_rewards+=reward
            rewards_list.append(reward)
            info_list.append(episode_info)
            
            
                

        if all(self.robots[i].deactivated for i in range(self.robot_num)):
            all_done = True
            if all(isinstance(info_list[i], ReachGoal) for i in range(self.robot_num)):
                episode_info=ReachGoal()
            if any(isinstance(info_list[i], Collision) for i in range(self.robot_num)):
                episode_info=Collision()
        if isinstance(episode_info, Timeout):
            assert all(isinstance(info_list[i], Timeout) for i in range(self.robot_num))
            all_done = True
        

        # apply action and update all agents
        for i in range(self.robot_num):  
            if self.robots[i].deactivated:
                continue
            
            if self.robots[i].kinematics == 'holonomic':
                self.robots[i].step(ActionXY(actions[i][0], actions[i][1]))
            else:
                self.robots[i].step(ActionRot(actions[i][0]))

        for i, human_action in enumerate(human_actions):
            self.humans[i].step(human_action)

        self.global_time += self.time_step # max episode length=time_limit/time_step
        self.step_counter =self.step_counter+1

        info={'info':episode_info}

        # Add or remove at most self.human_num_range humans
        # if self.human_num_range == 0 -> human_num is fixed at all times
        if self.human_num_range > 0 and self.global_time % 5 == 0:
            # remove humans
            if np.random.rand() < 0.5:
                # if no human is visible, anyone can be removed
                if len(self.observed_human_ids) == 0:
                    max_remove_num = self.human_num - self.min_human_num
                else:
                    max_remove_num = min(self.human_num - self.min_human_num, (self.human_num - 1) - max(self.observed_human_ids))
                remove_num = np.random.randint(low=0, high=max_remove_num + 1)
                for _ in range(remove_num):
                    self.humans.pop()
                self.human_num = self.human_num - remove_num
                self.last_human_states = self.last_human_states[:self.human_num]
            # add humans
            else:
                add_num = np.random.randint(low=0, high=self.human_num_range + 1)
                if add_num > 0:
                    # set human ids
                    true_add_num = 0
                    for i in range(self.human_num, self.human_num + add_num):
                        if i == self.config.sim.human_num + self.human_num_range:
                            break
                        self.generate_random_human_position(human_num=1)
                        self.humans[i].id = i
                        true_add_num = true_add_num + 1
                    self.human_num = self.human_num + true_add_num
                    if true_add_num > 0:
                        self.last_human_states = np.concatenate((self.last_human_states, np.array([[15, 15, 0, 0, 0.3]]*true_add_num)), axis=0)

        assert self.min_human_num <= self.human_num <= self.max_human_num

        
        # update human  robot states in lidar2d
        self.lidar.update_dynamic_map([[human.px, human.py] for human in self.humans], [[robot.px, robot.py] for robot in self.robots])
        # compute the observation
        # get robot observation
        self.observed_human_ids = []

        obs=[self.generate_ob(i,reset=False, sort=self.config.args.sort_humans) for i in range(self.robot_num)]
        
        # broadcast robot's observation to other robots
        if self.config.robot.broadcast_open:
            obs = self.broadcast(obs)

        # save the observation for the render
        if self.config.sim.render:
            self.ob = obs[0]
        
        # Update all humans' goals randomly midway through episode
        if self.random_goal_changing:
            if self.global_time % 5 == 0:
                self.update_human_goals_randomly()

        # Update a specific human's goal once its reached its original goal
        if self.end_goal_changing:
            for i, human in enumerate(self.humans):



                if norm((human.gx - human.px, human.gy - human.py)) < human.radius:
                    if self.robots[0].kinematics == 'holonomic':
                        self.humans[i] = self.generate_circle_crossing_human()
                        self.humans[i].id = i
                    else:
                        self.update_human_goal(human)
        
        return obs, rewards_list, all_done, info


    # find R(s, a)
    # danger_zone: how to define the personal_zone (if the robot intrudes into this zone, the info will be Danger)
    # circle (traditional) or future (based on true future traj of humans)
    def calc_reward(self, robot_index, action, danger_zone='circle'):
        # collision detection
        dmin = float('inf')
        danger_dists = []
        collision = False
        obstacle_collision = False
        robot_collision = False
        robot_pos_x = int(self.static_map_size / 2 + np.floor(self.robots[robot_index].px / self.cell_length))
        robot_pos_y = int(self.static_map_size / 2 + np.floor(self.robots[robot_index].py / self.cell_length))

        # collision check with humans
        for i, human in enumerate(self.humans):
            dx = human.px - self.robots[robot_index].px
            dy = human.py - self.robots[robot_index].py
            closest_dist = (dx ** 2 + dy ** 2) ** (1 / 2) - human.radius - self.robots[robot_index].radius

            if closest_dist < self.discomfort_dist:
                danger_dists.append(closest_dist)
            if closest_dist < 0:
                collision = True
                break
            elif closest_dist < dmin:
                dmin = closest_dist
              
        if (0<=robot_pos_x<self.static_map_size) and (0<=robot_pos_y<self.static_map_size) and  self.original_map[robot_pos_x, robot_pos_y] > 0:
            obstacle_collision = True

        for i in range(self.robot_num):
            if i == robot_index:
                continue
            dx = self.robots[i].px - self.robots[robot_index].px
            dy = self.robots[i].py - self.robots[robot_index].py
            closest_dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.robots[robot_index].radius - self.robots[i].radius
            if closest_dist <= 0:
                robot_collision = True
                break

        # check if reaching the goal
        if self.robots[robot_index].kinematics == 'unicycle':
            goal_radius = 0.6
        else:
            goal_radius = self.robots[robot_index].radius

        reaching_goal = norm(
            np.array(self.robots[robot_index].get_position()) - np.array(self.robots[robot_index].get_goal_position())) < goal_radius

        # use danger_zone to determine the condition for Danger
        if danger_zone == 'circle' or self.phase == 'train':
            danger_cond = dmin < self.discomfort_dist
            min_danger_dist = 0
        else:
            #enter here
            # if the robot collides with future states, give it a collision penalty
            relative_pos = self.human_future_traj[1:, :, :2] - np.array([self.robots[robot_index].px, self.robots[robot_index].py])
            relative_dist = np.linalg.norm(relative_pos, axis=-1)

            collision_idx = relative_dist < self.robots[robot_index].radius + self.config.humans.radius  # [predict_steps, human_num]

            danger_cond = np.any(collision_idx)
            # if robot is dangerously close to any human, calculate the min distance between robot and its closest human
            if danger_cond:
                min_danger_dist = np.amin(relative_dist[collision_idx])
            else:
                min_danger_dist = 0
        
        if self.global_time >= self.time_limit - 1:
            reward = -0.01
            done = True
            episode_info = Timeout()
        elif reaching_goal:
            if self.robots[robot_index].deactivated:
                return 0, False, ReachGoal()
            else:
                reward = self.success_reward
                done = False
                self.robots[robot_index].deactivated = True
                episode_info = ReachGoal()
        elif collision:
            if self.robots[robot_index].deactivated:
                return 0, False, Collision()
            else:
                reward = self.collision_penalty
                done = True
                self.robots[robot_index].deactivated = True
                episode_info = Collision()
        elif obstacle_collision:
            if self.robots[robot_index].deactivated:
                return 0, False, Collision()
            else:
                reward = -20
                done = True
                self.robots[robot_index].deactivated = True
                episode_info = Collision()
        elif robot_collision:
            if self.robots[robot_index].deactivated:
                return 0, False, Collision()
            else:
                reward = -40
                done = True
                self.robots[robot_index].deactivated = True
                episode_info = Collision()
        elif danger_cond:
            if self.robots[robot_index].deactivated:
                return 0, False, Collision()
            # only penalize agent for getting too close if it's visible
            # adjust the reward based on FPS
            # print(dmin)
            reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            done = False
            episode_info = Danger(min_danger_dist)
        else:
            if self.robots[robot_index].deactivated:
                return 0, False, Collision()
            
            done = False
            episode_info = Nothing()

            # use pot_factor calculate the potential reward
            pot_factor = 2 if self.robots[robot_index].kinematics == 'holonomic' else 3

            # potential reward
            # check if cross obstacle
            cross=True
            x1, y1, x2, y2= self.robots[robot_index].px, self.robots[robot_index].py, self.robots[robot_index].gx, self.robots[robot_index].gy
            x1=int(self.static_map_size / 2 + x1 / self.cell_length)
            x2=int(self.static_map_size / 2 + x2 / self.cell_length)
            y1=int(self.static_map_size / 2 + y1 / self.cell_length)
            y2=int(self.static_map_size / 2 + y2 / self.cell_length)

            if not (x1 < 0 or x1 >= self.static_map_size or y1 < 0 or y1 >= self.static_map_size or x2 < 0 or x2 >= self.static_map_size or y2 < 0 or y2 >= self.static_map_size ):
                #Get the line between state1 and state2 in grid space
                rr, cc = line(x1, y1, x2, y2)
                # Check if any of the coordinates cross an obstacle
                if np.any(self.original_map[rr, cc] >= 0.9):                    
                    cross=False             
                    if self.obst_directions[robot_index][0] == 0 and self.obst_directions[robot_index][1] == 0:
                        pot_factor = pot_factor / 3
                    else:
                        pot_factor = pot_factor / 10

            potential_cur = np.linalg.norm(
                np.array([self.robots[robot_index].px, self.robots[robot_index].py]) - np.array(self.robots[robot_index].get_goal_position()))
            reward = pot_factor * (-abs(potential_cur) - self.potential[robot_index])
            self.potential[robot_index] = -abs(potential_cur)

            # calculate the angular difference between the robot's heading direction and the velocity direction
            if self.robots[robot_index].kinematics == 'holonomic':
                former_angle = np.array([self.robots[robot_index].vx, self.robots[robot_index].vy], dtype=np.float32)
                former_v = np.linalg.norm(former_angle)
                new_v = np.linalg.norm(action)
                omega = 0 if (former_v == 0 or new_v == 0) else np.arccos(np.clip(np.dot(former_angle, action) / (former_v * new_v), -1, 1))

                # add a rotational penalty
                if omega + new_v > self.robots[robot_index].v_pref:
                    reward -= 0.08 * (omega + new_v - self.robots[robot_index].v_pref)

                if not cross:
                    reward += 0.15 * np.clip(np.cross(action, self.obst_directions[robot_index]),-0.1,1)
            
            else: 
                omega = (np.arctan2(self.robots[robot_index].gy - self.robots[robot_index].py, self.robots[robot_index].gx - self.robots[robot_index].px)
                          - self.robots[robot_index].theta + np.pi) % (2 * np.pi) - np.pi
                pot_reward = 0
                pot_reward += pot_factor * omega * action[0]
                

                if self.obst_directions[robot_index][0] <= 0.1 :
                    x_dif = -np.abs(action[0])/10 + 0.01
                else: 
                    d_theta = action[0] * 2
                    x_dif = - np.abs(self.obst_directions[robot_index][0] * np.cos(d_theta) + self.obst_directions[robot_index][1] * np.sin(d_theta)) / 20
                #if (self.obst_directions[robot_index][0] != 0 or self.obst_directions[robot_index][1] != 0) :
                    #rd = (np.arctan2(self.obst_directions[robot_index][1], self.obst_directions[robot_index][0]) + np.pi / 2)% (2 * np.pi) - np.pi
                    #pot_reward += 0.3* (np.clip(np.dot(np.array([-0.3,0.85]), self.obst_directions[robot_index]),-0.1,1) - 0.5)
                
                # if robot_index == 0:
                #     print('x_dif:',x_dif)
                if dmin < 3 :
                    human_panalty = 0.1 / (dmin + 0.1)
                    reward -= human_panalty
                else:
                    reward -= pot_factor * 0.1 * abs(omega)
                reward += pot_reward
                reward += x_dif
                
                # print(self.robots[robot_index].gx - self.robots[robot_index].px, self.robots[robot_index].gy - self.robots[robot_index].py,np.arctan2(self.robots[robot_index].gx - self.robots[robot_index].px, self.robots[robot_index].gy - self.robots[robot_index].py)% (2 * np.pi))

        # if the robot is near collision/arrival, it should be able to turn a large angle
        # if self.robots[robot_index].kinematics == 'unicycle':
        #     # add a rotational penalty
        #     r_spin = -4.5 * action[1] ** 2
            
        #     # add a penalty for going backwards
        #     if action[0] < 0:
        #         r_back = -2 * abs(action[0])
        #     else:
        #         r_back = 0.

        #     reward = reward + r_spin + r_back

        return reward, done, episode_info


    def convert_to_3_channel_bitmap(self,input_nparray):

        # Input tensor shape: (batch_size, 2, 32, 32)
        _, height, width = input_nparray.shape

        # Initialize the output tensor with zeros, shape: (batch_size, 3, 32, 32)
        output_tensor = np.zeros((3, height, width))

        # Extract the two channels from the input tensor
        channel_0 = input_nparray[0]  # Shape: (batch_size, 32, 32)
        channel_1 = input_nparray[1]  # Shape: (batch_size, 32, 32)

        # Create the 3-channel output
        for c in range(3):
            mask = (channel_1 == c+1)  # Create a mask where channel_1 equals the current channel index
            output_tensor[c] = channel_0 * mask  # Assign channel_0 values to the corresponding channel in the output tensor
        
        return output_tensor


    def render(self, ogm_for_vis,mode='human'):
        #print('enter crowd_sim_var_num/render')
        
        # change render to 2 robots
        """ Render the current status of the environment using matplotlib """


        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

        robot_color = 'gold'
        goal_color = 'red'
        arrow_color = 'red'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        def calcFOVLineEndPoint(ang, point, extendFactor):
            # choose the extendFactor big enough
            # so that the endPoints of the FOVLine is out of xlim and ylim of the figure
            FOVLineRot = np.array([[np.cos(ang), -np.sin(ang), 0],
                                   [np.sin(ang), np.cos(ang), 0],
                                   [0, 0, 1]])
            point.extend([1])
            # apply rotation matrix
            newPoint = np.matmul(FOVLineRot, np.reshape(point, [3, 1]))
            # increase the distance between the line start point and the end point
            newPoint = [extendFactor * newPoint[0, 0], extendFactor * newPoint[1, 0], 1]
            return newPoint
        

        ax=self.render_axis
        artists=[]
        arrowStartEnd=[]
        texts=[]
        robot_num = 0
        
       
        if not self.map_drawed:
            # clear the previous map
            for artist in self.map_artists:
                artist.remove()
            self.map_artists.clear()
            self.map_drawed = True
            for i in range(self.static_map_size):
                for j in range(self.static_map_size):
                    if self.original_map[i, j] > 0:
                        m = plt.Rectangle(((i - self.static_map_size / 2)*self.cell_length  , (j - self.static_map_size / 2)*self.cell_length ), self.cell_length, self.cell_length, fill=True, facecolor='black', alpha=0.5* self.original_map[i,j], linewidth=0, edgecolor='none',)
                        ax.add_artist(m)
                        self.map_artists.append(m)



        # add goals and robots
        draw_fov_once=False
        for robot in self.robots:
            # write robot
            robot_num += 1
            if robot.deactivated :
                if not np.linalg.norm(np.array([robot.px, robot.py]) - np.array([robot.gx, robot.gy])) <= robot.radius:
                    texts.append(plt.text(robot.px -0.16, robot.py -0.23, str(robot_num), color='grey' ,fontsize=12))
                    texts.append(plt.text(robot.gx -0.16, robot.gy -0.27, 'X', color='black' ,fontsize=16))
                    texts.append(plt.text(robot.px -0.19, robot.py -0.27, 'X', color='black' ,fontsize=16))
                else:
                    texts.append(plt.text(robot.px -0.16, robot.py -0.23, str(robot_num), color='yellow' ,fontsize=12))
            else:
                texts.append(plt.text(robot.px -0.16, robot.py -0.23, str(robot_num), color='white' ,fontsize=12))
            # draw link between robot and goal
            link = mlines.Line2D([robot.px, robot.gx], [robot.py, robot.gy], color='grey', marker = '*', markerfacecolor = 'yellow', linestyle=':', markersize=12, label='Goal')
            ax.add_artist(link)
            artists.append(link)

            # add an arc of robot's sensor range
            #sensor_range = plt.Circle(robot.get_position(), robot.sensor_range + robot.radius+self.config.humans.radius, fill=False, linestyle='--')
            sensor_range = plt.Rectangle(
                (robot.px - robot.sensor_range * (np.cos(robot.theta)-np.sin(robot.theta)) , 
                 robot.py - robot.sensor_range* (np.cos(robot.theta)+np.sin(robot.theta))), 
                2 * robot.sensor_range, 2 * robot.sensor_range, 
                fill=False, linestyle='--',color = (0,0,0,0.1), angle=np.degrees(robot.theta))
            
            ax.add_artist(sensor_range)
            artists.append(sensor_range)

            # compute orientation in each step and add arrow to show the direction
            radius = robot.radius         
            robot_theta = robot.theta if robot.kinematics == 'unicycle' else np.arctan2(robot.vy, robot.vx)
            arrowStartEnd.append(((robot.px, robot.py), (robot.px +  3 * robot.vx, robot.py +  3 * robot.vy)))

            # draw FOV for the robot
            # add robot FOV
            # !!! hanot been tested
            # robot.FOV = np.pi / 2
            if robot.FOV < 2 * np.pi:
                if not draw_fov_once:
                    draw_fov_once = True
                    FOVAng = robot.FOV / 2
                    FOVLine1 = mlines.Line2D([0, 0], [0, 0], linestyle='-.')
                    FOVLine2 = mlines.Line2D([0, 0], [0, 0], linestyle='-.')

                    endPointX = robot.px + radius * np.cos(robot_theta)
                    endPointY = robot.py + radius * np.sin(robot_theta)

                    # transform the vector back to world frame origin, apply rotation matrix, and get end point of FOVLine
                    # the start point of the FOVLine is the center of the robot
                    FOVEndPoint1 = calcFOVLineEndPoint(FOVAng, [endPointX - robot.px, endPointY - robot.py], 20. / radius)
                    FOVLine1.set_xdata(np.array([robot.px, robot.px + FOVEndPoint1[0]]))
                    FOVLine1.set_ydata(np.array([robot.py, robot.py + FOVEndPoint1[1]]))
                    FOVEndPoint2 = calcFOVLineEndPoint(-FOVAng, [endPointX - robot.px, endPointY - robot.py], 20. / radius)
                    FOVLine2.set_xdata(np.array([robot.px, robot.px + FOVEndPoint2[0]]))
                    FOVLine2.set_ydata(np.array([robot.py, robot.py + FOVEndPoint2[1]]))

                    ax.add_artist(FOVLine1)
                    ax.add_artist(FOVLine2)
                    artists.append(FOVLine1)
                    artists.append(FOVLine2)
        
        # draw obst_direction
        for r in range(self.robot_num):
            if self.obst_directions[r] is not None and (self.obst_directions[r][0] != 0 or self.obst_directions[r][1] != 0):
                # draw a arrow to show the direction of the obstacle
                #arrowStartEnd.append(((self.robots[r].px, self.robots[r].py), (self.robots[r].px + self.obst_directions[r][0]*np.cos(self.robots[r].theta) - self.obst_directions[r][1]*np.sin(self.robots[r].theta), self.robots[r].py + self.obst_directions[r][0]*np.sin(self.robots[r].theta) + self.obst_directions[r][1]*np.cos(self.robots[r].theta))))
                
                d_t = 0.5 * 3
                x_dif = self.obst_directions[r][0] - (self.obst_directions[r][0] * np.cos(d_t) + self.obst_directions[r][1] * np.sin(d_t))

                # texts.append(plt.text(self.robots[r].px + 0.5, self.robots[r].py + 1,
                #             str(x_dif), # str( np.clip(np.arctan2(self.obst_directions[r][1], self.obst_directions[r][0])/np.pi,-0.5,0.5)% 2 - 0.5),#0.3* (np.clip(np.dot(np.array([-0.3,0.85]), self.obst_directions[r]),-0.1,1) - 0.6)),
                #             color='black', fontsize=12))
            
        # print(ogm_for_vis)
        
        # Convert numpy array to a PyTorch tensor
        chan_0 = torch.from_numpy(self.ob['occupancy_map'][0])
        # Convert tensor to float
        chan_0 = chan_0.float()
        chan_0=ogm_for_vis[0,0,0,:,:].cpu().float()
        self.ob['occupancy_map'][1]=ogm_for_vis[0,0,1,:,:].cpu()
        

        # print(self.ob['occupancy_map'][1])
        # exit()
        #self.ob['occupancy_map'] = self.convert_to_3_channel_bitmap(self.ob['occupancy_map'])
        for i in range(self.map_size):
            c=0
            for j in range(self.map_size):
                global_x =  self.robots[0].px + (i - self.map_size / 2)*self.cell_length*np.cos(self.robots[0].theta) - (j - self.map_size / 2)*self.cell_length*np.sin(self.robots[0].theta)
                global_y =  self.robots[0].py + (i - self.map_size / 2)*self.cell_length*np.sin(self.robots[0].theta) + (j - self.map_size / 2)*self.cell_length*np.cos(self.robots[0].theta)
                # no negative? 
                thes=0.1 * torch.max(chan_0)
                
               
                if chan_0[i, j]> thes:
                    alpha_value = chan_0[i, j].item()
                    ogm = plt.Rectangle((global_x, global_y), self.cell_length, self.cell_length, fill=True, facecolor='black', alpha=alpha_value, linewidth=0, edgecolor='none',)
                    ax.add_artist(ogm)
                    artists.append(ogm)
                    c+=1
                # if self.ob['occupancy_map'][1][i, j] > 0:
                    
                #     ogm = plt.Rectangle((global_x, global_y), self.cell_length, self.cell_length, fill=True, facecolor='pink', alpha=self.ob['occupancy_map'][1][i, j].clip(0,1)*0.8, linewidth=0, edgecolor='none',)
                #     ax.add_artist(ogm)
                #     artists.append(ogm)
                #print('occ_percentage : {}'.format(c/(self.map_size*self.map_size)))

                
                # if self.ob['occupancy_map'][2][i, j] > 0:
                #     ogm = plt.Rectangle((global_x, global_y), self.cell_length, self.cell_length, fill=True, facecolor='pink', alpha=self.ob['occupancy_map'][2][i, j]*0.8, linewidth=0, edgecolor='none',)
                #     ax.add_artist(ogm)
                #     artists.append(ogm)

        # add arrow of humans to show the direction
        for i, human in enumerate(self.humans):
            theta = np.arctan2(human.vy, human.vx)
            arrowStartEnd.append(((human.px, human.py), (human.px + radius * np.cos(theta), human.py + radius * np.sin(theta))))        
        
        # draw robots on the map
        robots_mark=mlines.Line2D([robotx.px for robotx in self.robots], [roboty.py for roboty in self.robots], color='red',marker = 'o', linestyle='None', markersize=15, label='Goal')
        ax.add_artist(robots_mark)
        artists.append(robots_mark)

        # draw arrows on the map
        arrows = [patches.FancyArrowPatch(*arrow, color=arrow_color, arrowstyle=arrow_style)
                for arrow in arrowStartEnd]
        for arrow in arrows:
            ax.add_artist(arrow)
            artists.append(arrow)       


        # add humans and change the color of them based on visibility
        human_circles = [plt.Circle(human.get_position(), human.radius, fill=False, linewidth=1.5) for human in self.humans]
        



        actual_arena_size = self.arena_size + 0.5
        for i in range(len(self.humans)):
            ax.add_artist(human_circles[i])
            artists.append(human_circles[i])

            # green: visible; red: invisible
            # if self.detect_visible(self.robot, self.humans[i], robot1=True):
            if self.human_visibility[i]:
                human_circles[i].set_color(c='g')
            else:
                human_circles[i].set_color(c='r')


            # for j in range(len(self.robots)):
            for j in range(1):
                if self.humans[i].id in self.observed_human_ids[j]:
                    human_circles[i].set_color(c='b')

            texts.append(plt.text(self.humans[i].px - 0.1, self.humans[i].py - 0.1, str(self.humans[i].id), color='black', fontsize=12))

        sensor_range = plt.Rectangle(
            (self.robots[0].px - self.robots[0].sensor_range * (np.cos(self.robots[0].theta)-np.sin(self.robots[0].theta)) , 
                self.robots[0].py - self.robots[0].sensor_range* (np.cos(self.robots[0].theta)+np.sin(self.robots[0].theta))), 
            2 * self.robots[0].sensor_range, 2 * self.robots[0].sensor_range, 
            fill=False, linestyle='--',color = (0,0,0,0.8), angle=np.degrees(self.robots[0].theta))
        
        ax.add_artist(sensor_range)
        artists.append(sensor_range)

        
        # 保存当前帧
        # if self.frame_count<50:
        #     #print(self.frame_count)
        #     frame_path = os.path.join('frames', f"frame_{self.frame_count:04d}.png")
        #     plt.savefig(frame_path)
        #     self.frame_count += 1
        
        plt.pause(0.01)
        for item in artists:
            item.remove() # there should be a better way to do this. For example,
            # initially use add_artist and draw_artist later on
        for t in texts:
            t.remove()
        

