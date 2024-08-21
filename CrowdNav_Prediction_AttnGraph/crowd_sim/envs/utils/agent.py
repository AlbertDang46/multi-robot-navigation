import numpy as np
from numpy.linalg import norm
import abc
import logging
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.action import ActionXY, ActionRot
from crowd_sim.envs.utils.state import ObservableState, FullState


class Agent(object):
    def __init__(self, config, section):
        """
        Base class for robot and human. Have the physical attributes of an agent.

        """
        subconfig = config.robot if section == 'robot' else config.humans
        self.visible = subconfig.visible
        self.v_pref = subconfig.v_pref
        self.radius = subconfig.radius
        # randomize neighbor_dist of ORCA
        if config.env.randomize_attributes:
            config.orca.neighbor_dist = np.random.uniform(5, 10)
        self.policy = policy_factory[subconfig.policy](config)
        self.sensor = subconfig.sensor
        self.FOV = np.pi * subconfig.FOV
        # for humans: we only have holonomic kinematics; for robot: depend on config
        self.kinematics = 'holonomic' if section == 'humans' else config.action_space.kinematics
        self.px = None
        self.py = None
        self.gx = None
        self.gy = None
        self.subgx = None
        self.subgy = None
        self.vx = None
        self.vy = None
        self.theta = None
        self.time_step = config.env.time_step
        self.policy.time_step = config.env.time_step
        self.occpancy_shape = None


    def print_info(self):
        logging.info('Agent is {} and has {} kinematic constraint'.format(
            'visible' if self.visible else 'invisible', self.kinematics))


    def sample_random_attributes(self):
        """
        Sample agent radius and v_pref attribute from certain distribution
        :return:
        """
        self.v_pref = np.random.uniform(0.5, 1.5)
        self.radius = np.random.uniform(0.3, 0.5)

    def set(self, px, py, gx, gy, vx, vy, theta, radius=None, v_pref=None):
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        self.subgx = gx
        self.subgy = gy
        self.theta = theta

        if radius is not None:
            self.radius = radius
        if v_pref is not None:
            self.v_pref = v_pref

    # self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta
    def set_list(self, px, py, vx, vy, radius, gx, gy, v_pref, theta):
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        self.theta = theta
        self.radius = radius
        self.v_pref = v_pref

    def get_observable_state(self):
        return ObservableState(self.px, self.py, self.vx, self.vy, self.radius)

    def get_observable_state_list(self):
        return [self.px, self.py, self.vx, self.vy, self.radius]

    def get_observable_state_list_noV(self):
        return [self.px, self.py, self.radius]

    def get_next_observable_state(self, action):
        self.check_validity(action)
        pos = self.compute_position(action, self.time_step)
        next_px, next_py = pos
        if self.kinematics == 'holonomic':
            next_vx = action.vx
            next_vy = action.vy
        else:
            next_theta = self.theta + action.r
            next_vx = action.v * np.cos(next_theta)
            next_vy = action.v * np.sin(next_theta)
        return ObservableState(next_px, next_py, next_vx, next_vy, self.radius)

    def get_full_state(self):
        return FullState(self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta)

    def get_full_state_list(self):
        return [self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta]

    def get_full_state_list_noV(self):

        omega = (np.arctan2(self.gy - self.py, self.gx - self.px)
                          - self.theta + np.pi) % (2 * np.pi) - np.pi
        return [self.px, self.py, self.radius, 
            np.linalg.norm([self.gy - self.py, self.gx - self.px]), omega, self.v_pref, self.theta, self.vx, self.vy]
        # return [self.px, self.py, self.radius, self.gx, self.gy, self.v_pref, self.theta, self.vx, self.vy]
        # return [self.px, self.py, self.radius, self.gx, self.gy, self.v_pref]

    def get_position(self):
        return self.px, self.py

    def set_position(self, position):
        self.px = position[0]
        self.py = position[1]

    def get_goal_position(self):
        return self.gx, self.gy

    def get_velocity(self):
        return self.vx, self.vy


    def set_velocity(self, velocity):
        self.vx = velocity[0]
        self.vy = velocity[1]


    @abc.abstractmethod
    def act(self, ob):
        """
        Compute state using received observation and pass it to policy

        """
        return

    def check_validity(self, action):
        if self.kinematics == 'holonomic':
            assert isinstance(action, ActionXY)
        else:
            assert isinstance(action, ActionRot)

    def compute_position(self, action, delta_t):
        self.check_validity(action)
        if self.kinematics == 'holonomic':
            px = self.px + action.vx * delta_t
            py = self.py + action.vy * delta_t
        # unicycle
        else:
            # naive dynamics
            # theta = self.theta + action.r * delta_t # if action.r is w
            # # theta = self.theta + action.r # if action.r is delta theta
            # px = self.px + np.cos(theta) * action.v * delta_t
            # py = self.py + np.sin(theta) * action.v * delta_t

            # differential drive
            epsilon = 0.0001
            if abs(action.v) < epsilon:
                d_theta = 0
                d_l = self.v_pref * delta_t
            else:
                d_theta = action.v / 1.5 / self.radius * delta_t
                self.theta = (self.theta + d_theta) % (2 * np.pi)
                d_l = (1 - np.abs(action.v)) * delta_t *self.v_pref

                #print('action.v, d_theta, d_l:', action.v, d_theta, d_l)

            self.vx = d_l * np.cos(self.theta) / delta_t
            self.vy = d_l * np.sin(self.theta) / delta_t

            px = self.px + d_l * np.cos(self.theta)
            py = self.py + d_l * np.sin(self.theta)

        return px, py

    def step(self, action):
        """
        Perform an action and update the state
        """
        self.check_validity(action)
        pos = self.compute_position(action, self.time_step)
        self.px, self.py = pos
        if self.kinematics == 'holonomic':
            self.vx = action.vx
            self.vy = action.vy
        

    def one_step_lookahead(self, pos, action):
        px, py = pos
        self.check_validity(action)
        new_px = px + action.vx * self.time_step
        new_py = py + action.vy * self.time_step
        new_vx = action.vx
        new_vy = action.vy
        return [new_px, new_py, new_vx, new_vy]

    def reached_destination(self):
        return norm(np.array(self.get_position()) - np.array(self.get_goal_position())) < self.radius

