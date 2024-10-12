import os
import shutil
import time
#from test import test
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from rl import ppo
from rl.networks import network_utils
from arguments import get_args
from rl.networks.envs import make_vec_envs
from rl.networks.model import Policy
from rl.networks.storage import RolloutStorage
from rl.evaluation import reset_folder, create_gif_from_frames


from crowd_nav.configs.config import Config
from crowd_sim import *
import copy
import wandb
import os
from crowd_sim.envs.utils.info import *
from create_map import create_new_map

FUTURE_STEP=4

def main():
	"""
	main function for training a robot policy network
	"""
	
	# read arguments
	algo_args = get_args()

	# create a directory for saving the logs and weights
	if not os.path.exists(algo_args.output_dir):
		os.makedirs(algo_args.output_dir)
	# if output_dir exists and overwrite = False
	elif not algo_args.overwrite:
		raise ValueError('output_dir already exists!')
	

	save_config_dir = os.path.join(algo_args.output_dir, 'configs')
	if not os.path.exists(save_config_dir):
		os.makedirs(save_config_dir)
	shutil.copy('crowd_nav/configs/config.py', save_config_dir)
	shutil.copy('crowd_nav/configs/__init__.py', save_config_dir)
	shutil.copy('arguments.py', algo_args.output_dir)

	create_new_map()


	env_config = config = Config()

	torch.manual_seed(algo_args.seed)
	torch.cuda.manual_seed_all(algo_args.seed)
	if algo_args.cuda:
		if algo_args.cuda_deterministic:
			# reproducible but slower
			torch.backends.cudnn.benchmark = False
			torch.backends.cudnn.deterministic = True
		else:
			# not reproducible but faster
			torch.backends.cudnn.benchmark = True
			torch.backends.cudnn.deterministic = False



	torch.set_num_threads(algo_args.num_threads)
	device = torch.device("cuda" if algo_args.cuda else "cpu")


	env_name = algo_args.env_name

	if config.sim.render:
		algo_args.num_processes = 1
		algo_args.num_mini_batch = 1

	# Assuming this is part of a larger code block where `config.sim.render` is defined
	if config.sim.render:
		fig, ax = plt.subplots(figsize=(7, 7))
		ax.set_xlim(-10, 10)
		ax.set_ylim(-10, 10)
		ax.set_xlabel('x(m)', fontsize=16)
		ax.set_ylabel('y(m)', fontsize=16)

		plt.ion()
		plt.show()
	else:
		ax = None
		 
	# wandb_log = not config.sim.render
	# if wandb_log:
	# 	wandb.init(project="smooth_action_space ", name=algo_args.output_dir.split("/")[-1], config=vars(algo_args))

	# Reset the frames folder if rendering
	if config.sim.render:
		reset_folder('frames')
		reset_folder('train_visualization')
		gif_num = 0

	# Create a wrapped, monitored VecEnv
	envs = make_vec_envs(env_name, algo_args.seed, algo_args.num_processes,
						 algo_args.gamma, None, device, False, config=env_config, ax=ax, pretext_wrapper=config.env.use_wrapper)
	

	# create a policy network
	actor_critic = Policy(
		envs.observation_space.spaces, # pass the Dict into policy to parse
		envs.action_space,
		base_kwargs=algo_args)

	# storage buffer to store the agent's experience
	rollouts = RolloutStorage(algo_args.num_steps,
							  algo_args.num_processes,
							  envs.observation_space.spaces,
							  envs.action_space,
							  algo_args.rnn_hidden_size,
							  algo_args.human_human_edge_rnn_size
							  )
	
	# create a rollout for each robot
	all_rollouts = []
	for i in range(config.sim.robot_num):
		all_rollouts.append(copy.deepcopy(rollouts))

	# continue training from an existing model if resume = True
	if algo_args.resume:
		load_path = config.training.load_path
		actor_critic.load_state_dict(torch.load(load_path))
		print("Loaded the following checkpoint:", load_path)


	# allow the usage of multiple GPUs to increase the number of examples processed simultaneously
	nn.DataParallel(actor_critic).to(device)


	# create the ppo optimizer
	agent = ppo.PPO(
		actor_critic,
		algo_args.clip_param,
		algo_args.ppo_epoch,
		algo_args.num_mini_batch,
		algo_args.value_loss_coef,
		algo_args.entropy_coef,
		lr=algo_args.lr,
		eps=algo_args.eps,
		max_grad_norm=algo_args.max_grad_norm)

	obs = envs.reset()
	
	#original	
	if isinstance(obs, dict):
		for key in obs:	
			for i in range(obs[key].shape[0]):
					
					rollouts.obs[key][0][i].copy_(obs[key][i][0])
	else:
		rollouts.obs[0].copy_(obs)
	#change to multi-agent rollout
	
	for r in range(config.sim.robot_num):
		if isinstance(obs, dict):
			for key in obs:	
				for i in range(obs[key].shape[0]):
					all_rollouts[r].obs[key][0][i].copy_(obs[key][i][r])
		else:
			all_rollouts[r].obs[0].copy_(obs)


	for r in range(config.sim.robot_num):
		all_rollouts[r].to(device)

	episode_rewards = deque(maxlen=100)

	start = time.time()
	num_updates = int(
		algo_args.num_env_steps) // algo_args.num_steps // algo_args.num_processes
	
	# for each robot, store the observation
	all_obs = []
	for r in range(config.sim.robot_num):
		single_obs = {}
		for keyy in obs.keys():	
			single_obs[keyy] = []
			for i in range(obs[keyy].shape[0]):
				single_obs[keyy].append(obs[keyy][i][r])
			single_obs[keyy] = torch.stack(single_obs[keyy], dim = 0)
		all_obs.append(single_obs)

	
	# for each robot, store the initial hidden state
	init_hidden_states = {}
	for key in all_rollouts[0].recurrent_hidden_states:
		init_hidden_states[key] = all_rollouts[0].recurrent_hidden_states[key][0]
	all_hid_states=[init_hidden_states for _ in range(config.sim.robot_num)]

	successful_lidar_seq= []
	successful_vel_pos_seq=[]
	future_seq=[]
	# start the training loop
	for j in range(num_updates):
		# schedule learning rate if needed
		if algo_args.use_linear_lr_decay:
			network_utils.update_linear_schedule(
				agent.optimizer, j, num_updates,
				agent.optimizer.lr if algo_args.algo == "acktr" else algo_args.lr)
		
		# step the environment for a few times
		current_episode_lidar=[]
		current_episode_vel_pos=[]
		current_episode_future_lidar=[]
		add=0
		pos_deque={}
		for step in range(algo_args.num_steps):
			# Sample actions
			all_actions = []
			all_values = []
			all_log_probs = []
			all_lidars=[]
			all_vel_pos=[]
			with torch.no_grad():
				# get the action for each robot				
				for i in range(config.sim.robot_num):
					value_i, action_i, log_i, recurrent_hidden_states_i,ogm_for_vis_i,lidar_i,vel_pos_i = actor_critic.act(
						all_obs[i], all_hid_states[i],
						all_rollouts[i].masks[step],i)
					if i==0:
						ogm_for_vis=ogm_for_vis_i
					all_values.append(value_i)
					all_log_probs.append(log_i)
					all_actions.append(action_i)					
					all_hid_states[i]= copy.deepcopy(recurrent_hidden_states_i)	
					all_lidars.append(lidar_i)
					all_vel_pos.append(vel_pos_i)
					if i not in pos_deque:
						pos_deque[i]=deque(maxlen=FUTURE_STEP+1)
						
					pos_deque[i].append(vel_pos_i[0,0,0,[0,1,8]])# b 1 3			
				all_actions = torch.stack(all_actions, dim=1)
				all_lidars=torch.stack(all_lidars, dim=1).squeeze()
				all_vel_pos=torch.stack(all_vel_pos, dim=1).squeeze()
			current_episode_lidar.append(all_lidars.cpu().numpy())
			current_episode_vel_pos.append(all_vel_pos.cpu().numpy())
			current_episode_future_lidar.append([])


			# if config.sim.render:
			# 	envs.render(ogm_for_vis)
			obs, rewards, done, infos= envs.step(all_actions)
			if config.sim.render:
				# use render to collect data
				
				all_past_lidar=envs.render(pos_deque)
				
				if all_past_lidar is not None:
					for t in range(all_past_lidar.shape[1]):
						current_episode_future_lidar[-(1+all_past_lidar.shape[1]-t)].append(all_past_lidar[:,t,:,:])
			
				
			for r in range(config.sim.robot_num):
				single_obs = {}
				for keyy in obs.keys():	
					single_obs[keyy] = []
					for i in range(obs[keyy].shape[0]):
						single_obs[keyy].append(obs[keyy][i][r])
					single_obs[keyy] = torch.stack(single_obs[keyy], dim = 0)
				all_obs[r] = single_obs
			
			# If done then clean the history of observations.
			masks = torch.FloatTensor(
				[[0.0] if done_ else [1.0] for done_ in done])
			bad_masks = torch.FloatTensor(
				[[0.0] if 'bad_transition' in info.keys() else [1.0]
				 for info in infos])
			for info in infos:
				if 'episode' in info.keys():
					episode_rewards.append(info['episode']['r'])
					# if config.sim.render:
					# 	create_gif_from_frames('frames', os.path.join('train_visualization', 'episode_%d.gif' % gif_num))
					# 	gif_num += 1
				if 'info' in info.keys():
					if isinstance(info['info'],ReachGoal):
						add=1
			if not masks:
				break # if reset then break

			#change to multi-agent rollout insert
			for robot_index in range(config.sim.robot_num):
				single_obs = {}
				for keyy in obs.keys():	
					single_obs[keyy] = []
					for i in range(obs[keyy].shape[0]):
						single_obs[keyy].append(obs[keyy][i][robot_index])
					single_obs[keyy] = torch.stack(single_obs[keyy], dim = 0)
				masks = torch.FloatTensor([[0.0] if (done[e] or rewards[e][robot_index] == 0) else [1.0] for e in range(len(done))])

				all_rollouts[robot_index].insert(single_obs, all_hid_states[robot_index], torch.stack([all_action[robot_index] for all_action in all_actions]),
							all_log_probs[robot_index], all_values[robot_index], torch.tensor(rewards[:,robot_index:robot_index+1]), masks, bad_masks)
		cur_future_seq = []
		for seq in current_episode_future_lidar:
			# Check if the sequence is shorter than FUTURE_STEP, and pad if necessary	
			if len(seq) == FUTURE_STEP:
				cur_future_seq.append(np.array(seq))

		# Convert the padded sequences to a NumPy array
		cur_future_seq= np.array(cur_future_seq)
		max_len=algo_args.num_steps-FUTURE_STEP
		if len(current_episode_lidar)==30:
			current_episode_lidar=np.array(current_episode_lidar[:len(cur_future_seq)])
			current_episode_vel_pos=np.array(current_episode_vel_pos[:len(cur_future_seq)])
			successful_lidar_seq.append(current_episode_lidar)
			successful_vel_pos_seq.append(current_episode_vel_pos)
			future_seq.append(cur_future_seq)
			print(len(current_episode_lidar),len(current_episode_vel_pos),len(cur_future_seq))
		with torch.no_grad():
			#change to multi-agent rollout update
			all_rollouts_obs=[{} for _ in range(config.sim.robot_num)]
			all_rollouts_hidden_s = [{} for _ in range(config.sim.robot_num)]
			for robot_index in range(config.sim.robot_num):
			
				for key in all_rollouts[robot_index].obs:
					all_rollouts_obs[robot_index][key] = all_rollouts[robot_index].obs[key][-1]
					
				for key in all_rollouts[robot_index].recurrent_hidden_states:
					all_rollouts_hidden_s[robot_index][key] = all_rollouts[robot_index].recurrent_hidden_states[key][-1]

				next_value = actor_critic.get_value(
					all_rollouts_obs[robot_index], all_rollouts_hidden_s[robot_index],
					all_rollouts[robot_index].masks[-1],robot_index).detach()
				
		mean_action_loss=0	
		for robot_index in range(config.sim.robot_num):
			# compute advantage and gradient, and update the network parameters
			all_rollouts[robot_index].compute_returns(next_value, algo_args.use_gae, algo_args.gamma,
										algo_args.gae_lambda, algo_args.use_proper_time_limits)
			
			value_loss, action_loss, dist_entropy = agent.update(all_rollouts[robot_index],robot_index)

			mean_action_loss+=action_loss
			
			mean_action_loss/=config.sim.robot_num
			all_rollouts[robot_index].after_update()

		# save the model for every interval-th episode or for the last epoch
		if (j % algo_args.save_interval == 0
			or j == num_updates - 1) :
			save_path = os.path.join(algo_args.output_dir, 'checkpoints')
			if not os.path.exists(save_path):
				os.mkdir(save_path)   
   
			torch.save(actor_critic.state_dict(), os.path.join(save_path, '%.5i'%j + ".pt"))
			
		# output the training results
		if j % algo_args.log_interval == 0 and len(episode_rewards) > 1:
			total_num_steps = (j + 1) * algo_args.num_processes * algo_args.num_steps
			end = time.time()
			lidar_seq = np.array(successful_lidar_seq)
			np.save('3r3p_successful_lidar_seq_2.npy', lidar_seq)

			vel_pos_seq = np.array(successful_vel_pos_seq)
			np.save('3r3p_successful_vel_pos_seq_2.npy', vel_pos_seq)

			future_seq_=np.array(future_seq)
			np.save('3r3p_future_seq_2.npy', future_seq_)

			print("Saving successful actions and info mask",len(lidar_seq),len(vel_pos_seq),len(future_seq_))
			
			print(
				"Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward "
				"{:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
					.format(j, total_num_steps,
							int(total_num_steps / (end - start)),
							len(episode_rewards), np.mean(episode_rewards),
							np.median(episode_rewards), np.min(episode_rewards),
							np.max(episode_rewards), dist_entropy, value_loss,
							mean_action_loss))
			
			# log the training progress
			# if  wandb_log:
			# 	wandb.log({"median_reward":np.median(episode_rewards),
			# 			"mean_reward":np.mean(episode_rewards),
			# 			"min_reward":np.min(episode_rewards),
			# 			"max_reward":np.max(episode_rewards),
			# 			"dist_entropy":dist_entropy,
			# 			"value_loss":value_loss,
			# 			"action_loss":action_loss,})

			df = pd.DataFrame({'misc/nupdates': [j], 'misc/total_timesteps': [total_num_steps],
							   'fps': int(total_num_steps / (end - start)), 'eprewmean': [np.mean(episode_rewards)],
							   'loss/policy_entropy': dist_entropy, 'loss/policy_loss': action_loss,
							   'loss/value_loss': value_loss})
			if os.path.exists(os.path.join(algo_args.output_dir, 'progress.csv')) and j > 20:
				df.to_csv(os.path.join(algo_args.output_dir, 'progress.csv'), mode='a', header=False, index=False)
			else:
				df.to_csv(os.path.join(algo_args.output_dir, 'progress.csv'), mode='w', header=True, index=False)

			# create new map
			create_new_map()
			episode_rewards.clear()
	
	
	# if wandb_log:
	# 	wandb.finish()
			

if __name__ == '__main__':
	main()

