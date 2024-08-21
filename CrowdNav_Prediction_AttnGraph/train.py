import os
import shutil
import time
from test import test
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


from crowd_nav.configs.config import Config
from crowd_sim import *
import copy
import  wandb
import os
from crowd_sim.envs.utils.info import *
from create_map import create_new_map
from rl.evaluation import create_gif_from_frames,evaluate_training

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
		 
	wandb_log = not config.sim.render
	if wandb_log:
		wandb.init(project="smooth_action_space ",config={"human_num":config.sim.human_num,"robot_num":2})
	# Create a wrapped, monitored VecEnv
	envs = make_vec_envs(env_name, algo_args.seed, algo_args.num_processes,
						 algo_args.gamma, None, device, False, config=env_config, ax=ax, pretext_wrapper=config.env.use_wrapper)
	
	eval_envs = make_vec_envs(env_name, algo_args.eval_seed, 1,
									  algo_args.gamma, None, device, False, config=env_config, ax=ax, pretext_wrapper=config.env.use_wrapper)

	# create a policy network
	actor_critic = Policy(
		envs.observation_space.spaces, # pass the Dict into policy to parse
		envs.action_space,
		base_kwargs=algo_args,
		base=config.robot.policy)

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
	eval_all_hid_states = copy.deepcopy(all_hid_states)



	# start the training loop
	for j in range(num_updates):
		# schedule learning rate if needed
		if algo_args.use_linear_lr_decay:
			network_utils.update_linear_schedule(
				agent.optimizer, j, num_updates,
				agent.optimizer.lr if algo_args.algo == "acktr" else algo_args.lr)

		generated_gif=False
		# step the environment for a few times
		for step in range(algo_args.num_steps):
			# Sample actions
			all_actions = []
			all_values = []
			all_log_probs = []
			with torch.no_grad():
				# get the action for each robot				
				for i in range(config.sim.robot_num):
					value_i, action_i, log_i, recurrent_hidden_states_i = actor_critic.act(
						all_obs[i], all_hid_states[i],
						all_rollouts[i].masks[step])

					all_values.append(value_i)
					all_log_probs.append(log_i)
					all_actions.append(action_i)					
					all_hid_states[i]= copy.deepcopy(recurrent_hidden_states_i)					
				all_actions = torch.stack(all_actions, dim=1)
				
			# # if we use real prediction, send predictions to env for rendering
			# if env_name == 'CrowdSimPredRealGST-v0' and env_config.env.use_wrapper:
			# 	# [nenv, max_human_num, 2*(pred_steps+1)] -> [nenv, max_human_num, 2*pred_steps]
			# 	out_pred = rollouts_obs['spatial_edges'][:, :, 2:].to('cpu').numpy()
			# 	# send manager action to all processes
			# 	ack = envs.talk2Env(out_pred)
			# 	assert all(ack)

			frame_dir = "frames"
			if not os.path.exists(frame_dir):
				os.makedirs(frame_dir)
			if config.sim.render:
				envs.render()
				# if j % algo_args.eval_interval == 0 and not generated_gif and j>0:
				# 	create_gif_from_frames(frame_dir, "evaluation_{}.gif".format(j))
				# 	envs.frame_count=0
				# 	generated_gif=True

			obs, rewards, done, infos= envs.step(all_actions)

			for r in range(config.sim.robot_num):
				single_obs = {}
				for keyy in obs.keys():	
					single_obs[keyy] = []
					for i in range(obs[keyy].shape[0]):
						single_obs[keyy].append(obs[keyy][i][r])
					single_obs[keyy] = torch.stack(single_obs[keyy], dim = 0)
				all_obs[r] = single_obs
			
			
			for info in infos:
				if 'episode' in info.keys():
					episode_rewards.append(info['episode']['r'])

			# If done then clean the history of observations.
			masks = torch.FloatTensor(
				[[0.0] if done_ else [1.0] for done_ in done])
			bad_masks = torch.FloatTensor(
				[[0.0] if 'bad_transition' in info.keys() else [1.0]
				 for info in infos])
			

			#change to multi-agent rollout insert
			for robot_index in range(config.sim.robot_num):
				single_obs = {}
				for keyy in obs.keys():	
					single_obs[keyy] = []
					for i in range(obs[keyy].shape[0]):
						single_obs[keyy].append(obs[keyy][i][robot_index])
					single_obs[keyy] = torch.stack(single_obs[keyy], dim = 0)
				#print(single_obs['robot_node'])#generate robot_node all zeros why?
				masks = torch.FloatTensor([[0.0] if (done[e] or rewards[e][robot_index] == 0) else [1.0] for e in range(len(done))])

				all_rollouts[robot_index].insert(single_obs, all_hid_states[robot_index], torch.stack([all_action[robot_index] for all_action in all_actions]),
							all_log_probs[robot_index], all_values[robot_index], torch.tensor(rewards[:,robot_index:robot_index+1]), masks, bad_masks)

		with torch.no_grad():
			#change to multi-agent rollout update
			all_rollouts_obs=[{} for _ in range(config.sim.robot_num)]
			all_rollouts_hidden_s = [{} for _ in range(config.sim.robot_num)]
			for robot_index in range(config.sim.robot_num):
			
				for key in all_rollouts[robot_index].obs:
					all_rollouts_obs[robot_index][key] = all_rollouts[robot_index].obs[key][-1]
					
				for key in all_rollouts[robot_index].recurrent_hidden_states:
					all_rollouts_hidden_s[robot_index][key] = all_rollouts[robot_index].recurrent_hidden_states[key][-1]
					#print(all_rollouts_hidden_s[robot_index][key])
				next_value = actor_critic.get_value(
					all_rollouts_obs[robot_index], all_rollouts_hidden_s[robot_index],
					all_rollouts[robot_index].masks[-1]).detach()
				
		mean_action_loss=0	
		for robot_index in range(config.sim.robot_num):
			
			# why input robot_node and spatial_edges all zero
			# compute advantage and gradient, and update the network parameters
			all_rollouts[robot_index].compute_returns(next_value, algo_args.use_gae, algo_args.gamma,
										algo_args.gae_lambda, algo_args.use_proper_time_limits)
			
			value_loss, action_loss, dist_entropy = agent.update(all_rollouts[robot_index])

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
			if  wandb_log:
				wandb.log({"median_reward":np.median(episode_rewards),
						"mean_reward":np.mean(episode_rewards),
						"min_reward":np.min(episode_rewards),
						"max_reward":np.max(episode_rewards),
						"dist_entropy":dist_entropy,
						"value_loss":value_loss,
						"action_loss":action_loss,})

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

		# if j % algo_args.eval_interval == 0 and j>0:
			
		# 	# set an evaluation environment
		# 	eval_envs_ = copy.deepcopy(eval_envs)
		# 	mean_reward, success_rate = evaluate_training(eval_envs_, actor_critic, num_episodes=100, num_steps=algo_args.num_steps, robot_num=config.sim.robot_num, eval_all_hid_states=eval_all_hid_states)
		# 	#print(f"Evaluation: mean reward: {mean_reward}, success rate: {success_rate}")
		# 	# Save evaluation results
		# 	df_eval = pd.DataFrame({'misc/nupdates': [j], 'eval/mean_reward': [mean_reward], 'eval/success_rate': [success_rate]})
		# 	eval_csv_path = os.path.join(algo_args.output_dir, 'eval_progress.csv')
		# 	if os.path.exists(eval_csv_path) and j > 20:
		# 		df_eval.to_csv(eval_csv_path, mode='a', header=False, index=False)
		# 	else:
		# 		df_eval.to_csv(eval_csv_path, mode='w', header=True, index=False)

	
	
	if wandb_log:
		wandb.finish()
			

if __name__ == '__main__':
	main()

