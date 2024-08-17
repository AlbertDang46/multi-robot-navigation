import os
import numpy as np
import torch
import matplotlib as plt
from PIL import Image
from crowd_sim.envs.utils.info import *
import copy
# 
def create_gif_from_frames(frame_dir, gif_path, duration=100):
    
    frames = [Image.open(os.path.join(frame_dir, f)) for f in sorted(os.listdir(frame_dir)) if f.endswith('.png')]
    if frames:
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=duration, loop=0)
        print(f"GIF saved at {gif_path}")
    

def evaluate_training(eval_envs, actor_critic, num_episodes, num_steps, robot_num, eval_all_hid_states):
	
	all_episode_rewards = []
	success_count = 0
	all_hid_states=copy.deepcopy(eval_all_hid_states)
	obs = eval_envs.reset()
	for _ in range(num_episodes):
		episode_rewards = 0
		
		# for each robot, store the observation
		all_obs = []
		for r in range(robot_num):
			single_obs = {}
			for keyy in obs.keys():	
				single_obs[keyy] = []
				for i in range(obs[keyy].shape[0]):
					single_obs[keyy].append(obs[keyy][i][r])
				single_obs[keyy] = torch.stack(single_obs[keyy], dim = 0)
			all_obs.append(single_obs)

		for step in range(num_steps):
			all_actions = []
			with torch.no_grad():
				for i in range(robot_num):
					
					value_i, action_i, log_i, recurrent_hidden_states_i = actor_critic.act(
                        all_obs[i], all_hid_states[i], torch.ones(1, 1).cuda())
					all_actions.append(action_i[0])
					all_hid_states[i] = recurrent_hidden_states_i
				all_actions = torch.stack(all_actions, dim=0)
            
			
			obs, reward, done, infos = eval_envs.step(all_actions)
			if step == num_steps - 1:
				done=True
			# for each robot, store the observation
			all_obs = []
			for r in range(robot_num):
				single_obs = {}
				for keyy in obs.keys():	
					single_obs[keyy] = []
					for i in range(obs[keyy].shape[0]):
						single_obs[keyy].append(obs[keyy][i][r])
					single_obs[keyy] = torch.stack(single_obs[keyy], dim = 0)
				all_obs.append(single_obs)
			episode_rewards += reward.mean().item()
			# if done.any():
			# 	break
		all_episode_rewards.append(episode_rewards)
		for info in infos:
			
			if isinstance(info['info'], ReachGoal):
				success_count += 1
		
		# for i in infos:
		# 	infos=i
		# success_count=(infos == 'Reaching goal')
		

	mean_reward = np.mean(all_episode_rewards)
	success_rate = success_count / num_episodes
	print(f"Evaluation over {num_episodes} episodes: mean reward: {mean_reward}, success rate: {success_rate}")
	return mean_reward, success_rate


def evaluate(actor_critic, eval_envs, num_processes, device, test_size, logging, config, args,visualize=False, num_robot = 1):
    """ function to run all testing episodes and log the testing metrics """
    # initializations
    eval_episode_rewards = []
    #print(config.robot.policy)#selfAttn_merge_srnn
    
    if config.robot.policy not in ['orca', 'social_force']:
        eval_recurrent_hidden_states = {}

        node_num = 1
        edge_num = actor_critic.base.human_num + 1
        eval_recurrent_hidden_states['human_node_rnn'] = torch.zeros(num_processes, node_num, actor_critic.base.human_node_rnn_size,
                                                                     device=device)

        eval_recurrent_hidden_states['human_human_edge_rnn'] = torch.zeros(num_processes, edge_num,
                                                                           actor_critic.base.human_human_edge_rnn_size,
                                                                           device=device)

    eval_masks = torch.zeros(num_processes, 1, device=device)

    success_times = []
    collision_times = []
    timeout_times = []

    success = 0
    collision = 0
    timeout = 0
    too_close_ratios = []
    min_dist = []

    collision_cases = []
    timeout_cases = []

    all_path_len = []

    # to make it work with the virtualenv in sim2real
    if hasattr(eval_envs.venv, 'envs'):
        baseEnv = eval_envs.venv.envs[0].env
    else:
        baseEnv = eval_envs.venv.unwrapped.envs[0].env
    time_limit = baseEnv.time_limit

    # start the testing episodes
    for k in range(test_size):
        gif_generated=False
        baseEnv.episode_k = k
        done = False
        rewards = []
        stepCounter = 0
        episode_rew = 0
        # 1. reset the environment
        env_num = 1
        obs = eval_envs.reset()
        actions = torch.zeros([env_num,num_robot, 2], device=device)
        hid_states = [eval_recurrent_hidden_states] * num_robot

        

        global_time = 0.0
        path_len = 0.
        too_close = 0.
        last_pos = [obs['robot_node'][0, i, 0, :2].cpu().numpy() for i in range(num_robot)]


        while not done:
            stepCounter = stepCounter + 1
            # for each robot
            for robot_index in range(num_robot):
                # 1. get observation of this robot
                single_obs = {}
                for keyy in obs.keys():	
                    single_obs[keyy] = []
                    for i in range(obs[keyy].shape[0]):
                        single_obs[keyy].append(obs[keyy][i][robot_index])
                    single_obs[keyy] = torch.stack(single_obs[keyy], dim = 0)
                

                # 2. get action and hid_state of this robot
                if config.robot.policy not in ['orca', 'social_force']:
                    # run inference on the NN policy                
                    with torch.no_grad():
                        _,actions_i , _, hid_states_i = actor_critic.act(
                            single_obs,
                            hid_states[robot_index],
                            eval_masks,
                            deterministic=True)
                
                    actions[0][robot_index] = copy.deepcopy(actions_i)
                    hid_states[robot_index] = copy.deepcopy(hid_states_i)
                else:
                    actions[robot_index] = torch.zeros([1, 2], device=device)
                if not done:
                    global_time = baseEnv.global_time

            # if the vec_pretext_normalize.py wrapper is used, send the predicted traj to env
            if args.env_name == 'CrowdSimPredRealGST-v0' and config.env.use_wrapper:
                out_pred = obs['spatial_edges'][:, :, 2:].to('cpu').numpy()
                # send manager action to all processes
                ack = eval_envs.talk2Env(out_pred)
                assert all(ack)
            # 创建一个目录来存储帧
            frame_dir = "frames"
            if not os.path.exists(frame_dir):
                os.makedirs(frame_dir)
            
            if visualize:
                eval_envs.render()
            if not gif_generated and k==0:
                create_gif_from_frames(frame_dir, "evaluation0.gif")
                eval_envs.frame_count=0
                gif_generated=True
                
            

            # Obser reward and next obs

            # actions=torch.stack([action for i in range(num_robot)],dim=1)
            
 
            obs, rew, done, infos = eval_envs.step(actions)
            
            rewards.append(rew)

            single_obs = {}
            for keyy in obs.keys():	
                single_obs[keyy] = []
                for i in range(obs[keyy].shape[0]):
                    single_obs[keyy].append(obs[keyy][i][0])
                single_obs[keyy] = torch.stack(single_obs[keyy], dim = 0)
            
        
            path_len = path_len + np.linalg.norm(single_obs['robot_node'][0, 0, :2].cpu().numpy() - last_pos)
            last_pos = [obs['robot_node'][0, i, 0, :2].cpu().numpy() for i in range(num_robot)]


            if isinstance(infos[0]['info'], Danger):
                too_close = too_close + 1
                min_dist.append(infos[0]['info'].min_dist)

            episode_rew += rew[0]


            eval_masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device)

            for info in infos:
                if 'episode' in info.keys():
                    eval_episode_rewards.append(info['episode']['r'])

        # an episode ends!
        print('')
        print('Reward={}'.format(episode_rew))
        print('Episode', k, 'ends in', stepCounter)
        all_path_len.append(path_len)
        too_close_ratios.append(too_close/stepCounter*100)

        
        if isinstance(infos[0]['info'], ReachGoal):
            success += num_robot
            for _ in range(num_robot):
                success_times.append(global_time)
                print('Success')
        elif isinstance(infos[0]['info'], Collision):
            for r in range(num_robot):
                if episode_rew[r] >=25:
                    success += 1
                    success_times.append(global_time)
                    print('Success')
                else:
                    collision += 1
                    collision_cases.append(k)
                    collision_times.append(global_time)
                    print('Collision')

        elif isinstance(infos[0]['info'], Timeout):
            for r in range(num_robot):
                if episode_rew[r] >=25:
                    success += 1
                    success_times.append(global_time)
                elif episode_rew[r] >= -5:
                    timeout += 1
                    timeout_cases.append(k)
                    timeout_times.append(time_limit)
                    print('Timeout')
                else:
                    collision += 1
                    collision_cases.append(k)
                    collision_times.append(global_time)
                    print('Collision')
        elif infos[0]['info'] is None:
            pass
        else:
            raise ValueError('Invalid end signal from environment')

    # all episodes end
    success_rate = success / test_size / num_robot
    collision_rate = collision / test_size / num_robot
    timeout_rate = timeout / test_size / num_robot
    assert success + collision + timeout == test_size * num_robot
    avg_nav_time = sum(success_times) / len(
        success_times) if success_times else time_limit  # baseEnv.env.time_limit

    # logging
    logging.info(
        'Testing success rate: {:.2f}, collision rate: {:.2f}, timeout rate: {:.2f}, '
        'nav time: {:.2f}, path length: {:.2f}, average intrusion ratio: {:.2f}%, '
        'average minimal distance during intrusions: {:.2f}'.
            format(success_rate, collision_rate, timeout_rate, avg_nav_time, np.mean(all_path_len),
                   np.mean(too_close_ratios), np.mean(min_dist)))

    logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
    logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))
    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))

    eval_envs.close()