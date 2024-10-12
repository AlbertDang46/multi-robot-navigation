import os
import numpy as np
import torch
import matplotlib as plt
from PIL import Image
from crowd_sim.envs.utils.info import *
import copy
# 

def reset_folder(folder):
    if os.path.exists(folder):
        file_list = os.listdir(folder)
        for file_name in file_list:
            file_path = os.path.join(folder, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(folder)

def create_gif_from_frames(frame_dir, gif_path, duration=2):
    
    frames = [Image.open(os.path.join(frame_dir, f)) for f in sorted(os.listdir(frame_dir)) if f.endswith('.png')]
    print(f"Creating GIF from {len(frames)} frames")
    if frames:
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=0, loop=0, optimize=True, dpi = (120,120))
        print(f"GIF saved at {gif_path}")

    reset_folder(frame_dir)

def evaluate(actor_critic, eval_envs, num_processes, device, test_size, logging, config, args,visualize=False, num_robot = 1):
    """ function to run all testing episodes and log the testing metrics """
    # initializations
    eval_episode_rewards = []
    #print(config.robot.policy)#selfAttn_merge_srnn
    
    if config.robot.policy not in ['orca', 'social_force']:
        eval_recurrent_hidden_states = {}

        node_num = 1
        edge_num = config.sim.robot_num + 1
        eval_recurrent_hidden_states['human_node_rnn'] = torch.zeros(num_processes, node_num, args.rnn_hidden_size,
                                                                     device=device)

        eval_recurrent_hidden_states['human_human_edge_rnn'] = torch.zeros(num_processes, edge_num,
                                                                           args.human_human_edge_rnn_size,
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

    all_velocities = []
    all_acceleration = []
    all_velocities_in_crowd = []
    sharp_turns = []


    # to make it work with the virtualenv in sim2real
    if hasattr(eval_envs.venv, 'envs'):
        baseEnv = eval_envs.venv.envs[0].env
    else:
        baseEnv = eval_envs.venv.unwrapped.envs[0].env
    time_limit = baseEnv.time_limit

    gif_dir = "gifs"
    if os.path.exists(gif_dir):
        file_list = os.listdir(gif_dir)
        for file_name in file_list:
            file_path = os.path.join(gif_dir, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(gif_dir)

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
        last_pos = [obs['robot_info'][0, i, 0, :2].cpu().numpy() for i in range(num_robot)]
        last_vel = [obs['robot_info'][0, i, 0, 2:4].cpu().numpy() for i in range(num_robot)]

        # Reset the frames folder
        frame_dir = "frames"
        if os.path.exists(frame_dir):
            file_list = os.listdir(frame_dir)
            for file_name in file_list:
                file_path = os.path.join(frame_dir, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        else:
            os.makedirs(frame_dir)
        
        
         

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

        
            if visualize:
                eval_envs.render(mode = 'record')
            obs, rew, done, infos = eval_envs.step(actions)
            
            rewards.append(rew)

            single_obs = {}
            for keyy in obs.keys():	
                single_obs[keyy] = []
                for i in range(obs[keyy].shape[0]):
                    single_obs[keyy].append(obs[keyy][i][0])
                single_obs[keyy] = torch.stack(single_obs[keyy], dim = 0)
            
            for i in range(num_robot):
                dl = np.linalg.norm(obs['robot_info'][0, i, 0, :2].cpu().numpy() - last_pos[i])
                path_len += dl
                if dl > 0:
                    #all_sudden_turns.append(np.arccos(np.dot(last_vel[i], obs['robot_info'][0, i, 0, 2:4].cpu().numpy()) / (np.linalg.norm(last_vel[i]) * np.linalg.norm(obs['robot_info'][0, i, 0, 2:4].cpu().numpy()))) * 180 / np.pi)
                    all_velocities.append(np.linalg.norm(obs['robot_info'][0, i, 0, 2:4].cpu().numpy() - last_vel[i]))
                    all_acceleration.append(np.linalg.norm(obs['robot_info'][0, i, 0, 2:4].cpu().numpy() - last_vel[i]))
                    if sum(sum(obs['occupancy_map'][0, i, 0])) > 25 :
                        all_velocities_in_crowd.append(np.linalg.norm(obs['robot_info'][0, i, 0, 2:4].cpu().numpy()))
                        if np.linalg.norm(last_vel[i]) > 0 and np.linalg.norm(obs['robot_info'][0, i, 0, 2:4].cpu().numpy()) > 0:
                            velocity_change_angle = np.arccos(np.clip(np.dot(last_vel[i], obs['robot_info'][0, i, 0, 2:4].cpu().numpy()) / (np.linalg.norm(last_vel[i]) * np.linalg.norm(obs['robot_info'][0, i, 0, 2:4].cpu().numpy())),-1,1))
                            velocity_change_angle = np.arccos(np.clip(np.dot(last_vel[i], obs['robot_info'][0, i, 0, 2:4].cpu().numpy()) / (np.linalg.norm(last_vel[i]) * np.linalg.norm(obs['robot_info'][0, i, 0, 2:4].cpu().numpy())),-1,1))
                            sharp_turns.append(velocity_change_angle)

            last_pos = [obs['robot_info'][0, i, 0, :2].cpu().numpy() for i in range(num_robot)]
            last_vel = [obs['robot_info'][0, i, 0, 2:4].cpu().numpy() for i in range(num_robot)]



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
        all_path_len.append(path_len  / num_robot)
        too_close_ratios.append(too_close/stepCounter*100)
        
        if visualize:
            create_gif_from_frames(frame_dir,os.path.join("gifs",f"evaluation{k:04d}.gif"))
        
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
        'Testing success rate: {:.4f}, collision rate: {:.4f}, timeout rate: {:.4f}, '
        'nav time: {:.2f}, path length: {:.2f}, average intrusion ratio: {:.2f}%, '
        'average minimal distance during intrusions: {:.2f}, average acceleration: {:.2f}, sharp turn rate {:.4f}'.
            format(success_rate, collision_rate, timeout_rate, avg_nav_time, np.mean(all_path_len),
                   np.mean(too_close_ratios), np.mean(min_dist), np.mean(all_acceleration), np.mean(np.array(sharp_turns)>0.15)))
    logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
    logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))
    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))
    
    # Calculate velocity distribution
    velocity_distribution = np.histogram(sharp_turns, bins=10)

    # Print velocity distribution
    for i in range(len(velocity_distribution[0])):
        print(f"AA range: {velocity_distribution[1][i]:.2f} - {velocity_distribution[1][i+1]:.2f}, Frequency: {velocity_distribution[0][i]/len(sharp_turns)*100:.2f}%, Total: {velocity_distribution[0][i]}")

    # Calculate velocity distribution in crowd
    velocity_distribution_in_crowd = np.histogram(all_velocities_in_crowd, bins=10)
    for i in range(len(velocity_distribution_in_crowd[0])):
        print(f"Velocity range in crowd: {velocity_distribution_in_crowd[1][i]:.2f} - {velocity_distribution_in_crowd[1][i+1]:.2f}, Frequency: {velocity_distribution_in_crowd[0][i]/len(all_velocities_in_crowd)*100:.2f}%, Total: {velocity_distribution_in_crowd[0][i]}")
    eval_envs.close()