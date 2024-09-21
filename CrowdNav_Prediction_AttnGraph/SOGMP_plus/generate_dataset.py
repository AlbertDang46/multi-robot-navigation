# import copy
# import numpy as np
# import os

# # Define base paths
# base_dir = 'SOGMP_plus/OGM-datasets/dataset_6'
# train_dir = os.path.join(base_dir, 'train')
# val_dir = os.path.join(base_dir, 'val')

# # Create directories for training and validation datasets
# os.makedirs(os.path.join(train_dir, 'scans'), exist_ok=True)
# os.makedirs(os.path.join(train_dir, 'positions'), exist_ok=True)
# os.makedirs(os.path.join(train_dir, 'velocities'), exist_ok=True)
# os.makedirs(os.path.join(val_dir, 'scans'), exist_ok=True)
# os.makedirs(os.path.join(val_dir, 'positions'), exist_ok=True)
# os.makedirs(os.path.join(val_dir, 'velocities'), exist_ok=True)

# # Load the original .npy files
# lidar_data = np.load('dataset/successful_lidar_seq3.npy') 
# vel_pos_data = np.load('dataset/successful_vel_pos_seq3.npy') 

# lidar_data=lidar_data[:40]
# vel_pos_data=vel_pos_data[:40]

# traj_num,seq_len,robot_num,batchsize,_,_ = lidar_data.shape
# lidar_data = np.transpose(lidar_data, (0, 3, 1, 2, 4, 5)).reshape(traj_num * batchsize, seq_len, robot_num, 90, 2)

# # Transpose and reshape vel_pos_data
# vel_pos_data = np.transpose(vel_pos_data, (0, 3, 1, 2, 4)).reshape(traj_num * batchsize, seq_len, robot_num, 9)

# new_vel_pos = vel_pos_data[:, :, :, [7, 8, 6, 0, 1, 6]]
# # lidar_data = lidar_data.permute(0,3,1,2,4,5).reshape(traj_num*batchsize,seq_len,robot_num,90,2)
# # vel_pos_data = vel_pos_data.permute(0,3,1,2,4).reshape(traj_num*batchsize,seq_len,robot_num,9)
# # new_vel_pos = vel_pos_data[:,:,:,[7, 8, 6, 0, 1, 6]] #vx, vy, theta, x, y, theta

# # Compute the norm of the velocity vector (vx, vy)
# v = np.linalg.norm(new_vel_pos[:,:,:,:2], axis=-1, keepdims=True)  # (128, 30, 3, 1)

# # Concatenate the norm and other selected features
# new_vel_pos = np.concatenate([v, new_vel_pos[:,:,:,2:]], axis=-1) # 128 30 3 5

# # Initialize container for angular changes
# angular_changes =copy.deepcopy(new_vel_pos[:,:,:,1]) 

# # Compute angular changes across time steps
# for t in range(1, new_vel_pos.shape[0]):  # start from 1 since there's no previous for t=0
#     last_theta = new_vel_pos[t-1,:,:,1]  # Previous theta
#     current_theta = new_vel_pos[t,:,:,1]  # Current theta

#     # Calculate angular difference
#     delta_theta = (current_theta - last_theta + np.pi) % (2 * np.pi) - np.pi
#     angular_changes[t] = delta_theta  # Store computed angular changes

# # Optionally, you might want to update the original tensor with these changes
# new_vel_pos[:,:,:,1] = angular_changes

# # Determine the split index for 80% training and 20% validation

# num_traj, num_steps, num_robots, num_features = new_vel_pos.shape
# # lidar_data = lidar_data.reshape(num_traj*num_steps, num_robots, 90, 2)
# # new_vel_pos = new_vel_pos.reshape(num_traj*num_steps, num_robots, -1)
# #train_idx = int(lidar_data.shape[0] * 0.8)

# # Get the total number of data points
# traj_num = lidar_data.shape[0] # traj_num 30 3 90 2
# # print(lidar_data.shape)
# # exit()
# # Randomly shuffle the indices of the data
# random_indices = np.random.permutation(traj_num)
# #random_indices = np.random.permutation(traj_num*(num_steps-2*seq_len))


# val_idx = int(traj_num * 0.8)

# # Select the first 80% of indices as the training set
# #train_indices = random_indices[:train_idx]
# val_indices = random_indices[val_idx:]

# # print('Train:', min(train_indices),max(train_indices))
# # print('Val:', min(val_indices),max(val_indices))
# # exit()
# # Save each slice as a separate file and categorize into train/val
# COUNT_train=0
# COUNT_val=0
# history_seq_len=5
# for i in range(lidar_data.shape[0]):
#     if i in val_indices:
#         start_idx = COUNT_val%num_steps
#         if start_idx>=num_steps-2*history_seq_len:
#             start_idx =start_idx%(num_steps-2*history_seq_len)
#         for k in range(2*history_seq_len):
#             scan_path = 'scans/{}.npy'.format(COUNT_val)
#             pos_path = 'positions/{}.npy'.format(COUNT_val)
#             vel_path = 'velocities/{}.npy'.format(COUNT_val)
#             COUNT_val+=1
#             np.save(os.path.join(val_dir, scan_path), lidar_data[i,start_idx+k])
#             np.save(os.path.join(val_dir, pos_path), new_vel_pos[i,start_idx+k,:,2:])
#             np.save(os.path.join(val_dir, vel_path), new_vel_pos[i,start_idx+k,:,:2])
#     for j in range(num_steps):
#         scan_path = 'scans/{}.npy'.format(COUNT_train)
#         pos_path = 'positions/{}.npy'.format(COUNT_train)
#         vel_path = 'velocities/{}.npy'.format(COUNT_train)
#         COUNT_train+=1
#         np.save(os.path.join(train_dir, scan_path), lidar_data[i,j])
#         np.save(os.path.join(train_dir, pos_path), new_vel_pos[i,j,:,2:])
#         np.save(os.path.join(train_dir, vel_path), new_vel_pos[i,j,:,:2])
#         print(lidar_data[i,j].shape, new_vel_pos[i,j,:,2:].shape, new_vel_pos[i,j,:,:2].shape)
# print(COUNT_train, COUNT_val)
# COUNT_train=0
# COUNT_val=0
# # Create file lists for training and validation datasets
# with open(os.path.join(train_dir, 'scans/train.txt'), 'w') as f_train_scan, \
#      open(os.path.join(train_dir, 'positions/train.txt'), 'w') as f_train_pos, \
#      open(os.path.join(train_dir, 'velocities/train.txt'), 'w') as f_train_vel, \
#      open(os.path.join(val_dir, 'scans/val.txt'), 'w') as f_val_scan, \
#      open(os.path.join(val_dir, 'positions/val.txt'), 'w') as f_val_pos, \
#      open(os.path.join(val_dir, 'velocities/val.txt'), 'w') as f_val_vel:

#     for i in range(lidar_data.shape[0]):
#         if i in val_indices:
#             start_idx = COUNT_val%num_steps
#             if start_idx>=num_steps-2*history_seq_len:
#                 start_idx =start_idx%(num_steps-2*history_seq_len)
#             for k in range(2*history_seq_len):
#                 scan_path = 'scans/{}.npy'.format(COUNT_val)
#                 pos_path = 'positions/{}.npy'.format(COUNT_val)
#                 vel_path = 'velocities/{}.npy'.format(COUNT_val)
#                 COUNT_val+=1
#                 f_val_scan.write(scan_path + '\n')
#                 f_val_pos.write(pos_path + '\n')
#                 f_val_vel.write(vel_path + '\n')
#         for j in range(num_steps):
#             scan_path = 'scans/{}.npy'.format(COUNT_train)
#             pos_path = 'positions/{}.npy'.format(COUNT_train)
#             vel_path = 'velocities/{}.npy'.format(COUNT_train)
#             COUNT_train+=1
#             f_train_scan.write(scan_path + '\n')
#             f_train_pos.write(pos_path + '\n')
#             f_train_vel.write(vel_path + '\n')
            
        

#         # if i in train_indices:
#         #     print('Train',i, COUNT_train)
#         #     scan_path = 'scans/{}.npy'.format(COUNT_train)
#         #     pos_path = 'positions/{}.npy'.format(COUNT_train)
#         #     vel_path = 'velocities/{}.npy'.format(COUNT_train)
#         #     COUNT_train+=1
#         #     f_train_scan.write(scan_path + '\n')
#         #     f_train_pos.write(pos_path + '\n')
#         #     f_train_vel.write(vel_path + '\n')
#         # else:
#         #     print('VAL',i, COUNT_val)
#         #     scan_path = 'scans/{}.npy'.format(COUNT_val)
#         #     pos_path = 'positions/{}.npy'.format(COUNT_val)
#         #     vel_path = 'velocities/{}.npy'.format(COUNT_val)
#         #     COUNT_val+=1
#         #     f_val_scan.write(scan_path + '\n')
#         #     f_val_pos.write(pos_path + '\n')
#         #     f_val_vel.write(vel_path + '\n')


import copy
import numpy as np
import os

# Define base paths
base_dir = 'SOGMP_plus/OGM-datasets/dataset_reset_large'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
FUTURE_STEP=4
# Create directories for training and validation datasets
os.makedirs(os.path.join(train_dir, 'scans'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'positions'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'velocities'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'targets'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'scans'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'positions'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'velocities'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'targets'), exist_ok=True)

# Load the original .npy files
lidar_data = np.load('dataset/successful_lidar_seq_0915.npy') 
vel_pos_data = np.load('dataset/successful_vel_pos_seq_0915.npy')
future_lidar_data =  np.load('dataset/future_seq_0915.npy')



new_vel_pos = vel_pos_data[:, :, :, [7, 8, 6, 0, 1, 6]] #vx, vy, theta, x, y, theta

# Compute the norm of the velocity vector (vx, vy)
v = np.linalg.norm(new_vel_pos[:,:,:,:2], axis=-1, keepdims=True)  # (traj_num, 30, 3, 1)

# Concatenate the norm and other selected features
new_vel_pos = np.concatenate([v, new_vel_pos[:,:,:,2:]], axis=-1) # traj_num 30 3 5

# Initialize container for angular changes
angular_changes =copy.deepcopy(new_vel_pos[:,:,:,1]) 

# Compute angular changes across time steps
for t in range(1, new_vel_pos.shape[0]):  # start from 1 since there's no previous for t=0
    last_theta = new_vel_pos[t-1,:,:,1]  # Previous theta
    current_theta = new_vel_pos[t,:,:,1]  # Current theta

    # Calculate angular difference
    delta_theta = (current_theta - last_theta + np.pi) % (2 * np.pi) - np.pi
    angular_changes[t] = delta_theta  # Store computed angular changes

# Optionally, you might want to update the original tensor with these changes
new_vel_pos[:,:,:,1] = angular_changes

# Determine the split index for 80% training and 20% validation
traj_num,seq_len,robot_num,_,_ = lidar_data.shape

lidar_data = lidar_data.reshape(traj_num*seq_len, robot_num, 90, 2)
new_vel_pos = new_vel_pos.reshape(traj_num*seq_len, robot_num, -1)
future_lidar_data = np.transpose(future_lidar_data, (0, 1, 3, 2, 4, 5))
future_lidar_data = future_lidar_data.reshape(traj_num*seq_len, robot_num, FUTURE_STEP, 90, 2)



# Randomly shuffle the indices of the data
random_indices = np.random.permutation(traj_num)

train_idx = int(traj_num*0.9)

# Select the first 90% of indices as the training set
train_indices = random_indices[:train_idx]
val_indices = random_indices[train_idx:]

# print('Train:', min(train_indices),max(train_indices))
# print('Val:', min(val_indices),max(val_indices))
# exit()
# Save each slice as a separate file and categorize into train/val

COUNT_train=0
COUNT_val=0
# COUNT=0
# two_train=0
# two_val=0
# one_train=0
# one_val=0
# train_index_distribution = {}
# val_index_distribution = {}
for i in range(future_lidar_data.shape[0]):
    if i//seq_len in train_indices:
        #print('Train',i//30)
        
        scan_path = 'scans/{}.npy'.format(COUNT_train)
        pos_path = 'positions/{}.npy'.format(COUNT_train)
        vel_path = 'velocities/{}.npy'.format(COUNT_train)
        target_path = 'targets/{}.npy'.format(COUNT_train)
        COUNT_train+=1
        #ind=seq_len*(i//(seq_len-FUTURE_STEP))+i%(seq_len-FUTURE_STEP)
        np.save(os.path.join(train_dir, scan_path), lidar_data[i])
        np.save(os.path.join(train_dir, pos_path), new_vel_pos[i, :,2:])
        np.save(os.path.join(train_dir, vel_path), new_vel_pos[i, :, :2])
        np.save(os.path.join(train_dir, target_path), future_lidar_data[i])
        
    else:
        if i//seq_len in val_indices:
            
            scan_path = 'scans/{}.npy'.format(COUNT_val)
            pos_path = 'positions/{}.npy'.format(COUNT_val)
            vel_path = 'velocities/{}.npy'.format(COUNT_val)
            target_path = 'targets/{}.npy'.format(COUNT_val)
            COUNT_val+=1
            # Validation data
            ind=seq_len*(i//(seq_len-FUTURE_STEP))+i%(seq_len-FUTURE_STEP)
            np.save(os.path.join(val_dir, scan_path), lidar_data[i])
            np.save(os.path.join(val_dir, pos_path), new_vel_pos[i, :,2:])
            np.save(os.path.join(val_dir, vel_path), new_vel_pos[i, :, :2])
            np.save(os.path.join(val_dir, target_path), future_lidar_data[i])
    #print(COUNT_train, COUNT_val, (COUNT_train+COUNT_val),COUNT)
# print("Training index distribution for label == 2:", train_index_distribution)
# print("Validation index distribution for label == 2:", val_index_distribution)
#print(two_train/(COUNT_train),two_val/(COUNT_val))
#0.062458847736625515 0.7032516718106996 0.05619791666666667 0.7150810185185185

COUNT_train=0
COUNT_val=0
COUNT=0
# Create file lists for training and validation datasets
with open(os.path.join(train_dir, 'scans/train.txt'), 'w') as f_train_scan, \
     open(os.path.join(train_dir, 'positions/train.txt'), 'w') as f_train_pos, \
     open(os.path.join(train_dir, 'velocities/train.txt'), 'w') as f_train_vel, \
     open(os.path.join(train_dir, 'targets/train.txt'), 'w') as f_train_target, \
     open(os.path.join(val_dir, 'scans/val.txt'), 'w') as f_val_scan, \
     open(os.path.join(val_dir, 'positions/val.txt'), 'w') as f_val_pos, \
    open(os.path.join(val_dir, 'velocities/val.txt'), 'w') as f_val_vel, \
     open(os.path.join(val_dir, 'targets/val.txt'), 'w') as f_val_target:

    for i in range(future_lidar_data.shape[0]):
        

        if i//seq_len in train_indices:
            
            scan_path = 'scans/{}.npy'.format(COUNT_train)
            pos_path = 'positions/{}.npy'.format(COUNT_train)
            vel_path = 'velocities/{}.npy'.format(COUNT_train)
            target_path = 'targets/{}.npy'.format(COUNT_train)
            COUNT_train+=1
            f_train_scan.write(scan_path + '\n')
            f_train_pos.write(pos_path + '\n')
            f_train_vel.write(vel_path + '\n')
            f_train_target.write(target_path + '\n')
        else:
            if i//seq_len in val_indices:
                
                scan_path = 'scans/{}.npy'.format(COUNT_val)
                pos_path = 'positions/{}.npy'.format(COUNT_val)
                vel_path = 'velocities/{}.npy'.format(COUNT_val)
                target_path = 'targets/{}.npy'.format(COUNT_val)
                COUNT_val+=1
                f_val_scan.write(scan_path + '\n')
                f_val_pos.write(pos_path + '\n')
                f_val_vel.write(vel_path + '\n')
                f_val_target.write(target_path + '\n')

print(COUNT_train, COUNT_val, (COUNT_train+COUNT_val),lidar_data.shape[0],future_lidar_data.shape[0])

