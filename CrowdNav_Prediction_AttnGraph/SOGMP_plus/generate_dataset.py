import copy
import numpy as np
import os

# Define base paths
base_dir = 'SOGMP_plus/OGM-datasets/dataset5'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

# Create directories for training and validation datasets
os.makedirs(os.path.join(train_dir, 'scans'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'positions'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'velocities'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'scans'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'positions'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'velocities'), exist_ok=True)

# Load the original .npy files
lidar_data = np.load('dataset/successful_lidar_seq3.npy') 
vel_pos_data = np.load('dataset/successful_vel_pos_seq3.npy') 

lidar_data=lidar_data[:40]
vel_pos_data=vel_pos_data[:40]

traj_num,seq_len,robot_num,batchsize,_,_ = lidar_data.shape
lidar_data = np.transpose(lidar_data, (0, 3, 1, 2, 4, 5)).reshape(traj_num * batchsize, seq_len, robot_num, 90, 2)

# Transpose and reshape vel_pos_data
vel_pos_data = np.transpose(vel_pos_data, (0, 3, 1, 2, 4)).reshape(traj_num * batchsize, seq_len, robot_num, 9)

new_vel_pos = vel_pos_data[:, :, :, [7, 8, 6, 0, 1, 6]]
# lidar_data = lidar_data.permute(0,3,1,2,4,5).reshape(traj_num*batchsize,seq_len,robot_num,90,2)
# vel_pos_data = vel_pos_data.permute(0,3,1,2,4).reshape(traj_num*batchsize,seq_len,robot_num,9)
# new_vel_pos = vel_pos_data[:,:,:,[7, 8, 6, 0, 1, 6]] #vx, vy, theta, x, y, theta

# Compute the norm of the velocity vector (vx, vy)
v = np.linalg.norm(new_vel_pos[:,:,:,:2], axis=-1, keepdims=True)  # (128, 30, 3, 1)

# Concatenate the norm and other selected features
new_vel_pos = np.concatenate([v, new_vel_pos[:,:,:,2:]], axis=-1) # 128 30 3 5

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

num_traj, num_steps, num_robots, num_features = new_vel_pos.shape
lidar_data = lidar_data.reshape(num_traj*num_steps, num_robots, 90, 2)
new_vel_pos = new_vel_pos.reshape(num_traj*num_steps, num_robots, -1)
#train_idx = int(lidar_data.shape[0] * 0.8)

# Get the total number of data points
traj_num = lidar_data.shape[0]

# Randomly shuffle the indices of the data
random_indices = np.random.permutation(traj_num)

# Compute the train index for 80% of the data
train_idx = int(traj_num * 0.8)

# Select the first 80% of indices as the training set
train_indices = random_indices[:train_idx]
val_indices = random_indices[train_idx:]

# print('Train:', min(train_indices),max(train_indices))
# print('Val:', min(val_indices),max(val_indices))
# exit()
# Save each slice as a separate file and categorize into train/val
COUNT_train=0
COUNT_val=0
for i in range(lidar_data.shape[0]):
    if i in train_indices:
        print('Train',i, COUNT_train)
        scan_path = 'scans/{}.npy'.format(COUNT_train)
        pos_path = 'positions/{}.npy'.format(COUNT_train)
        vel_path = 'velocities/{}.npy'.format(COUNT_train)
        COUNT_train+=1
        np.save(os.path.join(train_dir, scan_path), lidar_data[i])
        np.save(os.path.join(train_dir, pos_path), new_vel_pos[i, :,2:])
        np.save(os.path.join(train_dir, vel_path), new_vel_pos[i, :, :2])
    else:
        print('VAL',i, COUNT_val)
        scan_path = 'scans/{}.npy'.format(COUNT_val)
        pos_path = 'positions/{}.npy'.format(COUNT_val)
        vel_path = 'velocities/{}.npy'.format(COUNT_val)
        COUNT_val+=1
        # Validation data
        np.save(os.path.join(val_dir, scan_path), lidar_data[i])
        np.save(os.path.join(val_dir, pos_path), new_vel_pos[i, :,2:])
        np.save(os.path.join(val_dir, vel_path), new_vel_pos[i, :, :2])
COUNT_train=0
COUNT_val=0
# Create file lists for training and validation datasets
with open(os.path.join(train_dir, 'scans/train.txt'), 'w') as f_train_scan, \
     open(os.path.join(train_dir, 'positions/train.txt'), 'w') as f_train_pos, \
     open(os.path.join(train_dir, 'velocities/train.txt'), 'w') as f_train_vel, \
     open(os.path.join(val_dir, 'scans/val.txt'), 'w') as f_val_scan, \
     open(os.path.join(val_dir, 'positions/val.txt'), 'w') as f_val_pos, \
     open(os.path.join(val_dir, 'velocities/val.txt'), 'w') as f_val_vel:

    for i in range(lidar_data.shape[0]):
        

        if i in train_indices:
            print('Train',i, COUNT_train)
            scan_path = 'scans/{}.npy'.format(COUNT_train)
            pos_path = 'positions/{}.npy'.format(COUNT_train)
            vel_path = 'velocities/{}.npy'.format(COUNT_train)
            COUNT_train+=1
            f_train_scan.write(scan_path + '\n')
            f_train_pos.write(pos_path + '\n')
            f_train_vel.write(vel_path + '\n')
        else:
            print('VAL',i, COUNT_val)
            scan_path = 'scans/{}.npy'.format(COUNT_val)
            pos_path = 'positions/{}.npy'.format(COUNT_val)
            vel_path = 'velocities/{}.npy'.format(COUNT_val)
            COUNT_val+=1
            f_val_scan.write(scan_path + '\n')
            f_val_pos.write(pos_path + '\n')
            f_val_vel.write(vel_path + '\n')

