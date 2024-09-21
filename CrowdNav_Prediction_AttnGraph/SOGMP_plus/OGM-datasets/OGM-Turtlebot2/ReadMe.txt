Summary: An occupancy grid map dataset collected by a simulated Turtlebot2 with a the maximum speed of 0.8 m/s navigates around a lobby Gazebo environment with 34 moving pedestrians using random start points and goal points

Data: <robot position, robot velocity, and lidar scan measurement>

Sampling rate: 10 Hz

Size:
Total: 94,891 tuples
Training: 67,000 tuples
Validation: 10,891 tuples
Test: 17,000 tuples

Format: *.npy files for python

Folder structure:
./train: containing 3 folders with robot positions, robot velocities, and lidar scans for training 
./train/positions: containing 67,000 position data in NPY file format for training and a train.txt path file 
./train/velocities: containing 67,000 velocity data in NPY file format for training and a train.txt path file 
./train/scans: containing 67,000 scan data in NPY file format for training and a train.txt path file 
-----------------------------------------------
./val: containing 3 folders with robot positions, robot velocities, and lidar scans for validation
./val/positions: containing 10,891 position data in NPY file format for validation and a val.txt path file 
./val/velocities: containing 10,891 velocity data in NPY file format for validation and a val.txt path file 
./val/scans: containing 10,891 scan data in NPY file format for validation and a val.txt path file
 -----------------------------------------------
./test: containing 3 folders with robot positions, robot velocities, and lidar scans for validation
./test/positions: containing 17,000 position data in NPY file format for testing and a test.txt path file 
./test/velocities: containing 17,000 velocity data in NPY file format for testing and a test.txt path file 
./test/scans: containing 17,000 scan data in NPY file format for testing and a test.txt path file

Contact Information: Zhanteng Xie (zhanteng.xie@temple.edu), Philip Dames (pdames@temple.edu)
