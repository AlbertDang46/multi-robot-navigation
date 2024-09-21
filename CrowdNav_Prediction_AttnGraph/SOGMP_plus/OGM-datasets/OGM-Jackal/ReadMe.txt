Summary: An occupancy grid map dataset extracted from two sub-datasets of the socially compliant navigation dataset (SCAND), which was collected by the Jackal robot with the maximum speed of 2.0 m/s at the outdoor environment of the UT Austin

Original dataset:
library2pond: library2pond.bag, https://utexas.box.com/shared/static/cgovwgp4phsu0y830o9sb987yhlhjys6.bag  
pond2library: pond2library.bag, https://utexas.box.com/shared/static/y5odsg5pxk2ecgl0qf8yd7l4g1j4mq7c.bag

Data: <robot position, robot velocity, and lidar scan measurement>

Sampling rate: 10 Hz

Size:
Total: 2,280 tuples
Test: 2,280 tuples

Format: *.npy files for python

Folder structure:
./test_library2pond: containing 3 folders with robot positions, robot velocities, and lidar scans for validation
./test_library2pond/positions: containing 980 position data in NPY file format for testing and a test.txt path file 
./test_library2pond/velocities: containing 980 velocity data in NPY file format for testing and a test.txt path file 
./test_library2pond/scans: containing 980 scan data in NPY file format for testing and a test.txt path file
-----------------------------------------------
./test_pond2library: containing 3 folders with robot positions, robot velocities, and lidar scans for validation
./test_pond2library/positions: containing 1300 position data in NPY file format for testing and a test.txt path file 
./test_pond2library/velocities: containing 1300 velocity data in NPY file format for testing and a test.txt path file 
./test_pond2library/scans: containing 1300 scan data in NPY file format for testing and a test.txt path file

Contact Information: Zhanteng Xie (zhanteng.xie@temple.edu), Philip Dames (pdames@temple.edu)
