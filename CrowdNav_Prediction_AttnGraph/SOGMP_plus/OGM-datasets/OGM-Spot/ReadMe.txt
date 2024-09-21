Summary: An occupancy grid map dataset extracted from two sub-datasets of the socially compliant navigation dataset (SCAND), which was collected by the Spot robot with the maximum speed of 1.6 m/s at the Union Building of the UT Austin

Original dataset:
Union1: 2021-11-10-11-09-14.bag, https://utexas.box.com/shared/static/w80vefrx0kihzrt711l71xi81q3lb5co.bag  
Union2: 2021-11-10-12-26-50.bag, https://utexas.box.com/shared/static/3g6kf1k294zcug37uoaat66u7gkihcml.bag

Data: <robot position, robot velocity, and lidar scan measurement>

Sampling rate: 10 Hz

Size:
Total: 1,890 tuples
Test: 1,890 tuples

Format: *.npy files for python

Folder structure:
./test_union1: containing 3 folders with robot positions, robot velocities, and lidar scans for validation
./test_union1/positions: containing 1,000 position data in NPY file format for testing and a test.txt path file 
./test_union1/velocities: containing 1,000 velocity data in NPY file format for testing and a test.txt path file 
./test_union1/scans: containing 1,000 scan data in NPY file format for testing and a test.txt path file
-----------------------------------------------
./test_union2: containing 3 folders with robot positions, robot velocities, and lidar scans for validation
./test_union2/positions: containing 890 position data in NPY file format for testing and a test.txt path file 
./test_union2/velocities: containing 890 velocity data in NPY file format for testing and a test.txt path file 
./test_union2/scans: containing 890 scan data in NPY file format for testing and a test.txt path file

Contact Information: Zhanteng Xie (zhanteng.xie@temple.edu), Philip Dames (pdames@temple.edu)
