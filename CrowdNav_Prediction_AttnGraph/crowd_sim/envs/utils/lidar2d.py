import numpy as np
from typing import List
from math import atan2, degrees, pi
from skimage.draw import line
import matplotlib.pyplot as plt

class Lidar2d:
    def __init__(self, bitmap: np.ndarray, num_ray: int = 1000, sensor_range: int = -1, cell_length: float = 1, FOV = np.pi/2):
        assert len(bitmap.shape) == 2
        assert cell_length > 0
        assert num_ray > 0
        assert sensor_range >= 0 or sensor_range == -1
        self.cell_length = cell_length
        self.bitmap = bitmap
        self.dynamic_map = np.zeros_like(bitmap)
        self.num_ray = num_ray
        self.FOV = FOV
        self.sensor_range = sensor_range if sensor_range >= 0 else int(np.sqrt(bitmap.shape[0]**2 + bitmap.shape[1]**2))
        self.rr = []
        self.cc = []
        self.length = []
        for ray in range(num_ray):
            angle = 2*np.pi*ray/num_ray
            rx = int(np.cos(angle) * (self.sensor_range) + 0.5)
            ry = int(np.sin(angle) * (self.sensor_range) + 0.5)
            #print(rx,ry)
            rr, cc = line(0,0,rx,ry)
            self.rr.append(rr)
            self.cc.append(cc)
            self.length.append(np.sqrt(rr**2 + cc**2) * self.cell_length )


    def update_dynamic_map(self, human_list: List[List[int]],robot_list: List[List[int]]) -> np.ndarray:
        self.dynamic_map = np.zeros_like(self.bitmap)

        for human in human_list:
            x = int(human[0]/self.cell_length + (self.bitmap.shape[0])/2)
            y = int(human[1]/self.cell_length + (self.bitmap.shape[1])/2)
            for i in range(x-1,x+2):
                for j in range(y-1,y+2):
                    if 0 <= i < self.dynamic_map.shape[0] and 0 <= j < self.dynamic_map.shape[1]:
                        self.dynamic_map[i,j] = 2

        for robot in robot_list:
            x = int(robot[0]/self.cell_length + (self.bitmap.shape[0])/2)
            y = int(robot[1]/self.cell_length + (self.bitmap.shape[1])/2)
            if 0 <= x < self.dynamic_map.shape[0] and 0 <= y < self.dynamic_map.shape[1]:
                self.dynamic_map[x,y] = 3
        return self.dynamic_map
    
    
    
    def get_raw_data(self, x, y, theta = 0) -> np.ndarray:
        result = np.full((self.num_ray,2), np.inf)
        for ray in range(self.num_ray):
            result[ray][1] = 0
            rr = self.rr[ray] + x
            cc = self.cc[ray] + y
            for i in range(1,len(rr)):
                if 0 <= rr[i] < self.bitmap.shape[0] and 0 <= cc[i] < self.bitmap.shape[1]:
                    if self.bitmap[rr[i], cc[i]] > 0:
                        result[ray][0] = self.length[ray][i]
                        result[ray][1] = 1
                        break
                    if self.dynamic_map[rr[i], cc[i]] == 2:
                        result[ray][0] = self.length[ray][i]
                        result[ray][1] = 2
                        break
                    if self.dynamic_map[rr[i], cc[i]] == 3:
                        result[ray][0] = self.length[ray][i]
                        result[ray][1] = 3
                        break
              
        assert 0 <= theta < 2*np.pi


        result = np.roll(result, int(-theta/(2*np.pi)*self.num_ray+0.5), axis=0)

        ray_num_in_FOV = int(self.num_ray * self.FOV / (2*np.pi))
        result[ray_num_in_FOV // 2: -ray_num_in_FOV // 2,1] = np.clip(result[ray_num_in_FOV // 2: -ray_num_in_FOV // 2,1], 0 ,1) 

        return result
    


    def convert_to_bitmap(self, raw_data: np.ndarray, map_size: int) -> np.ndarray:
        assert raw_data.shape[0] == self.num_ray
        local_ogm = np.full((2,map_size,map_size), -1)
        center_index = map_size//2

        for ray in range(self.num_ray):
            # IF no obstacle, set all the cells to 0
            if raw_data[ray][1] == 0: 
                for i in range(1,len(self.rr[ray])):
                    x = self.rr[ray][i] + center_index
                    y = self.cc[ray][i] + center_index
                    if 0 <= x < map_size and 0 <= y < map_size and local_ogm[0,x,y] == -1:
                        local_ogm[1,x,y] = 0
                continue

            for i in range(1,len(self.rr[ray])):
                x = self.rr[ray][i] + center_index
                y = self.cc[ray][i] + center_index
                if 0 <= x < map_size and 0 <= y < map_size:
                    if self.length[ray][i] >= raw_data[ray][0]:
                        local_ogm[0,x,y] = 1
                        local_ogm[1,x,y] = raw_data[ray][1]
                        # x = int(self.rr[ray][i-1] + (map_size-1)/2)
                        # y = int(self.cc[ray][i-1] + (map_size-1)/2)
                        # local_ogm[x,y] = 1
                        break
                    elif local_ogm[0,x,y] != 1:
                        local_ogm[0,x,y] = 0
                        local_ogm[1,x,y] = 0
                else:
                    break
        local_ogm[0] = np.clip(local_ogm[0], 0, 1)

        return local_ogm
    


def merge_ogm(local_ogm, recieved_ogm, relative_position: List[float],theta1 = 0, theta2 = 0,cell_length:float = 1, trust_rate : float = 0.5) -> np.ndarray:
    assert trust_rate >= 0 and trust_rate <= 1
    assert theta1 >= 0 and theta1 < 2*np.pi
    assert theta2 >= 0 and theta2 < 2*np.pi
    #local_ogm = np.copy(local_ogm)

    local_rel_pos_x = relative_position[0] * np.cos(theta1) + relative_position[1] * np.sin(theta1)
    local_rel_pos_y = -relative_position[0] * np.sin(theta1) + relative_position[1] * np.cos(theta1)
    rel_x = (local_rel_pos_x/cell_length)
    rel_y = (local_rel_pos_y/cell_length)
    relative_theta = (theta2 - theta1) % (2*np.pi)
    cos_theta = np.cos(relative_theta)
    sin_theta = np.sin(relative_theta)
    size_shift_x = (local_ogm.shape[1] + recieved_ogm.shape[1]*(sin_theta - cos_theta))/2
    size_shift_y = (local_ogm.shape[2] - recieved_ogm.shape[2]*(sin_theta + cos_theta))/2

    if (abs(2 * rel_x) > local_ogm.shape[1] + recieved_ogm.shape[1]) or (abs(2 *rel_y) > local_ogm.shape[2] + recieved_ogm.shape[2]):
        return local_ogm
    
    for i in range(recieved_ogm.shape[1]):
        for j in range(recieved_ogm.shape[2]):
            if recieved_ogm[1,i,j] == -1 :
                continue
            x = int(i * cos_theta - j * sin_theta + rel_x + size_shift_x + 0.5)
            y = int(i * sin_theta + j * cos_theta + rel_y + size_shift_y + 0.5)
            if 0 <= x < local_ogm.shape[1] and 0 <= y < local_ogm.shape[2]:
                # If the cell in local is unknown, update the cell
                if local_ogm[1,x,y] == -1 or local_ogm[1,x,y] == 1:
                    local_ogm[0,x,y] = recieved_ogm[0,i,j]
                    local_ogm[1,x,y] = recieved_ogm[1,i,j]
                # If the cell in local is known and label is same, update the cell with trust rate
                elif recieved_ogm[1,i,j] == local_ogm[1,x,y]:
                    local_ogm[0,x,y] = trust_rate * recieved_ogm[0,i,j] + (1 - trust_rate) * local_ogm[0,x,y]
    return local_ogm





    
if __name__ == "__main__":
    # Create a sample bitmap
    #bitmap = np.load('./././bitmaps/bitmap_2/bitmap.npy')
    bitmap = np.zeros((64, 64))
    bitmap[32:61, 2:32] = 1
    bitmap[30:35, 30:35] = 1
    # Create a Lidar2d object
    lidar = Lidar2d(bitmap, num_ray=360, sensor_range=24, cell_length=0.311)
    lidar.update_dynamic_map([[-1,-2],[-2,4]],[[5,2]])

    lidar1 = lidar.get_raw_data(27,27,5.6)
    ogm1 = lidar.convert_to_bitmap(lidar1, 56)
    ogm1_ori = np.copy(ogm1)
    lidar2 = lidar.get_raw_data(37,37,2)
    ogm2 = lidar.convert_to_bitmap(lidar2, 48)
    merged_ogm = merge_ogm(ogm1, ogm2, [10,10], 5.6, 2)

    # Create subplots
    fig, axs = plt.subplots(1, 8, figsize=(15, 5))

    # Plot lidar1
    axs[0].imshow(ogm1_ori[0], cmap='gray_r')
    axs[0].set_title('Lidar1')
    axs[0].text(28,28, '1', color='red', fontsize=12, ha='center')
    axs[1].imshow(ogm1_ori[1], cmap='gray_r')
    axs[1].set_title('Lidar1')
    axs[1].text(28,28, '1', color='red', fontsize=12, ha='center')


    # Plot lidar2
    axs[2].imshow(ogm2[0], cmap='gray_r')
    axs[2].set_title('Lidar2')
    axs[2].text(24,24, '2', color='red', fontsize=12, ha='center')
    axs[3].imshow(ogm2[1], cmap='gray_r')
    axs[3].set_title('Lidar2')
    axs[3].text(24,24, '2', color='red', fontsize=12, ha='center')

    # Plot merged lidar
    axs[4].imshow(merged_ogm[0], cmap='gray_r')
    axs[4].set_title('Merged Lidar')
    axs[4].text(28,28, '1', color='red', fontsize=12, ha='center')
    axs[4].text(38,38, '2', color='red', fontsize=12, ha='center')
    axs[5].imshow(merged_ogm[1], cmap='gray_r')
    axs[5].set_title('Merged Lidar')
    axs[5].text(28,28, '1', color='red', fontsize=12, ha='center')
    axs[5].text(38,38, '2', color='red', fontsize=12, ha='center')
    
    # grountruth
    axs[6].imshow(bitmap, cmap='gray_r')
    axs[6].set_title('Groundtruth')
    axs[6].text(27, 27, '1', color='red', fontsize=12, ha='center')
    axs[6].text(37, 37, '2', color='red', fontsize=12, ha='center')
    axs[7].imshow(lidar.dynamic_map, cmap='gray_r')
    axs[7].set_title('Dynamic Map')

    # Show the subplots
    plt.show()


    exit()
    position = [32,0]
    for i in range(64):
        # Change the position
        position = [i,i]
        
        # Get the nearest occupied grid
        nearest_grid = lidar.get_raw_data(position[0], position[1])
        
        # Clear the previous plot
        plt.clf()
        
        # Visualize the bitmap
        # plt.imshow(bitmap, cmap='gray')
        
        # Visualize the result
        # plt.polar(np.deg2rad(range(0, 360)), nearest_grid)
        # plt.title("Lidar2d Result")

        # Visualize the convert to bitmap
        local_ogm = lidar.convert_to_bitmap(nearest_grid, 32)
        plt.imshow(np.flipud(local_ogm), cmap= 'gray_r' )
        plt.show()
        
        # Update the plot
        plt.pause(0.2)
        plt.close()


