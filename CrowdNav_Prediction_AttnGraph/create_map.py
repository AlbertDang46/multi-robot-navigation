import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import medial_axis, dilation
from skimage.draw import line
import networkx as nx
import random
from collections import deque
import os
import shutil
import pickle as pkl


def generate_clustered_bitmap_bfs(size=64, proportion=1/6):
    # generate 2 obstacles in map
    bitmap = np.zeros((size, size), dtype=int)
    num_ones = int(size * size * np.random.uniform(0.04, 0.08))
    for _ in range(2):
        # Initialize a bitmap with zeros
        
        
        # Choose a random starting point
        start_row = random.randint(int(size/5), size - int(size/5))
        start_col = random.randint(int(size/5), size - int(size/5))
        queue = deque([(start_row, start_col)])
        visited = set(queue)
        ones_added = 0

        #generate the random integers to pop from the neighbors in the BFS
        random_int = random.randint(0, 7)
        random_int2 = random.randint(0, 6)
        random_int3 = random.randint(0, 5)
        random_int4 = random.randint(1, 4)
        while ones_added < num_ones and queue:
            row, col = queue.popleft()
            bitmap[row, col] = 1
            ones_added += 1
            
            # Get neighboring cells
            neighbors = [
                (row - 1, col), (row + 1, col),
                (row, col - 1), (row, col + 1),
                (row - 1, col - 1), (row - 1, col + 1),
                (row + 1, col - 1), (row + 1, col + 1),
            ]
            neighbors.pop(random_int)
            neighbors.pop(random_int2)
            # neighbors.pop(random_int3)
            # neighbors.pop(random_int4)
 
            
            for r, c in neighbors:
                if 1 < r < size -2 and 1 < c < size -2  and (r, c) not in visited:
                    if np.random.uniform() < 0.9:
                        queue.append((r, c))
                        visited.add((r, c))

    
    return bitmap

def visualize_bitmap(bitmap, file_name=None):
    plt.figure(figsize=(7, 7))
    plt.imshow(bitmap, cmap='gray_r', interpolation='nearest')
    plt.axis('off')
    
    if file_name:
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    
    plt.show()

def generate_topology_map(image, generated_points_num = 60 , points_num_around_obstacle=40, dilation_num=4):
    dilated_image = np.copy(image)
    for _ in range(dilation_num):
        dilated_image = dilation(dilated_image)
    around_points = np.argwhere(dilation(dilated_image) - dilated_image)
    dilated_image[dilated_image.shape[0]//4:dilated_image.shape[0]//4*3, dilated_image.shape[1]//4:dilated_image.shape[1]//4*3] = 1

    generated_points = []
    topology_graph = nx.Graph()

    # randomly select points in the free space
    for _ in range(generated_points_num):
        point = np.random.randint(0, dilated_image.shape[0]), np.random.randint(0, dilated_image.shape[1])
        while dilated_image[point] == 1 :
            point = np.random.randint(0, dilated_image.shape[0]), np.random.randint(0, dilated_image.shape[1])
        generated_points.append(point)
        topology_graph.add_node(point)
    
    
    # randomly select points around the obstacle
    rand_points = around_points[np.random.choice(np.arange(len(around_points)), points_num_around_obstacle, replace=False)]
    for p in rand_points:
        topology_graph.add_node(tuple(p))
        
    
    # check the visibility of the points in the graph
    for p1 in topology_graph.nodes:
        for p2 in topology_graph.nodes:
            if p1 == p2:
                continue
            rr, cc = line(*p1, *p2)
            if not np.any(image[rr, cc] == 1):
                topology_graph.add_edge(p1, p2, weight=np.linalg.norm(np.array(p1) - np.array(p2)))


    

    # Randomly select two nodes from the topology graph
    start, end = np.random.choice(range(len(topology_graph.nodes)), 2, replace=False)
    start_node = list(topology_graph.nodes)[start]
    end_node = list(topology_graph.nodes)[end]

    # Use Dijkstra's algorithm to find the shortest path between the two nodes
    
    shortest_path = nx.dijkstra_path(topology_graph, start_node, end_node)

    return generated_points, topology_graph

def create_new_map(size=64, proportion=1/6):
    print("The new maps is being created")
    # Create a new folder
    folder_name = "bitmaps"
    os.makedirs(folder_name, exist_ok=True)
    
    # Clear the target folder
    shutil.rmtree(folder_name)
    os.makedirs(folder_name, exist_ok=True)

    # Create 10 new bitmaps and save them in separate subfolders
    for i in range(10):
        subfolder_name = os.path.join(folder_name, f"bitmap_{i}")
        os.makedirs(subfolder_name, exist_ok=True)
        
        # Generate the bitmap using BFS
        random_bitmap_bfs = generate_clustered_bitmap_bfs(size=size, proportion=proportion)

        # Save the bitmap as a numpy file
        bitmap_file = os.path.join(subfolder_name, "bitmap.npy")
        np.save(bitmap_file, random_bitmap_bfs)

        rand_points, topo_graph = generate_topology_map(random_bitmap_bfs)
        # Save the random points as a numpy file
        np.save(os.path.join(subfolder_name, "rand_points.npy"), rand_points)
        # Save the topology graph as an pickle file use pickle 
        pkl.dump(topo_graph, open(os.path.join(subfolder_name, "topology_graph.pkl"), "wb"))
        
        

    
if __name__ == "__main__":
    create_new_map()

    exit()
    # Create a new folder
    folder_name = "bitmaps"
    os.makedirs(folder_name, exist_ok=True)
    
    # Clear the target folder
    shutil.rmtree(folder_name)
    os.makedirs(folder_name, exist_ok=True)

    # Create 10 new bitmaps and save them in separate subfolders
    for i in range(10):
        subfolder_name = os.path.join(folder_name, f"bitmap_{i}")
        os.makedirs(subfolder_name, exist_ok=True)
        
        # Generate the bitmap using BFS
        random_bitmap_bfs = generate_clustered_bitmap_bfs()
        
        # Save the bitmap as a numpy file
        bitmap_file = os.path.join(subfolder_name, "bitmap.npy")
        np.save(bitmap_file, random_bitmap_bfs)
        
        # Save the bitmap image
        image_file = os.path.join(subfolder_name, "bitmap.png")
        visualize_bitmap(random_bitmap_bfs, image_file)

    
    # Get a random number from 0 to 9
    random_number = random.randint(0, 9)

    # Read the static map from the bitmap folder
    bitmap_folder = "bitmaps"
    subfolder_name = f"bitmap_{random_number}"
    bitmap_file = os.path.join("bitmaps", f"bitmap_{random.randint(0, 9)}", "bitmap.npy")
    static_map = np.load(bitmap_file)
    img = plt.imread(os.path.join("bitmaps", f"bitmap_{random.randint(0, 9)}", "bitmap.png"))
    plt.imshow(img)