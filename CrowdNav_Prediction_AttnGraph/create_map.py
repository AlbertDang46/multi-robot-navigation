import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import os
import shutil


def generate_clustered_bitmap_bfs(size=64, proportion=1/6):
    # generate 2 obstacles in map
    bitmap = np.zeros((size, size), dtype=int)
    num_ones = int(size * size * np.random.uniform(0.02, 0.05))
    for _ in range(3):
        # Initialize a bitmap with zeros
        
        
        # Choose a random starting point
        start_row = random.randint(int(size/4), size - int(size/4))
        start_col = random.randint(int(size/4), size - int(size/4))
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


def create_new_map():
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
        random_bitmap_bfs = generate_clustered_bitmap_bfs()
        
        # Save the bitmap as a numpy file
        bitmap_file = os.path.join(subfolder_name, "bitmap.npy")
        np.save(bitmap_file, random_bitmap_bfs)


    
if __name__ == "__main__":

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