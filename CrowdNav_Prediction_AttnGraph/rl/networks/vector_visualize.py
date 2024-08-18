import numpy as np

import matplotlib.pyplot as plt

def tensor_to_map(tensor, mapsize, save_path):
    # Convert tensor to numpy array
    array = tensor.cpu().numpy()

    # Normalize array values between 0 and 1
    array = (array - np.min(array)) / (np.max(array) - np.min(array))

    # Reshape array to match mapsize
    array = np.reshape(array, mapsize)

    # Create grayscale map
    plt.imshow(array, cmap='gray')

    # Save the grayscale map directly without displaying it
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
