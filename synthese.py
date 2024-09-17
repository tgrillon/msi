import matplotlib.pyplot as plt
import numpy as np 
import time
import sys

import utils
from PIL import Image 

#================INITIALIZATION=================#

elapsed_time_0 = time.time()

# Define patch patch_sizes
patch_size= 5
pad_size= patch_size//2

# Define epsilon error
epsilon= 0.2

# Load source image 
source= Image.open("data/text4.png")
source_array= np.array(source)

# Define destination image
height, width= 64, 64
destination_array= np.zeros(shape=(width, height, 3), dtype=np.uint8) 

# Initisalize I_smp
x= np.random.randint(0, source_array.shape[0]-patch_size)
y= np.random.randint(0, source_array.shape[1]-patch_size)
patch_sample= source_array[x:x+patch_size, y:y+patch_size]

# Initialize I with I_smp
x= np.random.randint(0, destination_array.shape[0]-patch_size)
y= np.random.randint(0, destination_array.shape[1]-patch_size)

destination_array[x:x+patch_size, y:y+patch_size]= patch_sample

#===============================================#
#===============================================#

binary_array= np.any(destination_array != [0, 0, 0], axis=-1).astype(int)
edges= utils.find_unfilled_with_neighbors(binary_array)

while len(edges) > 0: 
  remaining_pixels= np.count_nonzero(np.all(destination_array==[0,0,0],axis=2))
  completion_percentage= 100.*(1-remaining_pixels/(width*height))
  output = f"Remaining pixels: {remaining_pixels}, Completion: {completion_percentage:.2f}%"
    
  sys.stdout.write('\r' + output)
  sys.stdout.flush()
  neighbors, max_value= utils.compute_number_of_neighbors(binary_array, edges, patch_size)
  threshold= (1-epsilon)*max_value

  # Get the index of pixels with most neighbors 
  indices= np.where(neighbors > threshold)[0]  
  random_i= np.random.randint(0, indices.shape[0])
  y, x= edges[indices[random_i]]
  extended_array = np.pad(destination_array, 
                        pad_width=((pad_size, pad_size), (pad_size, pad_size), (0, 0)), 
                        mode='constant', 
                        constant_values=0)
  patch= extended_array[x:x+patch_size, y:y+patch_size]

  destination_array[x, y], distances= utils.find_matching_color(source_array, patch, patch_size, epsilon)

  binary_array = np.any(destination_array != [0, 0, 0], axis=-1).astype(int)
  edges= utils.find_unfilled_with_neighbors(binary_array)

elapsed_time_1 = time.time()
print("\nTotal elapsed time: ", elapsed_time_1-elapsed_time_0)

fig, axes = plt.subplots(1, 3, figsize=(10, 10))
axes = axes.flatten()

axes[0].imshow(patch.astype(np.uint8), cmap='gray', interpolation='none')
axes[0].axis('off')  # Hide axes

axes[1].imshow(destination_array, cmap='gray', interpolation='none')
axes[1].axis('off')  # Hide axes

axes[2].imshow(source_array, cmap='gray', interpolation='none')
axes[2].axis('off')  # Hide axes

plt.tight_layout()
plt.show()
