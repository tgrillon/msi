import matplotlib.pyplot as plt
import numpy as np 
import time

import utils
from PIL import Image 

#================INITIALIZATION=================#

# Define patch patch_sizes
patch_size= 5

# Define epsilon error
epsilon= 0.2

# Load source image 
source= Image.open("data/text2.png")
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
#TODO Tant qu'il reste des pixels vides dans l'image destination (destination)
#TODO   Faire l'étape 1 
#TODO   Tant qu'il reste des pixels vides dans edg
#TODO     Faire l'étape 2 
#===============================================#

elapsed_time_0 = time.time()

binary_array = np.any(destination_array != [0, 0, 0], axis=-1).astype(int)
edges= utils.find_unfilled_with_neighbors(binary_array)

while len(edges) > 0: 
  neighbors, max_value= utils.compute_number_of_neighbors(binary_array, edges, patch_size)
  print(neighbors)
  print("max_value: ", max_value)
  break

elapsed_time_1 = time.time()
print("Step 1: ", elapsed_time_1-elapsed_time_0)


# ngh, maxN= utils.numberOfNeighbors(binary_array, edg, patch_size)

# elapsed_time_0 = time.time()

# s= 0.95
# ths= s*maxN

# # Get the index of pixels with most neighbors 
# ids= np.where(ngh > ths)  
# ids= ids[0]

# i= np.random.randint(0, ids.shape[0]-1)
# ptch= destination_array[edg[ids[i]][0]-patch_size//2:edg[ids[i]][0]+patch_size//2+1, edg[ids[i]][1]-patch_size//2:edg[ids[i]][1]+patch_size//2+1] 
# ptch= ptch.astype(np.float64)

# color, dist= utils.getMatchingColor(source_array, ptch, patch_size, epsilon)
# print(color)

# elapsed_time_1 = time.time()
# print("Step 2: ", elapsed_time_1-elapsed_time_0)

# fig, axes = plt.subplots(1, 3, figsize=(10, 10))
# axes = axes.flatten()

# axes[0].imshow(ptch.astype(np.uint8), cmap='gray', interpolation='none')
# axes[0].axis('off')  # Hide axes

# axes[1].imshow(source_array, cmap='gray', interpolation='none')
# axes[1].axis('off')  # Hide axes

# axes[2].imshow(dist, cmap='gray', interpolation='none')
# axes[2].axis('off')  # Hide axes

# plt.tight_layout()
# plt.show()
