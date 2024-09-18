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
epsilon= 0.02

# Load source image 
source= Image.open("data/text0.png")
source_array= np.array(source)

# Define destination image
height, width= 32, 32
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

def process_pixel():
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

result_image = plt.imshow(destination_array, cmap='gray', interpolation='none')
plt.pause(0.0001)

check_time= True
while len(edges) > 0: 
  if check_time:
    check_time_0 = time.time()
  
  process_pixel()
  binary_array = np.any(destination_array != [0, 0, 0], axis=-1).astype(int)
  edges= utils.find_unfilled_with_neighbors(binary_array)

  if check_time:
    check_time_1 = time.time()
    check_time= False

  result_image.remove()
  result_image = plt.imshow(destination_array)
  plt.pause(0.000000001)

  #=======Print infos=======#
  remaining_pixels= np.count_nonzero(np.all(destination_array==[0,0,0],axis=2))
  completion_percentage= 100.*(1-remaining_pixels/(width*height))
  output_completion = f"[Remaining pixels: {remaining_pixels}, Completion: {completion_percentage:.2f}%]"

  estimated_time= (check_time_1-check_time_0)*remaining_pixels
  output_estimation = f"[Estimated time: {estimated_time:.2f}sec/{estimated_time/60.:.2f}min]"
  sys.stdout.write('\r' + output_completion + '----' + output_estimation)
  sys.stdout.flush()

elapsed_time_1 = time.time()
print("\nTotal elapsed time: ", elapsed_time_1-elapsed_time_0)

fig, axes = plt.subplots(1, 3, figsize=(10, 10))
axes = axes.flatten()

axes[0].imshow(patch_sample, cmap='gray', interpolation='none')
axes[0].set_title('Patch sample')
axes[0].axis('off')  # Hide axes

axes[1].imshow(source_array, cmap='gray', interpolation='none')
axes[1].set_title('Input image')
axes[1].axis('off')  # Hide axes

axes[2].imshow(destination_array, cmap='gray', interpolation='none')
axes[2].set_title('Output image')
axes[2].axis('off')  # Hide axes

plt.tight_layout()
random_number = np.random.randint(1000, 9999)
file_name = f"output_figure_{width}x{height}_p{patch_size}_{random_number}.png"
plt.savefig(file_name, format='png', dpi=300)  
plt.show()

output_image = Image.fromarray(destination_array)
file_name = f"output_image_{width}x{height}_p{patch_size}_{random_number}.png"
output_image.save(file_name)
