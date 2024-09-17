import numpy as np 
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

def find_unfilled_with_neighbors(binary_array):
  shifted_up= np.roll(binary_array, shift=-1, axis=0)
  shifted_down= np.roll(binary_array, shift=1, axis=0)
  shifted_left= np.roll(binary_array, shift=-1, axis=1)
  shifted_right= np.roll(binary_array, shift=1, axis=1)

  combined_mask= shifted_up | shifted_down | shifted_left | shifted_right
  # print(combined_mask)
  unfilled_mask= binary_array == 0
  result_mask= unfilled_mask & combined_mask

  coordinates= np.argwhere(result_mask)
  
  result_coordinates= [tuple(coord[::-1]) for coord in coordinates]

  return result_coordinates

def compute_number_of_neighbors(binary_array, candidates, patch_size) -> [np.array, np.uint64] :
  neighbors_count= np.zeros(len(candidates), dtype=np.uint64)
  mask= np.ones((patch_size, patch_size), dtype=np.uint64)
  max_value= 0
  conv_array= convolve(binary_array, mask)
  for i, (x, y) in enumerate(candidates):
    neighbors_count[i]= conv_array[x, y] 

  max_value= np.max(neighbors_count)
  return [neighbors_count, max_value]
  # pad_size= patch_size//2 
  # extended_array= np.pad(binary_array, pad_width=pad_size, mode='constant', constant_values=0)
  # mask= np.ones((patch_size, patch_size), dtype=np.uint64)
  # neighbors_count= np.zeros(len(candidates), dtype=np.uint64)
  # conv_array= np.convolve(binary_array.flatten(), mask.flatten(), mode="same")
  # print(conv_array) 
  # for idx, (x, y) in enumerate(candidates):
  #   patch= extended_array[x-pad_size:x+pad_size+1, y-pad_size:y+pad_size+1] 
  #   neighbors_count[idx]= np.convolve(patch.flatten(), mask.flatten(), mode="valid")
  # max_value = np.max(neighbors_count)
  # return neighbors_count, max_value

def compute_patch_distance(patch_a, patch_b, min_error, epsilon) -> np.float64:
    squared_diff= (patch_a - patch_b) ** 2
    
    cumsum_squared_diff= np.cumsum(squared_diff)
    
    for i in range(cumsum_squared_diff.size):
        if cumsum_squared_diff[i] > (1 + epsilon) * min_error:
            return cumsum_squared_diff[i]
    
    return cumsum_squared_diff[-1]

def find_matching_color(source_array, patch_sample, patch_size, epsilon) -> [np.array, np.array]: 
  min_error= float("inf")
  patch_distances= np.zeros(shape=(source_array.shape[0], source_array.shape[1], 1), dtype=np.float64)
  for x in range(patch_size//2, source_array.shape[0]-patch_size//2): 
    for y in range(patch_size//2, source_array.shape[1]-patch_size//2): 
      patch_source= source_array[x-patch_size//2:x+patch_size//2+1, y-patch_size//2:y+patch_size//2+1]
      patch_source= patch_source.astype(np.float64)
      patch_distances[x, y]= compute_patch_distance(patch_source, patch_sample, min_error, epsilon)
      min_error= min(min_error, patch_distances[x, y])

  candidates_indices= np.where((patch_distances[patch_size//2:source_array.shape[0]-patch_size//2, patch_size//2:source_array.shape[1]-patch_size//2, 0] <= (epsilon + 1) * min_error) & (patch_distances[patch_size//2:source_array.shape[0]-patch_size//2, patch_size//2:source_array.shape[1]-patch_size//2, 0] > 0))
  x_coords, y_coords= candidates_indices
  candidates= list(zip(x_coords, y_coords))

  i= np.random.randint(0, len(candidates))
  return [source_array[candidates[i][0], candidates[i][1]], patch_distances]


# arr= np.array([
#   [0, 0, 0, 0, 0, 0, 0, 0, 0], 
#   [0, 0, 0, 0, 0, 0, 0, 0, 0], 
#   [0, 0, 0, 0, 0, 0, 0, 0, 0], 
#   [0, 0, 0, 1, 1, 1, 0, 0, 0], 
#   [0, 0, 0, 1, 1, 1, 0, 0, 0], 
#   [0, 0, 0, 1, 1, 1, 0, 0, 0], 
#   [0, 0, 0, 0, 0, 0, 0, 0, 0], 
#   [0, 0, 0, 0, 0, 0, 0, 0, 0], 
#   [0, 0, 0, 0, 0, 0, 0, 0, 0], 
# ])

# mask= np.ones((5, 5), dtype=np.uint64)
# extended_arr= np.pad(arr, pad_width=2, mode='constant', constant_values=0)
# conv_array= convolve(extended_arr, mask)
# conv_array= np.convolve(extended_arr.flatten(), mask.flatten(), mode="valid")
# conv_array= np.reshape(conv_array, (extended_arr.shape[0], extended_arr.shape[1]))
# print(conv_array)

# print(np.pad(arr, pad_width=2, mode='constant', constant_values=0))
# print(find_unfilled_with_neighbors(arr))
