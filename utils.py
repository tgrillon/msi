import numpy as np 
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from numpy.lib.stride_tricks import as_strided

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
  pad_size= patch_size//2
  extended_array= np.pad(binary_array, pad_width=pad_size, mode='constant', constant_values=0)
  max_value= 0
  conv_array= convolve(extended_array, mask)
  for i, (x, y) in enumerate(candidates):
    neighbors_count[i]= conv_array[y+pad_size, x+pad_size] 

  max_value= np.max(neighbors_count)
  return [neighbors_count, max_value]

def compute_patch_distance(patch_a, patch_b, min_error, epsilon) -> np.float64:
    patch_a_flat = patch_a.reshape(-1, 3)
    patch_b_flat = patch_b.reshape(-1, 3)

    mask= np.any(patch_a_flat.reshape(-1, 3) != 0, axis=1)
    
    masked_a= patch_a_flat[mask]
    masked_b= patch_b_flat[mask]
    
    squared_diff= (masked_a - masked_b) ** 2
    
    cumsum_squared_diff= np.cumsum(squared_diff)
    
    for i in range(cumsum_squared_diff.size):
        if cumsum_squared_diff[i] > (1 + epsilon) * min_error:
            return cumsum_squared_diff[i]
    
    return cumsum_squared_diff[-1]

def find_matching_color(source_array, patch_source, patch_size, epsilon) -> [np.array, np.array]: 
  pad_size= patch_size//2
  extended_array = np.pad(
    source_array, 
    pad_width=((pad_size, pad_size), (pad_size, pad_size), (0, 0)), 
    mode='constant', 
    constant_values=0
  )
  
  shape=(source_array.shape[0], source_array.shape[1], patch_size, patch_size, 3)
  strides=extended_array.strides[:2] + extended_array.strides
  patches = as_strided(extended_array, shape=shape, strides=strides)

  patches_f = patches.astype(np.float64)
  patch_source_f = patch_source.astype(np.float64)
  
  patch_distances= np.zeros(shape=(source_array.shape[0], source_array.shape[1]), dtype=np.float64)
  min_error= float("inf")
  for x in range(0, source_array.shape[0]): 
    for y in range(0, source_array.shape[1]): 
      patch_sample= patches_f[x, y]
      distance= compute_patch_distance(patch_source_f, patch_sample, min_error, epsilon)
      patch_distances[x, y]= distance
      if min_error > distance:
         min_error= distance

  candidates_indices= np.where((patch_distances[:, :] <= (epsilon + 1) * min_error))
  x_coords, y_coords= candidates_indices
  candidates= list(zip(x_coords, y_coords))

  if len(candidates) == 0:
    return [None, patch_distances]

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
