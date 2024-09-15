import matplotlib.pyplot as plt
import numpy as np 
import time
from PIL import Image 

# Define patch dimensions
dim= 5

# Define epsilon error
eps= 0.2

# Load source image 
src= Image.open("data/text2.png")
srcBis= Image.new(size=(src.size[0]+dim-1, src.size[1]+dim-1), mode="RGB") 

srcArr= np.array(srcBis)
srcArr[dim//2:dim//2+src.size[1], dim//2:dim//2+src.size[0]]= np.array(src) 

# Define destination image
h, w= 250, 250
dst= Image.new(size=(w+dim-1, h+dim-1), mode="RGB") 
dstArr= np.array(dst)

# Initisalize I_smp
x= np.random.randint(dim//2, srcArr.shape[0]-dim-1-dim//2)
y= np.random.randint(dim//2, srcArr.shape[1]-dim-1-dim//2)
smp= srcArr[x:x+dim, y:y+dim]

# Initialize I with I_smp
x= np.random.randint(dim//2, dstArr.shape[0]-dim-1-dim//2)
y= np.random.randint(dim//2, dstArr.shape[1]-dim-1-dim//2)

dstArr[x:x+dim, y:y+dim]= smp

# plt.imshow(dstArr)
# plt.show()

# Pick a pixel p not yet filled with a maximal number of filled neighbors 
# p= [x, y]

def filled(p) -> bool : 
  return p[0]>0 or p[1]>0 or p[2]>0  

def hasANeighbor(arr, x, y) -> bool:
  return x+1 < arr.shape[0] and filled(arr[x+1, y]) or x-1 >= 0 and filled(arr[x-1, y]) or y+1 < arr.shape[1] and filled(arr[x, y+1]) or y-1 >= 0 and filled(arr[x, y-1])

def getEdges(arr) -> np.array:
  edg= np.empty((0,2), dtype=np.int64)
  for x in range(arr.shape[0]):
    for y in range(arr.shape[1]):
      if not filled(arr[x, y]) and hasANeighbor(arr, x, y):
        edg= np.append(edg, np.array([[x, y]]), axis=0)
  return edg

def binaryArray(arr) -> np.array:
  bin= np.zeros((arr.shape[0], arr.shape[1], 1), dtype=np.uint64)
  for x in range(arr.shape[0]):
    for y in range(arr.shape[1]):
      if filled(arr[x, y]):
        bin[x, y]= 1
      else: 
        bin[x, y]= 0
  return bin

def numberOfNeighbors(arr, edg, dim) -> [np.array, np.uint64] :
  ret= np.empty(shape=(0,1), dtype=np.float64)
  msk= np.ones((dim, dim, 1), dtype=np.uint64)
  maxN= 0
  for e in edg:
    smp= arr[e[0]-dim//2:e[0]+dim//2+1, e[1]-dim//2:e[1]+dim//2+1]
    # Compute and store the number of neighbors filled  
    cnv= np.convolve(smp.flatten(), msk.flatten(), mode="valid")
    ret= np.append(ret, np.array(cnv))

    # Update max value 
    maxN= max(maxN, cnv)
    
  return [ret, maxN]

edg= getEdges(dstArr)
bin= binaryArray(dstArr)

ngh, maxN= numberOfNeighbors(bin, edg, dim)

s= 0.95
ths= s*maxN

# Get the index of pixels with most neighbors 
ids= np.where(ngh > ths)  
ids= ids[0]

i= np.random.randint(0, ids.shape[0]-1)
ptch= dstArr[edg[ids[i]][0]-dim//2:edg[ids[i]][0]+dim//2+1, edg[ids[i]][1]-dim//2:edg[ids[i]][1]+dim//2+1] 
ptch= ptch.astype(np.float64)

# # Number of patches
# num_patches = len(ids)

# # Determine grid size (e.g., 2x2 grid for 4 patches)
# grid_size = int(np.ceil(np.sqrt(num_patches)))

# # Create subplots
# fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))

# # Flatten the axes array for easy iteration
# axes = axes.flatten()

# # Plot each patch
# for i in range(num_patches):
#   ptch= dstArr[edg[ids[i]][0]-dim//2:edg[ids[i]][0]+dim//2+1, edg[ids[i]][1]-dim//2:edg[ids[i]][1]+dim//2+1]
#   axes[i].imshow(ptch, cmap='gray', interpolation='none')
#   axes[i].axis('off')  # Hide axes

# plt.tight_layout()
# plt.show()

# plt.imshow(ptch)
# plt.show()

def distance2(p, q) -> np.float64:
  return (p[0]-q[0])*(p[0]-q[0])+(p[1]-q[1])*(p[1]-q[1])+(p[2]-q[2])*(p[2]-q[2])

def patchDistance(ptchA, ptchB, minErr, eps) -> np.float64:
  dist= 0  
  dim= ptchA.shape[0]
  for x in range(dim):
    for y in range(dim):
      dist= dist+distance2(ptchA[x, y], ptchB[x, y])
      # print(min)
      if dist>(1+eps)*minErr: return dist
  return dist

t0 = time.time()

minErr= float("inf")
dist= np.zeros(shape=(srcArr.shape[0], srcArr.shape[1], 1), dtype=np.float64)
for x in range(dim//2, srcArr.shape[0]-dim//2): 
  for y in range(dim//2, srcArr.shape[1]-dim//2): 
    ptchA= srcArr[x-dim//2:x+dim//2+1, y-dim//2:y+dim//2+1]
    ptchA= ptchA.astype(np.float64)
    dist[x, y]= patchDistance(ptchA, ptch, minErr, eps)
    # print(minErr, dist[x, y])
    minErr= min(minErr, dist[x, y])

t1 = time.time()
print("timestamp: ", t1-t0)

fig, axes = plt.subplots(1, 3, figsize=(10, 10))
axes = axes.flatten()

axes[0].imshow(ptch.astype(np.uint8), cmap='gray', interpolation='none')
axes[0].axis('off')  # Hide axes

axes[1].imshow(srcArr, cmap='gray', interpolation='none')
axes[1].axis('off')  # Hide axes

axes[2].imshow(dist, cmap='gray', interpolation='none')
axes[2].axis('off')  # Hide axes

plt.tight_layout()
plt.show()

