import matplotlib.pyplot as plt
import numpy as np 
from PIL import Image 

# Load source image 
src= Image.open("data/text2.png")
srcArray= np.array(src) 

# Define destination image
dst= Image.new(size=(src.size[0]*2, src.size[1]*2), mode="RGB") 
dstArr= np.array(dst)

# Define patch dimensions
dim= 23  

# Initisalize I_smp
x= np.random.randint(0, srcArray.shape[0]-dim-1)
y= np.random.randint(0, srcArray.shape[1]-dim-1)
smp= srcArray[x:x+dim, y:y+dim]

# Initialize I with I_smp
x= np.random.randint(0, dstArr.shape[0]-dim-1)
y= np.random.randint(0, dstArr.shape[1]-dim-1)
# dstArr[x:x+dim, y:y+dim]= smp
dstArr[dim//4:dim//4+dim, dim+dim//4:2*dim+dim//4]= smp

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
  bin= np.zeros((arr.shape[0], arr.shape[1], 1), dtype=np.uint8)
  for x in range(arr.shape[0]):
    for y in range(arr.shape[1]):
      if filled(arr[x, y]):
        bin[x, y]= 1
      else: 
        bin[x, y]= 0
  return bin

def extractSquareArroundP(arr, p, dim) -> [np.array, np.uint64, np.uint64, np.uint64, np.uint64]:
  minx= max((0, p[0]-dim//2))
  maxx= min((arr.shape[0], p[0]+dim//2))
  miny= max((0, p[1]-dim//2))
  maxy= min((arr.shape[1], p[1]+dim//2))
  return arr[p[0]-dim//2:p[0]+dim//2, p[1]-dim//2:p[1]+dim//2], minx, maxx, miny, maxy

def numberOfNeighbors(arr, edg, dim) -> [np.array, np.uint64] :
  ret= np.empty(shape=(0,1), dtype=np.float64)
  maxN= 0
  for e in edg:
    smp, minx, maxx, miny, maxy= extractSquareArroundP(arr, e, dim)
    msk= np.ones((maxx-minx, maxy-miny, 1), dtype=np.uint8)

    # Compute and store the number of neighbors filled 
    cnv= np.convolve(smp.flatten(), msk.flatten(), mode="valid")
    ret= np.append(ret, np.array(cnv))
    
    # Update max value 
    maxN= max(maxN, cnv)
    
  return [ret, maxN]

edg= getEdges(dstArr)
bin= binaryArray(dstArr)

ngh, maxN= numberOfNeighbors(bin, edg, dim)

s= 0.90
ths= s*maxN


# Get the index of pixels with most neighbors 
ids= np.where(ngh > ths)  
ids= ids[0]
print(ngh)
print(ids)

ridx= np.random.randint(0, ids.shape[0]-1)
print(edg[ridx])
patch= extractSquareArroundP(dstArr, edg[ridx], dim)[0]

# Number of patches
num_patches = len(ids)

# Determine grid size (e.g., 2x2 grid for 4 patches)
grid_size = int(np.ceil(np.sqrt(num_patches)))

# Create subplots
fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Plot each patch
for i in range(num_patches):
  patch= extractSquareArroundP(dstArr, edg[ids[i]], dim)[0]
  axes[i].imshow(patch, cmap='gray', interpolation='none')
  axes[i].axis('off')  # Hide axes

plt.tight_layout()
plt.show()

plt.imshow(dstArr)
plt.show()


# plt.imshow(binaryArray(dstArr))
# plt.show()

# print(binaryArray(dstArr))


