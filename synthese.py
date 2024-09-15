import matplotlib.pyplot as plt
import numpy as np 
import time

import utils
from PIL import Image 

#================INITIALIZATION=================#

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
#===============================================#
#TODO Tant qu'il reste des pixels vides dans l'image destination (dst)
#TODO   Faire l'étape 1 
#TODO   Tant qu'il reste des pixels vides dans edg
#TODO     Faire l'étape 2 
#===============================================#

t0 = time.time()

edg= utils.getEdges(dstArr)

bin = np.any(dstArr != [0, 0, 0], axis=-1).astype(int)

ngh, maxN= utils.numberOfNeighbors(bin, edg, dim)

t1 = time.time()
print("Step 1: ", t1-t0)
t0 = time.time()

s= 0.95
ths= s*maxN

# Get the index of pixels with most neighbors 
ids= np.where(ngh > ths)  
ids= ids[0]

i= np.random.randint(0, ids.shape[0]-1)
ptch= dstArr[edg[ids[i]][0]-dim//2:edg[ids[i]][0]+dim//2+1, edg[ids[i]][1]-dim//2:edg[ids[i]][1]+dim//2+1] 
ptch= ptch.astype(np.float64)

color, dist= utils.getMatchingColor(srcArr, ptch, dim, eps)
print(color)

t1 = time.time()
print("Step 2: ", t1-t0)

# fig, axes = plt.subplots(1, 3, figsize=(10, 10))
# axes = axes.flatten()

# axes[0].imshow(ptch.astype(np.uint8), cmap='gray', interpolation='none')
# axes[0].axis('off')  # Hide axes

# axes[1].imshow(srcArr, cmap='gray', interpolation='none')
# axes[1].axis('off')  # Hide axes

# axes[2].imshow(dist, cmap='gray', interpolation='none')
# axes[2].axis('off')  # Hide axes

# plt.tight_layout()
# plt.show()
