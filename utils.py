import numpy as np 

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

def distance2(p, q) -> np.float64:
  return (p[0]-q[0])*(p[0]-q[0])+(p[1]-q[1])*(p[1]-q[1])+(p[2]-q[2])*(p[2]-q[2])

def patchDistance(ptchA, ptchB, minErr, eps) -> np.float64:
  dist= 0  
  dim= ptchA.shape[0]
  for x in range(dim):
    for y in range(dim):
      dist= dist+distance2(ptchA[x, y], ptchB[x, y])
      if dist>(1+eps)*minErr: return dist
  return dist

def getMatchingColor(srcArr, ptch, dim, eps) -> [np.array, np.array]: 
  minErr= float("inf")
  dist= np.zeros(shape=(srcArr.shape[0], srcArr.shape[1], 1), dtype=np.float64)
  for x in range(dim//2, srcArr.shape[0]-dim//2): 
    for y in range(dim//2, srcArr.shape[1]-dim//2): 
      ptchA= srcArr[x-dim//2:x+dim//2+1, y-dim//2:y+dim//2+1]
      ptchA= ptchA.astype(np.float64)
      dist[x, y]= patchDistance(ptchA, ptch, minErr, eps)
      minErr= min(minErr, dist[x, y])

  indices = np.where((dist[dim//2:srcArr.shape[0]-dim//2, dim//2:srcArr.shape[1]-dim//2, 0] <= (eps + 1) * minErr) & (dist[dim//2:srcArr.shape[0]-dim//2, dim//2:srcArr.shape[1]-dim//2, 0] > 0))
  x_coords, y_coords = indices
  candidates = list(zip(x_coords, y_coords))
  print(candidates)

  i= np.random.randint(0, len(candidates))
  return [srcArr[candidates[i][0], candidates[i][1]], dist]
