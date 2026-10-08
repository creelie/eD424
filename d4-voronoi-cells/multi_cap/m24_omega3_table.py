import os
import numpy as np, time
from m24_volume_functional import omega3_one
h=1/40; g=np.arange(21)*h
T=np.zeros((21,21,21)); t0=time.time()
for i in range(21):
  for j in range(i,21):
    for k in range(j,21):
      a=omega3_one(g[i],g[j],g[k],90); b=omega3_one(g[i],g[j],g[k],180)
      val=(4*b-a)/3
      for p in {(i,j,k),(i,k,j),(j,i,k),(j,k,i),(k,i,j),(k,j,i)}: T[p]=val
np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)),'continuation_out','m24_w3table.npy'),T); print('done',time.time()-t0, T[20,20,20], T.max())
