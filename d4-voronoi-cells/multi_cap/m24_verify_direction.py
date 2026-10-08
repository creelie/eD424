"""
m24_verify_direction.py -- recheck a descent direction with the exact triple
overlap.

m24_descent.py works with w3 read off an interpolated table, which is fast
enough to solve with but not a proof of anything.  This reads a saved
direction back, recomputes the objective on its support with the exact
omega3 of m24_volume_functional.py, and walks along the direction in four
steps, reporting at each the least eigenvalue of every block, the least linear
slack and the change in the objective.

  python3 m24_verify_direction.py [degree]               default: 6

It needs continuation_out/m24_dir_d<degree>.npy, which m24_descent.py writes.
"""
import os
import sys, numpy as np
ARGV = sys.argv[1:] or ['6']
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import three_point_sdp as S
from m24_volume_functional import omega3_one, omega2
from m24_primal_blocks import blocks
N=24; one=np.array([1.0])
d=int(ARGV[0]); A=np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),'continuation_out',f'm24_dir_d{d}.npy')); u,v,t,x0,dv=A.T
LC=[S.leg_coeffs(k) for k in range(d+1)]; B,G=blocks(d,u,v,t,LC)
def w3ex(a,b,c):
    if min(a,b,c)<=0: return 0.0
    return (4*omega3_one(a,b,c,180)-omega3_one(a,b,c,90))/3
idx=np.where(np.abs(dv)>1e-7)[0]
cex=np.array([(omega2(u[i])+omega2(v[i])+omega2(t[i]))/(N-2)-w3ex(u[i],v[i],t[i]) for i in idx])
print(f"d={d}: directional derivative with exact omega3: {cex@dv[idx]:.4e}  (support {len(idx)} triples)")
def feas(x):
    worst=np.inf
    for k in range(d+1):
        n=d-k+1; Bk=B[k].reshape(len(u),n,n); M=np.tensordot(x,(Bk+Bk.transpose(0,2,1))/2,1)+N*S.Smat(k,d,one,one,one,LC)[0]
        w=np.linalg.eigvalsh(M); worst=min(worst,w.min()/max(1,np.abs(w).max()))
    lin=min(N+G[:,k]@x for k in range(1,d+1))
    return worst,lin,(x[3:] if False else x).min()
dvc=dv.copy(); dvc[np.abs(dvc)<1e-9]=0
for s_ in (1,10,50,200):
    x=x0+s_*dvc; x=np.maximum(x,0); x*= (N*(N-1)*(N-2)/6)/x.sum()
    wrel,lin,_=feas(x)
    obj=cex@x[idx] + 0  # objective on support only valid if x zero elsewhere
    offsupp=np.setdiff1d(np.where(x>0)[0],idx)
    print(f"  step {s_}: rel min eig {wrel:.2e}, min lin {lin:.2e}, objective change {cex@(x[idx]-x0[idx]):.4e}, stray mass {x[offsupp].sum():.1e}")
