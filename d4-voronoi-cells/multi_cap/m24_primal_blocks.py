"""
m24_primal_blocks.py -- the block structure of the three-point relaxation, and
the primal problem written with it.

blocks(d, u, v, t, LC) returns the semidefinite blocks and the linear
functionals of the degree-d three-point relaxation at the given triples, in
the basis of Legendre coefficients that three_point_sdp.py sets up;
d4_triples() returns the triples of the D_4 root system with the number of
ordered triples of roots realising each.  Run as a program it first checks the
blocks against the D_4 measure at three degrees, then minimises over a sampled
set of triples.

  python3 m24_primal_blocks.py [grid_n grid_m degree ...]   default: 12 8 6
"""
import os
import sys, time, numpy as np, cvxpy as cp
ARGV = sys.argv[1:] or ['12', '8', '6']
from itertools import combinations
from collections import Counter
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import three_point_sdp as S
from m24_dual_sdp import w3, omega2, tau
from m24_volume_functional import roots
N=24
def blocks(d,u,v,t,LC):
    one=np.ones_like(u)
    B=[]
    for k in range(d+1):
        M=6*S.Smat(k,d,u,v,t,LC)+6*(S.Smat(k,d,one,u,u,LC)+S.Smat(k,d,one,v,v,LC)+S.Smat(k,d,one,t,t,LC))/(N-2)
        B.append(M)
    G=np.stack([2*(S.gegen(k,u)+S.gegen(k,v)+S.gegen(k,t))/(N-2) for k in range(d+1)],1)
    return B,G
def d4_triples():
    W=roots(); Gm=W@W.T
    c=Counter()
    for i,j,k in combinations(range(24),3):
        c[tuple(sorted(np.round([Gm[i,j],Gm[i,k],Gm[j,k]],6)))]+=1
    return c
def run(d,u,v,t,x_fixed=None):
    LC=[S.leg_coeffs(k) for k in range(d+1)]
    B,G=blocks(d,u,v,t,LC)
    one=np.array([1.0])
    cost=(omega2(u)+omega2(v)+omega2(t))/(N-2)-w3(u,v,t)
    if x_fixed is not None:
        x=x_fixed
        mins=[np.linalg.eigvalsh(np.tensordot(x,B[k],1)+N*S.Smat(k,d,one,one,one,LC)[0]).min() for k in range(d+1)]
        lin=[N+x@G[:,k] for k in range(1,d+1)]
        return cost@x, min(mins), min(lin)
    T3=N*(N-1)*(N-2)/6
    p=cp.Variable(len(u),nonneg=True)
    cons=[cp.sum(p)==1]
    for k in range(1,d+1): cons.append((N+T3*G[:,k]@p)/N>=0)
    for k in range(d+1):
        n=d-k+1
        Bk=B[k].reshape(len(u),n,n)*T3
        C0=N*S.Smat(k,d,one,one,one,LC)[0]
        dg=np.sqrt(np.maximum(np.abs(np.einsum('pii->pi',Bk)).max(0),1e-12))
        Dm=1/np.maximum(dg,1e-6)
        Bs=(Bk*Dm[None,:,None]*Dm[None,None,:]).reshape(len(u),-1); Cs=C0*Dm[:,None]*Dm[None,:]
        Mv=cp.reshape(Bs.T@p,(n,n),order='C')+Cs
        cons.append((Mv+Mv.T)/2>>0)
    pr=cp.Problem(cp.Minimize(T3*(cost@p)),cons); t0=time.time()
    try: pr.solve(solver='CLARABEL',max_iter=300)
    except Exception: pr.solve(solver='SCS',eps=1e-7,max_iters=50000)
    x=T3*p.value
    return pr.value, pr.status, x, time.time()-t0
if __name__=="__main__":
    c=d4_triples(); P=np.array(list(c.keys())); cnt=np.array(list(c.values()),float)
    print("D4 triple types:",dict(c))
    for d in (6,8,12):
        val,mineig,minlin=run(d,P[:,0],P[:,1],P[:,2],cnt)
        print(f"sanity d={d}: D4 objective {val:.6f} (tau {tau:.6f}), min eig {mineig:.2e}, min lin {minlin:.2e}")
    u,v,t=S.sample_grid(int(ARGV[0]),int(ARGV[1]))
    u,v,t=np.r_[u,P[:,0]],np.r_[v,P[:,1]],np.r_[t,P[:,2]]
    for d in map(int,ARGV[2:]):
        val,st,x,el=run(d,u,v,t)
        top=np.argsort(x)[-8:][::-1]
        chk=run(d,u,v,t,x_fixed=x)
        print(f"   recheck: objective {chk[0]:.6f}, min eig {chk[1]:.2e}, min lin {chk[2]:.2e}",flush=True)
        print(f"d={d}: primal min {val:.6f} vs tau {tau:.6f} gap {tau-val:.5f} ({st}, {len(u)} triples, {el:.0f}s)",flush=True)
        print("   heaviest triples:",[(tuple(np.round([u[i],v[i],t[i]],3)),round(float(x[i]),1)) for i in top],flush=True)
