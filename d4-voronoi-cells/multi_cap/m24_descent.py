"""
m24_descent.py -- is the three-point relaxation of the 24-contact case at a
local minimum, or can the D_4 measure be pushed down?

Takes the D_4 triples with their multiplicities as the starting point and
solves, at each truncation degree, for the direction of steepest descent
inside the cone the relaxation allows: a signed measure summing to zero, with
nonnegative weight off the D_4 support, subject to the linear constraints that
are tight there and to the semidefinite constraints on the kernels of the
blocks at the starting point.  A minimum directional derivative of zero says
the D_4 measure is not improvable at that degree; a negative one names the
triples that would take weight.

  python3 m24_descent.py [grid_n grid_m degree ...]      default: 12 8 6

The saved directions are read back by m24_verify_direction.py.
"""
import os
import sys, time, numpy as np, cvxpy as cp
ARGV = sys.argv[1:] or ['12', '8', '6']
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import three_point_sdp as S
from m24_dual_sdp import w3, omega2, tau
from m24_primal_blocks import blocks, d4_triples
N=24; one=np.array([1.0])
c=d4_triples(); P=np.array(list(c.keys())); cnt=np.array(list(c.values()),float)
gu,gv,gt=S.sample_grid(int(ARGV[0]),int(ARGV[1]))
# drop grid points coinciding with D4 types
key=lambda a,b,e: tuple(np.round(sorted([a,b,e]),6))
D4set={tuple(p) for p in np.round(P,6)}
m=np.array([key(a,b,e) not in D4set for a,b,e in zip(gu,gv,gt)])
u=np.r_[P[:,0],gu[m]]; v=np.r_[P[:,1],gv[m]]; t=np.r_[P[:,2],gt[m]]; ns=len(P)
x0=np.r_[cnt,np.zeros(m.sum())]
cost=(omega2(u)+omega2(v)+omega2(t))/(N-2)-w3(u,v,t)
print("D4 objective",cost@x0,"tau",tau)
for d in map(int,ARGV[2:]):
    LC=[S.leg_coeffs(k) for k in range(d+1)]
    B,G=blocks(d,u,v,t,LC); t0=time.time()
    dl=cp.Variable(len(u)); cons=[cp.sum(dl)==0, dl[ns:]>=0, cp.sum(dl[ns:])<=1, cp.abs(dl[:ns])<=50]
    kd=[]
    for k in range(1,d+1):
        if abs(N+G[:,k]@x0)<1e-8: cons.append(G[:,k]@dl>=0)
    for k in range(d+1):
        n=d-k+1; Bk=B[k].reshape(len(u),n,n); Bk=(Bk+Bk.transpose(0,2,1))/2
        M0=np.tensordot(x0,Bk,1)+N*S.Smat(k,d,one,one,one,LC)[0]
        w,V=np.linalg.eigh((M0+M0.T)/2); K=V[:,w<1e-8*max(1,np.abs(w).max())]
        kd.append(K.shape[1])
        if K.shape[1]==0: continue
        R=np.einsum('ai,pij,jb->pab',K.T,Bk,K); s=np.abs(R).max() or 1
        q=K.shape[1]; Mk=cp.reshape((R/s).reshape(len(u),-1).T@dl,(q,q),order='C')
        cons.append((Mk+Mk.T)/2>>0)
    pr=cp.Problem(cp.Minimize(cost@dl),cons)
    try: pr.solve(solver='CLARABEL',max_iter=300)
    except Exception: pr.solve(solver='SCS',eps=1e-8,max_iters=20000)
    dv=dl.value
    top=np.argsort(dv[ns:])[-5:][::-1]+ns
    print(f"d={d}: kernel dims {kd}; min directional derivative {pr.value:.3e} [{pr.status}, {time.time()-t0:.0f}s]",flush=True)
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)),'continuation_out',f'm24_dir_d{d}.npy'),np.c_[u,v,t,x0,dv])
    if dv is not None and pr.value<-1e-7:
        print("   support shift:",[(tuple(np.round(P[i],2)),round(float(dv[i]),3)) for i in range(ns) if abs(dv[i])>1e-4])
        print("   new triples:",[(tuple(np.round([u[i],v[i],t[i]],3)),round(float(dv[i]),3)) for i in top])
