"""
m24_primal_sdp.py -- the primal three-point relaxation of the 24-contact case,
solved with a margin.

Minimises the volume functional over measures on admissible triples subject to
the relaxation's constraints held with a slack eps, which keeps the solver
inside the cone rather than on its boundary, and rechecks the returned measure
against the constraints without the slack.  The gap it reports is tau minus the
value: positive means the relaxation does not reach what the 24-contact case
needs at that degree.

  python3 m24_primal_sdp.py [grid_n grid_m eps degree ...]  default: 12 8 1e-6 6

The measures are saved for m24_descent.py and m24_verify_direction.py.
"""
import os
import sys, time, numpy as np, cvxpy as cp
ARGV = sys.argv[1:] or ['12', '8', '1e-6', '6']
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import three_point_sdp as S
from m24_dual_sdp import w3, omega2, tau
from m24_primal_blocks import blocks, d4_triples
N=24; T3=N*(N-1)*(N-2)/6; one=np.array([1.0])
def prep(d,u,v,t):
    LC=[S.leg_coeffs(k) for k in range(d+1)]
    B,G=blocks(d,u,v,t,LC); out=[]
    for k in range(d+1):
        n=d-k+1; Bk=B[k].reshape(len(u),n,n); Bk=(Bk+Bk.transpose(0,2,1))/2*T3
        C0=N*S.Smat(k,d,one,one,one,LC)[0]
        H=np.einsum('pij,pkj->ik',Bk,Bk)/len(u)+C0@C0.T
        w,V=np.linalg.eigh(H); keep=w>1e-10*w.max(); V=V[:,keep]; w=w[keep]; Dm=V/w**0.25
        Dm=Dm/np.abs(Dm).max()
        Bs=np.einsum('ai,pij,jb->pab',Dm.T,Bk,Dm); Cs=Dm.T@C0@Dm
        s=max(np.abs(Bs).max(),np.abs(Cs).max()); out.append((Bs/s,Cs/s))
    return out,G
def solve(d,u,v,t,eps):
    blk,G=prep(d,u,v,t)
    cost=T3*((omega2(u)+omega2(v)+omega2(t))/(N-2)-w3(u,v,t))
    p=cp.Variable(len(u),nonneg=True)
    cons=[cp.sum(p)==1]+[(N+T3*G[:,k]@p)/N>=eps for k in range(1,d+1)]
    for Bs,Cs in blk:
        n=Cs.shape[0]
        M=cp.reshape(Bs.reshape(len(u),-1).T@p,(n,n),order='C')+Cs
        cons.append((M+M.T)/2-eps*np.eye(n)>>0)
    pr=cp.Problem(cp.Minimize(cost@p),cons); t0=time.time()
    
    try: pr.solve(solver='CLARABEL',max_iter=500)
    except Exception: pr.solve(solver='SCS',eps=1e-7,max_iters=40000,acceleration_lookback=20)
    pv=np.maximum(p.value,0); pv/=pv.sum()
    me=min(np.linalg.eigvalsh(np.tensordot(pv,Bs,1)+Cs).min() for Bs,Cs in blk)
    ml=min((N+T3*G[:,k]@pv)/N for k in range(1,d+1))
    return cost@pv, me, ml, pr.status, time.time()-t0, pv
if __name__=="__main__":
    c=d4_triples(); P=np.array(list(c.keys()))
    u,v,t=S.sample_grid(int(ARGV[0]),int(ARGV[1]))
    u,v,t=np.r_[u,P[:,0]],np.r_[v,P[:,1]],np.r_[t,P[:,2]]
    eps=float(ARGV[2])
    for d in map(int,ARGV[3:]):
        try:
            val,me,ml,st,el,pv=solve(d,u,v,t,eps)
            print(f"d={d} eps={eps}: objective {val:.6f}  tau-obj {tau-val:.5f}  recheck mineig {me:.2e} minlin {ml:.2e} [{st}, {len(u)} triples, {el:.0f}s]",flush=True)
            np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)),'continuation_out',f'm24_primal_d{d}.npy'),np.c_[u,v,t,pv])
        except Exception as ex: print(f"d={d}: failed {type(ex).__name__}",flush=True)
