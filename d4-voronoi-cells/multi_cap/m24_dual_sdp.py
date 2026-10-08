"""
m24_dual_sdp.py -- the dual side of the three-point relaxation of the
24-contact case.

Solves the dual semidefinite program on a sampled set of admissible triples,
then hunts over a much finer grid and a random sample for the worst violation
of the constraint off that set, corrects the value by it, and adds the worst
offenders to the sample for the next round.  The corrected value is a genuine
bound; the ratio it prints is that bound against tau = 8 - vol B(sqrt2) + 24
cap, the quantity the 24-contact case needs.

  python3 m24_dual_sdp.py [degree {2|3} rounds]          default: 6 3 1

The environment variable BOX sets the box constraint on the dual variables.
"""
import os
BOX=float(os.environ.get('BOX','1e3'))
import sys, time, numpy as np, cvxpy as cp
ARGV = sys.argv[1:] or ['6', '3', '1']
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import three_point_sdp as S
from scipy.interpolate import RegularGridInterpolator
from m24_volume_functional import omega2, CAP
S.N=24; N=24; PI=np.pi
T=np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),'continuation_out','m24_w3table.npy')); g=np.arange(21)/40
RGI=RegularGridInterpolator((g,g,g),T,method='cubic')
def w3(u,v,t):
    P=np.stack([u,v,t],-1); inside=(P>=0).all(-1)
    out=np.zeros(len(u)); 
    if inside.any(): out[inside]=np.maximum(RGI(np.clip(P[inside],0,.5)),0)
    return out
tau=8-2*PI**2+24*CAP
SC=100.0
def rows(d,u,v,t,LC):
    rhs=SC*(omega2(u)+omega2(v)+omega2(t)-(N-2)*w3(u,v,t))
    Af=np.stack([S.gegen(k,u)+S.gegen(k,v)+S.gegen(k,t) for k in range(d+1)],1)
    one=np.ones_like(u); AF=[]
    for k in range(d+1):
        M=S.Smat(k,d,u,v,t,LC)+(S.Smat(k,d,one,u,u,LC)+S.Smat(k,d,one,v,v,LC)+S.Smat(k,d,one,t,t,LC))/(N-2)
        AF.append(M.reshape(len(u),-1))
    return rhs,Af,AF
def solve(d,u,v,t,LC,three=True):
    rhs,Af,AF=rows(d,u,v,t,LC)
    f=cp.Variable(d+1); Fs=[cp.Variable((d-k+1,d-k+1),symmetric=True) for k in range(d+1)]
    e=Af@f
    if three:
        for k in range(d+1): e=e+AF[k]@cp.vec(Fs[k],order='C')
    one=np.array([1.0])
    F111=sum(cp.sum(cp.multiply(S.Smat(k,d,one,one,one,LC)[0],Fs[k])) for k in range(d+1)) if three else 0
    bound=N*(N*f[0]-cp.sum(f))/2-N*F111/(6*(N-2))
    cons=[e<=rhs,f[1:]>=0,cp.abs(f)<=BOX]+[F>>0 for F in Fs]+[cp.trace(F)<=BOX for F in Fs]
    pr=cp.Problem(cp.Maximize(bound),cons)
    try: pr.solve(solver='CLARABEL',max_iter=400)
    except Exception as ex:
        print('clarabel failed',flush=True); return float('nan'),None,None,None,None,None
    print('status',pr.status,'maxtr',max(np.trace(F.value) for F in Fs) if three else 0,flush=True)
    return pr.value/SC, f.value, ([F.value for F in Fs] if three else None), rhs, Af, AF
def viol(d,fv,Fv,u,v,t,LC):
    w=-np.inf; keep=[]
    for lo in range(0,len(u),50000):
        sl=slice(lo,lo+50000)
        rhs,Af,AF=rows(d,u[sl],v[sl],t[sl],LC)
        val=Af@fv
        if Fv is not None:
            for k in range(d+1): val=val+AF[k]@Fv[k].reshape(-1)
        x=(val-rhs)/SC; w=max(w,x.max()); i=np.argsort(x)[-1500:]
        keep.append((x[i],u[sl][i],v[sl][i],t[sl][i]))
    X=np.concatenate([k[0] for k in keep]); o=np.argsort(X)[-1500:]
    return w,[np.concatenate([k[j] for k in keep])[o] for j in (1,2,3)]
if __name__=="__main__":
    d=int(ARGV[0]); three=ARGV[1]=='3'; rounds=int(ARGV[2])
    LC=[S.leg_coeffs(k) for k in range(d+1)]; rng=np.random.default_rng(3)
    print(f"tau = 8 - vol B(sqrt2) + 24 cap = {tau:.8f}; D4 value 96(w2-w3) = {96*(omega2(.5)-w3(np.array([.5]),np.array([.5]),np.array([.5]))[0]):.8f}",flush=True)
    u,v,t=S.sample_grid(30,14)
    fu,fv_,ft=S.sample_grid(60,25); ru,rv,rt=S.sample_random(40000,rng)
    cu,cv,ct=np.r_[fu,ru],np.r_[fv_,rv],np.r_[ft,rt]
    for r in range(rounds):
        t0=time.time(); val,fv,Fv,*_=solve(d,u,v,t,LC,three)
        if fv is None: print("solver failed",flush=True); break
        w,(U,V,TT)=viol(d,fv,Fv,cu,cv,ct,LC)
        corr=max(w,0)*(N*(N-1)*(N-2)/6)/(N-2)
        print(f"d={d} {'3pt' if three else '2pt'} round {r+1}: sampled {val:.6f}, maxviol {w:.2e}, corrected {val-corr:.6f}, ratio {(val-corr)/tau:.5f} ({len(u)} triples, {time.time()-t0:.0f}s)",flush=True)
        u,v,t=np.r_[u,U],np.r_[v,V],np.r_[t,TT]
    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)),'continuation_out',f'm24_dual_d{d}_{ARGV[1]}pt.npz'),f=fv,**({f"F{k}":Fv[k] for k in range(d+1)} if Fv else {}))
