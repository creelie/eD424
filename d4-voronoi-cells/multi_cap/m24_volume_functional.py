import os
import numpy as np
from itertools import combinations
from scipy.integrate import quad
PI=np.pi; S2=np.sqrt(2.0)
def I(th): return th-np.tan(th)+(np.tan(th)+np.tan(th)**3/3)/4
def omega2(u):
    u=np.asarray(u,float); g=np.arccos(np.clip(u,-1,1))
    return np.where(u>0, 2*PI*(I(PI/4)-I(np.minimum(g,PI/2)/2)), 0.0)
CAP=quad(lambda y:4*PI/3*(2-y*y)**1.5,1,S2)[0]
def J(a):
    a=np.minimum(a,S2); ph=np.arcsin(a/S2)
    return (PI/2-(ph-np.sin(4*ph)/4))
def omega3_one(u,v,t,n=260):
    G=np.array([[1,u,v],[u,1,t],[v,t,1]],float)
    ev=np.linalg.eigvalsh(G)
    if ev[0]<1e-9: G=G+(1e-9-ev[0])*np.eye(3)
    L=np.linalg.cholesky(G)            # rows = w_i in R^3
    c=L.sum(0); c/=np.linalg.norm(c)
    # quick emptiness test: region {d: <d,w_i> >= 1/sqrt2}; maximise min via sampling around c
    e1=np.cross(c,[1,0,0]); 
    if np.linalg.norm(e1)<1e-6: e1=np.cross(c,[0,1,0])
    e1/=np.linalg.norm(e1); e2=np.cross(c,e1)
    amax=PI/2
    a=(np.arange(n)+.5)/n*amax; b=(np.arange(2*n)+.5)/(2*n)*2*PI
    A,B=np.meshgrid(a,b,indexing='ij')
    D=(np.cos(A)[...,None]*c+np.sin(A)[...,None]*(np.cos(B)[...,None]*e1+np.sin(B)[...,None]*e2))
    m=(D@L.T).min(-1)
    val=np.where(m>=1/S2, J(1/np.maximum(m,1e-12)),0.0)
    return float((val*np.sin(A)).sum()*(amax/n)*(2*PI/(2*n)))
def roots():
    V=[]
    for i,j in combinations(range(4),2):
        for si in (1,-1):
            for sj in (1,-1):
                v=np.zeros(4); v[i],v[j]=si,sj; V.append(v/S2)
    return np.array(V)
if __name__=="__main__":
    W=roots(); G=W@W.T
    pairs=[(i,j) for i,j in combinations(range(24),2)]
    np2=sum(abs(G[i,j]-.5)<1e-9 for i,j in pairs)
    tri=[(i,j,k) for i,j,k in combinations(range(24),3) if min(G[i,j],G[i,k],G[j,k])>1e-9]
    from collections import Counter
    print("pairs at 60:",np2,"triples with all u>0:",Counter(tuple(sorted(np.round([G[i,j],G[i,k],G[j,k]],3))) for i,j,k in tri))
    print("check omega2 by direct 2D quad:",omega2(.5), )
    for n in (130,260,520):
        print("n",n,"omega3(1/2,1/2,1/2)=",omega3_one(.5,.5,.5,n))
    w3=omega3_one(.5,.5,.5,520)
    ntri=len(tri)
    Phi=2*PI**2-24*CAP+np2*omega2(.5)-ntri*w3
    print(f"CAP={CAP:.10f}  Phi(D4)={Phi:.8f}  (should be 8)")
