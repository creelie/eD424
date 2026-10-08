#!/usr/bin/env python3
"""Full completeness check: exactly 4 breakpoint curves on the w1-v2 arc
(universal, W, B=(1,1,1,-1)/sqrt2, C=sqrt2*e2), tested at 60 t-samples."""
import numpy as np
from scipy.spatial import HalfspaceIntersection
import sympy as sp

roots = []
labels = []
for i in range(4):
    for j in range(i + 1, 4):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(4)
                v[i] = si; v[j] = sj
                roots.append(v / np.sqrt(2)); labels.append((i, si, j, sj))
roots = np.array(roots)

def find(vec):
    v = np.array(vec, dtype=float); v /= np.linalg.norm(v)
    d = roots @ v; idx = np.argmax(d); assert d[idx] > 1 - 1e-9
    return idx

idx0 = find([1,1,0,0]); u0 = roots[idx0]
w1 = roots[find([0,0,1,1])]; v2 = roots[find([0,0,1,-1])]

def e_perp_arc(t): return np.cos(t)*w1 + np.sin(t)*v2

def dirs_at(theta,t):
    e = e_perp_arc(t); u1 = np.cos(theta)*u0 + np.sin(theta)*e
    d = roots.copy(); d[idx0]=u1; return d

def active_set(theta,t):
    dirs = dirs_at(theta,t)
    hs = np.hstack([dirs, -np.ones((len(dirs),1))])
    hi = HalfspaceIntersection(hs, np.zeros(4))
    verts = hi.intersections
    on_cap = np.abs(verts@dirs[idx0]-1.0)<1e-9
    neigh=set()
    for vi in np.where(on_cap)[0]:
        vv=verts[vi]
        for j in range(len(dirs)):
            if j==idx0: continue
            if abs(np.dot(vv,dirs[j])-1.0)<1e-9: neigh.add(j)
    return frozenset(neigh)

def bisect(t,lo,hi,tol=1e-11):
    s_lo=active_set(lo,t)
    for _ in range(55):
        mid=0.5*(lo+hi)
        if active_set(mid,t)==s_lo: lo=mid
        else: hi=mid
        if hi-lo<tol: break
    return 0.5*(lo+hi)

def find_breaks(t,n_scan=500,th_lo=0.003,th_hi=np.pi/2-0.003):
    thetas=np.linspace(th_lo,th_hi,n_scan)
    prev=active_set(thetas[0],t); out=[]
    for i in range(1,len(thetas)):
        cur=active_set(thetas[i],t)
        if cur!=prev:
            out.append(bisect(t,thetas[i-1],thetas[i])); prev=cur
    return out

U_curve=lambda tt: np.pi/3
W_curve=lambda tt: 2*np.arctan(np.cos(tt))
B_curve=lambda tt: 2*np.arctan(np.sin(tt))
def C_curve(tt):
    s = np.cos(tt)+np.sin(tt)
    if s>1: return np.arcsin(1.0/s)
    return None

# ---- exact symbolic derivation of B and C ----
t_s, theta_s, u = sp.symbols('t theta u', real=True)
w1s = sp.Matrix([0,0,1,1])/sp.sqrt(2); v2s = sp.Matrix([0,0,1,-1])/sp.sqrt(2)
u0s = sp.Matrix([1,1,0,0])/sp.sqrt(2)
e_perp_s = sp.cos(t_s)*w1s + sp.sin(t_s)*v2s
u1_s = sp.cos(theta_s)*u0s + sp.sin(theta_s)*e_perp_s
Bv = sp.Matrix([1,1,1,-1])/sp.sqrt(2)
Cv = sp.Matrix([0,0,sp.sqrt(2),0])
ip_B = sp.simplify((Bv.T*u1_s)[0])
ip_C = sp.simplify((Cv.T*u1_s)[0])
print("<B,u1> =", ip_B)
print("<C,u1> =", ip_C)
assert ip_B == sp.cos(theta_s) + sp.sin(theta_s)*sp.sin(t_s)
assert sp.simplify(ip_C - sp.sin(theta_s)*(sp.cos(t_s)+sp.sin(t_s))) == 0

exprB = sp.simplify(sp.together(ip_B.subs({sp.sin(theta_s):2*u/(1+u**2), sp.cos(theta_s):(1-u**2)/(1+u**2)})-1))
solB = sp.solve(sp.Eq(exprB,0), u)
print("B-curve solve:", solB, " -> nontrivial root should be sin(t)")
assert sp.sin(t_s) in solB or sp.simplify(solB[-1]-sp.sin(t_s))==0

print("\nB-curve exact: tan(theta/2) = sin(t)  =>  theta = 2*arctan(sin t)")
print("C-curve exact: sin(theta)*(cos(t)+sin(t)) = 1")

# ---- completeness scan ----
samples = np.linspace(0.01, np.pi/4-0.01, 60)
all_ok = True
max_curves = 0
for ti in samples:
    breaks = find_breaks(ti)
    max_curves = max(max_curves, len(breaks))
    unexplained = []
    for b in breaks:
        hit = (abs(b-U_curve(ti))<2e-4 or abs(b-W_curve(ti))<2e-4 or abs(b-B_curve(ti))<2e-4
               or (C_curve(ti) is not None and abs(b-C_curve(ti))<2e-4))
        if not hit:
            unexplained.append(b)
    if unexplained:
        all_ok = False
        print(f"t={ti:.4f}: UNEXPLAINED breaks remain: {unexplained}  (all breaks: {breaks})")

print(f"\nmax breakpoints seen at any sampled t: {max_curves}")
print("ALL BREAKPOINTS EXPLAINED BY THE 4 CURVES (universal,W,B,C):", all_ok)
