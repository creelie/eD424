import sympy as sp
sqrt2 = sp.sqrt(2)
u0 = sp.Matrix([1,1,0,0])/sqrt2
Zstar = 2*u0
roots8 = []
for i in (0,1):
    for j in (2,3):
        for sj in (1,-1):
            r = sp.zeros(4,1); r[i]=1; r[j]=sj
            roots8.append((i,j,sj, r/sqrt2))
all_eq_1 = all(sp.simplify((Zstar.T*r)[0,0])==1 for i,j,sj,r in roots8)
print(f"<Z*, r> = 1 exactly for all 8 roots +e1+-e3, +e1+-e4, +e2+-e3, +e2+-e4: {all_eq_1}")
