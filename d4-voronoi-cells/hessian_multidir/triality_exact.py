from fractions import Fraction as F
from itertools import combinations

g = [[F(1,2), F(1,2), F(1,2), F(1,2)],
     [F(1,2), F(1,2), F(-1,2),F(-1,2)],
     [F(1,2), F(-1,2),F(1,2), F(-1,2)],
     [F(1,2), F(-1,2),F(-1,2),F(1,2)]]

def matvec(M, v):
    return [sum(M[i][j]*v[j] for j in range(4)) for i in range(4)]

def matmul(A, B):
    n = len(A)
    return [[sum(A[i][k]*B[k][j] for k in range(n)) for j in range(n)] for i in range(n)]

def transpose(A):
    n = len(A)
    return [[A[j][i] for j in range(n)] for i in range(n)]

# exact orthogonality check
gT = transpose(g)
prod = matmul(g, gT)
I = [[F(1) if i==j else F(0) for j in range(4)] for i in range(4)]
print("g g^T == I exactly:", prod == I)

# roots (unnormalized, entries in {0,+-1}) -- work with these since g has rational entries
roots = []
for i,j in combinations(range(4),2):
    for si in (1,-1):
        for sj in (1,-1):
            v = [F(0)]*4
            v[i]=F(si); v[j]=F(sj)
            roots.append(tuple(v))
roots_set = set(roots)

u0 = [F(1),F(1),F(0),F(0)]   # unnormalized e1+e2
v1 = [F(1),F(-1),F(0),F(0)]  # unnormalized e1-e2
w1 = [F(0),F(0),F(1),F(1)]   # unnormalized e3+e4

gu0 = tuple(matvec(g,u0))
gv1 = tuple(matvec(g,v1))
print("g(u0) == u0 exactly:", gu0 == tuple(u0))
print("g(v1) == w1 exactly:", gv1 == tuple(w1))

img = set(tuple(matvec(g, list(r))) for r in roots)
print("g maps 24-root set to itself exactly:", img == roots_set)

others = roots_set - {tuple(u0)}
img_others = set(tuple(matvec(g, list(r))) for r in others)
print("g maps {other 23 roots} to itself exactly:", img_others == others)

print("\ndet check via cofactor-free method (product of eigen-signature not needed; use g g^T=I plus explicit 4x4 det formula):")
def det4(M):
    # Laplace expansion, exact
    import itertools
    n=4
    total = F(0)
    for perm in itertools.permutations(range(n)):
        sign = 1
        p = list(perm)
        # compute sign of permutation
        visited=[False]*n
        for i in range(n):
            if visited[i]: continue
            j=i; clen=0
            while not visited[j]:
                visited[j]=True; j=p[j]; clen+=1
            if clen%2==0: sign*=-1
        prod = F(1)
        for i in range(n):
            prod *= M[i][p[i]]
        total += sign*prod
    return total
print("det(g) =", det4(g))
