import numpy as np
from itertools import product, combinations, permutations

def mat_key(g, tol=6):
    return tuple(np.round(g, tol).flatten())

# generators: all coordinate permutations, all sign flips, and H
gens = []
for perm in permutations(range(4)):
    P = np.eye(4)[list(perm)]
    gens.append(P)
for s in product([1,-1], repeat=4):
    gens.append(np.diag(s))
H = np.array([[1,1,1,1],
              [1,1,-1,-1],
              [1,-1,1,-1],
              [1,-1,-1,1]], dtype=float) / 2.0
gens.append(H)

# generate group closure (bounded, should be finite <=1152 or so)
group = {mat_key(np.eye(4)): np.eye(4)}
frontier = [np.eye(4)]
while frontier:
    new_frontier = []
    for g in frontier:
        for gen in gens:
            h = g @ gen
            k = mat_key(h)
            if k not in group:
                group[k] = h
                new_frontier.append(h)
    frontier = new_frontier
    if len(group) > 5000:
        print("group too big, aborting")
        break

print("Group order:", len(group))

u0 = np.array([1,1,0,0])/np.sqrt(2)
v1 = np.array([1,-1,0,0])/np.sqrt(2)
w1 = np.array([0,0,1,1])/np.sqrt(2)
v2 = np.array([0,0,1,-1])/np.sqrt(2)

def close(a,b,tol=1e-6):
    return np.allclose(a,b,atol=tol)

stab = [g for g in group.values() if close(g @ u0, u0)]
print("Stabilizer of u0 order:", len(stab))

orbit_v1 = set()
for g in stab:
    orbit_v1.add(tuple(np.round(g @ v1, 6)))
print("Orbit of v1 under stabilizer, size:", len(orbit_v1))
print("Does orbit contain w1?", tuple(np.round(w1,6)) in orbit_v1)
print("Does orbit contain v2?", tuple(np.round(v2,6)) in orbit_v1)
print("Does orbit contain -w1?", tuple(np.round(-w1,6)) in orbit_v1)
print("Does orbit contain -v2?", tuple(np.round(-v2,6)) in orbit_v1)
for x in sorted(orbit_v1):
    print("  ", np.round(x,3))

# find the specific g in stab mapping v1 -> w1 exactly (numerically first)
target = None
for g in stab:
    if close(g @ v1, w1):
        target = g
        break
print("\nFound g with g(u0)=u0, g(v1)=w1:")
print(np.round(target, 4))
print("g @ u0 =", np.round(target @ u0, 6), " (should equal u0)")
print("g @ v1 =", np.round(target @ v1, 6), " (should equal w1)")
print("g orthogonal check, g@g.T:\n", np.round(target @ target.T, 6))
print("det(g) =", np.round(np.linalg.det(target), 6))

# check g maps full root set to itself (should, since g in group), and check the OTHER 23 roots specifically
roots = []
for i,j in combinations(range(4),2):
    for si in (1,-1):
        for sj in (1,-1):
            v = np.zeros(4)
            v[i] = si; v[j] = sj
            roots.append(v)
roots = np.array(roots)
img = roots @ target.T
# check set equality
base_key = set(tuple(np.round(r,6)) for r in roots)
img_key = set(tuple(np.round(r,6)) for r in img)
print("g maps 24-root set to itself:", base_key == img_key)

# does g fix u0*sqrt2 = (1,1,0,0) individually and only permute the other 23?
u0_full = np.array([1,1,0,0])
others = [r for r in roots if not np.allclose(r, u0_full)]
print("number of 'other' roots (excluding u0):", len(others))
img_others = others @ target.T if False else np.array(others) @ target.T
others_key = set(tuple(np.round(r,6)) for r in others)
img_others_key = set(tuple(np.round(r,6)) for r in img_others)
print("g maps {other 23 roots} to itself as a set:", others_key == img_others_key)
