import numpy as np

sqrt2 = np.sqrt(2)
u0 = np.array([1,1,0,0])/sqrt2
v1 = np.array([1,-1,0,0])/sqrt2
w1 = np.array([0,0,1,1])/sqrt2

# The 23 FIXED vertices of the D4 Voronoi cell (24-cell) relevant per
# Lemma lem:c2true Step 2: 7 axis vertices +-sqrt2*e_i (excluding +sqrt2*e1,
# which is the apex P) and 16 sign vertices (+-1,+-1,+-1,+-1)/sqrt2.
verts = []
vlabels = []
axis_signs = [(-1,1),(1,2),(-1,2),(1,3),(-1,3),(1,4),(-1,4)]  # excludes (+1,1)
for s,i in axis_signs:
    z = np.zeros(4); z[i-1] = s*sqrt2
    verts.append(z); vlabels.append(f"{'+' if s>0 else '-'}sqrt2*e{i}")
import itertools
for signs in itertools.product([1,-1], repeat=4):
    z = np.array(signs)/sqrt2
    verts.append(z)
    vlabels.append(str(signs))

print(f"Total fixed candidate vertices: {len(verts)}")
print()

classes = {}
for z, lab in zip(verts, vlabels):
    a = round(np.dot(z,u0),6)
    b = round(np.dot(z,v1),6)
    c = round(np.dot(z,w1),6)
    classes.setdefault((a,b,c), []).append(lab)

print(f"{'(a,b,c)':<20} {'#':<4} labels")
for key, labs in sorted(classes.items()):
    print(f"{str(key):<20} {len(labs):<4} {labs}")

print()
print("Now checking which classes give a real breakpoint curve in")
print("theta in (0,pi/2), t in [0,pi/2] via R(t)=sqrt(a^2+g(t)^2)>=1,")
print("g(t)=b*cos(t)+c*sin(t):")
ts = np.linspace(0.001, np.pi/2-0.001, 25)
for (a,b,c), labs in sorted(classes.items()):
    if a==0 and b==0 and c==0:
        continue
    found_any = False
    thetas_found = []
    for t in ts:
        g = b*np.cos(t)+c*np.sin(t)
        R = np.sqrt(a*a+g*g)
        if R < 1.0 - 1e-9:
            continue
        phi = np.arctan2(g,a)
        cosval = min(1.0,max(-1.0, 1.0/R))
        delta = np.arccos(cosval)
        for th in (phi-delta, phi+delta):
            if 1e-6 < th < np.pi/2-1e-6:
                found_any = True
                thetas_found.append((round(t,3), round(th,4)))
    if found_any:
        print(f"  ({a:+.2f},{b:+.2f},{c:+.2f}) [{labs[0]} type, x{len(labs)}]: "
              f"curve exists, e.g. {thetas_found[:3]} ... {thetas_found[-2:]}")
