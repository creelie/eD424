import numpy as np
classes = [
 (-1.0,-1.0,0.0),(-1.0,0.0,-1.0),(-1.0,0.0,0.0),(-1.0,0.0,1.0),(-1.0,1.0,0.0),
 (0.0,-1.0,-1.0),(0.0,-1.0,0.0),(0.0,-1.0,1.0),(0.0,0.0,-1.0),(0.0,0.0,1.0),
 (0.0,1.0,-1.0),(0.0,1.0,0.0),(0.0,1.0,1.0),
 (1.0,-1.0,0.0),(1.0,0.0,-1.0),(1.0,0.0,0.0),(1.0,0.0,1.0),
]
print("At t=0 (e_perp=v1, pure root-aligned, known breakpoints pi/4=0.7854,")
print("pi/3=1.0472, arccos(1/3)=1.2310):")
for (a,b,c) in classes:
    g = b  # cos(0)=1,sin(0)=0
    R = np.sqrt(a*a+g*g)
    if R < 1.0-1e-9:
        continue
    phi = np.arctan2(g,a)
    delta = np.arccos(min(1.0,max(-1.0,1.0/R)))
    for th in (phi-delta, phi+delta):
        if 1e-6 < th < np.pi/2-1e-6:
            print(f"  (a,b,c)=({a:+.1f},{b:+.1f},{c:+.1f})  R={R:.4f}  theta={th:.4f}")
