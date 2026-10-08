import sympy as sp

# C001 exact data (from paper, matches register_audit_all22.py / d4_levelB_analytic.py)
D_ref = sp.Integer(192)

# Active facet roots (unnormalized integer vectors; actual unit vectors are these /sqrt2)
labels = ["m12","m13","n13","n14","m23","m24"]
raw = {
    "m12": sp.Matrix([1,1,0,0]),
    "m13": sp.Matrix([1,0,1,0]),
    "n13": sp.Matrix([1,0,-1,0]),
    "n14": sp.Matrix([1,0,0,-1]),
    "m23": sp.Matrix([0,1,1,0]),
    "m24": sp.Matrix([0,1,0,1]),
}
sqrt2 = sp.sqrt(2)
u = {lbl: raw[lbl]/sqrt2 for lbl in labels}

# Build G0 (Gram matrix) and H_Omega directly from inner products (exact)
n = 6
G0 = sp.zeros(n,n)
H = sp.zeros(n,n)
for i,li in enumerate(labels):
    for j,lj in enumerate(labels):
        ip = sp.nsimplify(sp.simplify((u[li].T*u[lj])[0,0]))
        G0[i,j] = ip
        if i==j:
            H[i,j] = D_ref/3
        else:
            if ip == sp.Rational(1,2):
                H[i,j] = D_ref/12
            elif ip == sp.Rational(-1,2):
                H[i,j] = -D_ref/12
            elif ip == 0:
                H[i,j] = 0
            else:
                raise ValueError(f"unexpected inner product {ip} for {li},{lj}")

print("G0 =")
sp.pprint(G0)
print()
print("H_Omega =")
sp.pprint(H)
print()

u0 = u["m12"]
# orthonormal basis of the 3-space orthogonal to u0, using D4 roots (as in eperp scripts)
v1 = sp.Matrix([1,-1,0,0])/sqrt2
w1 = sp.Matrix([0,0,1,1])/sqrt2
v2 = sp.Matrix([0,0,1,-1])/sqrt2

# sanity: orthonormal, orthogonal to u0
for name,vec in [("v1",v1),("w1",w1),("v2",v2)]:
    assert sp.simplify((u0.T*vec)[0,0]) == 0, name
assert sp.simplify((v1.T*v1)[0,0]) == 1
assert sp.simplify((w1.T*w1)[0,0]) == 1
assert sp.simplify((v2.T*v2)[0,0]) == 1
assert sp.simplify((v1.T*w1)[0,0]) == 0
assert sp.simplify((v1.T*v2)[0,0]) == 0
assert sp.simplify((w1.T*v2)[0,0]) == 0
print("[OK] v1,w1,v2 orthonormal and orthogonal to u0")
print()

a,b,c = sp.symbols('a b c', real=True)
e_perp = a*v1 + b*w1 + c*v2

b_vec = sp.zeros(n,1)
for i,li in enumerate(labels):
    ip = sp.simplify((u[li].T*e_perp)[0,0])
    b_vec[i,0] = sp.together(ip/2)

print("b(e_perp) components (should have b_1 = 0 exactly since u_1=u0 perp e_perp):")
for i,li in enumerate(labels):
    print(f"  b_{i+1} ({li}) = {sp.simplify(b_vec[i,0])}")
print()

c2_expr = (b_vec.T * H * b_vec)[0,0]
c2_expr = sp.expand(c2_expr)
print("c2(e_perp) = b^T H_Omega b, expanded:")
print(c2_expr)
print()

# substitute constraint a^2+b^2+c^2=1 to simplify
c2_on_sphere = sp.simplify(c2_expr)
print("c2(e_perp), simplified (before imposing unit constraint):")
print(c2_on_sphere)
print()

# Express as (quadratic form in a,b,c), check if it's a multiple of (a^2+b^2+c^2)
poly = sp.Poly(c2_expr, a, b, c)
print("As polynomial in a,b,c:")
print(poly)
print()

target = sp.Rational(1,3)*(a**2+b**2+c**2)
diff = sp.expand(c2_expr - target)
print(f"c2_expr - (1/3)(a^2+b^2+c^2) = {diff}")
print(f"Identically zero (c2 constant = 1/3 on unit sphere)?  {diff == 0}")
