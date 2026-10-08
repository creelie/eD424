import sympy as sp

D_ref = sp.Integer(192)
labels = ["m12","m13","n13","n14","m23","m24"]
raw = {
    "m12": sp.Matrix([1,1,0,0]), "m13": sp.Matrix([1,0,1,0]),
    "n13": sp.Matrix([1,0,-1,0]), "n14": sp.Matrix([1,0,0,-1]),
    "m23": sp.Matrix([0,1,1,0]), "m24": sp.Matrix([0,1,0,1]),
}
sqrt2 = sp.sqrt(2)
u = {lbl: raw[lbl]/sqrt2 for lbl in labels}
n = 6
H = sp.zeros(n,n)
for i,li in enumerate(labels):
    for j,lj in enumerate(labels):
        ip = sp.nsimplify(sp.simplify((u[li].T*u[lj])[0,0]))
        if i==j: H[i,j] = D_ref/3
        elif ip == sp.Rational(1,2): H[i,j] = D_ref/12
        elif ip == sp.Rational(-1,2): H[i,j] = -D_ref/12
        elif ip == 0: H[i,j] = 0

u0 = u["m12"]
v1 = sp.Matrix([1,-1,0,0])/sqrt2   # the e_perp actually used in Lemma lem:c2true Step 1

b_vec = sp.zeros(n,1)
for i,li in enumerate(labels):
    b_vec[i,0] = sp.simplify((u[li].T*v1)[0,0])/2

c2_raw = sp.expand((b_vec.T*H*b_vec)[0,0])
print(f"b(v1) = {[b_vec[i,0] for i in range(n)]}")
print(f"b^T H_Omega b at e_perp=v1 (the direction of Lemma lem:c2true): {c2_raw}")
print(f"Known exact value (Lemma lem:c2true, independent derivation): c2 = 1/3")
print(f"Ratio (b^T H b) / (1/3) = {sp.nsimplify(c2_raw / sp.Rational(1,3))}")
print()
print("Candidate normalisations tried against the target 1/3:")
for name, val in [("D_ref=192 (Definition def:hessian's own constant)", D_ref),
                   ("D_Omega(ref)=3/2 (Sec 'Normalisation of D_Omega', C001)", sp.Rational(3,2))]:
    print(f"  c2_raw / {name} = {sp.nsimplify(c2_raw/val)}  (target 1/3)")
