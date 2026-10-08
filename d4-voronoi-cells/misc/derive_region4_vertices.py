import sympy as sp

def _extract_in_order(sol, zlist, simplify_fn=None):
    """Read a solve() result back in the supplied unknown order.

    sympy returns a dict keyed by symbol; its iteration order is not
    guaranteed to match the order the unknowns were passed in, so the
    components are looked up by key rather than taken from .values().
    Returns None when the system leaves a coordinate undetermined.
    """
    if simplify_fn is None:
        simplify_fn = sp.simplify
    if not sol:
        return None
    d = sol[0]
    out = []
    for zi in zlist:
        if zi not in d:
            return None
        out.append(simplify_fn(d[zi]))
    return sp.Matrix(out)


sqrt2 = sp.sqrt(2)
u0 = sp.Matrix([1,1,0,0])/sqrt2
v1 = sp.Matrix([1,-1,0,0])/sqrt2
w1 = sp.Matrix([0,0,1,1])/sqrt2

theta, t = sp.symbols('theta t', real=True, positive=True)
c, s = sp.cos(theta), sp.sin(theta)
ct, st = sp.cos(t), sp.sin(t)
eperp = ct*v1 + st*w1
u1 = sp.simplify(c*u0 + s*eperp)
print("u1(theta,t) =", list(u1))

def root(spec):
    # spec like "+e1-e3"
    r = sp.zeros(4,1)
    sgn = {'+':1, '-':-1}
    # parse pairs
    import re
    for m in re.finditer(r'([+-])e(\d)', spec):
        sign, idx = m.group(1), int(m.group(2))-1
        r[idx] = sgn[sign]
    return r/sqrt2

def solve_vertex(facet_labels):
    """Solve the linear system: <z,u1>=1 and <z,r_i>=1 for each listed
    fixed facet r_i (using as many as needed to pin z in R^4, i.e. 3 of
    them plus u1 for a 4-fixed+u1 generically-4-dim system; if more than
    3 fixed facets are listed, use only the first 3; the extra ones
    are automatically satisfied when the point is a vertex)."""
    z1,z2,z3,z4 = sp.symbols('z1 z2 z3 z4', real=True)
    z = sp.Matrix([z1,z2,z3,z4])
    eqs = [sp.Eq((u1.T*z)[0,0], 1)]
    for lab in facet_labels[:3]:
        r = root(lab)
        eqs.append(sp.Eq((r.T*z)[0,0], 1))
    sol = sp.solve(eqs, [z1,z2,z3,z4], dict=True)
    return sol

# 3-fixed+u1 vertices (8 total) -- derive a couple representative ones
patterns_3fixed = [
    ['+e1-e2','+e1-e3','+e1-e4'],
    ['+e1+e3','+e1+e4','+e3+e4'],
    ['+e2+e3','+e2+e4','+e3+e4'],
]
print()
for pat in patterns_3fixed:
    sol = solve_vertex(pat)
    print(f"vertex on {pat} + u1:")
    if sol:
        for k,v in sol[0].items():
            print(f"    {k} = {sp.simplify(v)}")
    else:
        print("    NO SOLUTION (system inconsistent or underdetermined)")
    print()

print("="*70)
print("ALL 12 on-cap vertices for region 4 (generic small-theta region):")
all_patterns = [
    ['+e1-e2','+e1-e3','+e1-e4'],
    ['+e1+e4','+e1-e2','+e1-e3'],
    ['+e1-e3','+e1-e4','+e2-e3','+e2-e4'],
    ['+e1+e3','+e1-e2','+e1-e4'],
    ['+e1+e3','+e1+e4','+e1-e2'],
    ['+e1+e3','+e1-e4','+e2+e3','+e2-e4'],
    ['+e1+e3','+e2+e3','+e3+e4'],
    ['+e1+e3','+e1+e4','+e3+e4'],
    ['+e1+e4','+e2+e4','+e3+e4'],
    ['+e2+e3','+e2+e4','+e3+e4'],
    ['+e1+e4','+e1-e3','+e2+e4','+e2-e3'],
    ['+e2+e3','+e2+e4','+e2-e3','+e2-e4'],
]
results = {}
for pat in all_patterns:
    sol = solve_vertex(pat)
    key = tuple(pat)
    if sol:
        _ordered = _extract_in_order(
            sol, list(sp.symbols('z1 z2 z3 z4', real=True)))
        results[key] = None if _ordered is None else list(_ordered)
    else:
        results[key] = None
    print(f"{pat}: {results[key]}")
