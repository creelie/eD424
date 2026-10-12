"""Write the certificate of the enumeration as a plain text file read by lean/generate/make_lean.py.

Format (one record per line, integers and rationals p/q separated by spaces):
  VERTICES n
  VERTEX i
  J  j00 j01 ... j44                 (25 rationals, row major)
  MIN m        followed by m lines k1 ... k5   (k in M*, J[k] = 4; one of +-k)
  GENS g       followed by g lines of 25 integers (T row major, T^T J T = J)
  RAYS n       followed by n lines of 15 integers (y_ij, i <= j, row major; Y[k] = sum Y_ij k_i k_j)
  REPS r       followed by r lines
                 U idx size                     (the ray is unbounded: J + tY in R for all t >= 0)
                 B idx size tstar to t00 ... t44  (bounded: J + t*Y = T^T J_to T)
  END
  EQUALITY e   followed by e lines  P vertex idx t   (idx = -1 for the vertex itself)
"""
import sys, json, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'compute'))
from fractions import Fraction as F
from forms import min_vectors
from rays import extreme_rays

def main(vpath, epath, out):
    data = json.load(open(vpath))
    eq = json.load(open(epath))
    lines = [f"VERTICES {len(data)}"]
    for V in data:
        J = [[F(x) for x in row] for row in V['J']]
        mv = min_vectors(J)
        R = extreme_rays(mv)
        pos = {tuple(r): i for i, r in enumerate(R)}
        lines.append(f"VERTEX {V['index']}")
        lines.append("J " + " ".join(str(x) for row in J for x in row))
        lines.append(f"MIN {len(mv)}")
        lines += [" ".join(map(str, k)) for k in mv]
        lines.append(f"GENS {len(V['gens'])}")
        lines += [" ".join(str(x) for row in T for x in row) for T in V['gens']]
        lines.append(f"RAYS {len(R)}")
        lines += [" ".join(map(str, r)) for r in R]
        lines.append(f"REPS {len(V['edges'])}")
        for e in V['edges']:
            idx = pos[tuple(e['ray'])]
            if e['tstar'] is None:
                lines.append(f"U {idx} {e['orbit']}")
            else:
                lines.append(f"B {idx} {e['orbit']} {e['tstar']} {e['to']} " +
                             " ".join(str(x) for row in e['T'] for x in row))
        lines.append("END")
    lines.append(f"EQUALITY {len(eq)}")
    for P in eq:
        if P['ray'] is None:
            lines.append(f"P {P['from']} -1 0")
        else:
            V = data[P['from']]
            J = [[F(x) for x in row] for row in V['J']]
            R = extreme_rays(min_vectors(J))
            idx = R.index(tuple(P['ray']))
            lines.append(f"P {P['from']} {idx} {P['t']}")
    open(out, 'w').write("\n".join(lines) + "\n")

if __name__ == '__main__':
    main(*sys.argv[1:4])
