#!/usr/bin/env python3
"""
gen_a18_lean.py: writes D4A18.lean, the Lean check of the two exact
statements about the configuration A_18 of sec:broad-sample of the paper.

  1. prop:a18-stabiliser: among the 384 signed coordinate permutations, which
     all map the D4 roots to themselves, exactly 48 fix A_18 as a set, and
     they are the ones that fix the coordinate x_1 together with its sign.
  2. sec:broad-sample: the configurations grown from roots 5 and 12 are the
     images of A_18 under two explicit orthogonal maps that preserve the
     roots (hessian_multidir/a18_other_starts.py finds them).

Everything is integer arithmetic on the 24 integer roots +-e_i +-e_j,
indexed as in hessian_multidir/multidir_chain_hessian_extended.py, and is
settled by `decide +kernel`.

Run: python gen_a18_lean.py   (writes D4A18.lean beside this script)
"""
import itertools
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "hessian_multidir"))
import a18_other_starts as a18mod  # noqa: E402

ROOTS = a18mod.ROOTS
A18 = a18mod.grow(0)
FROM5 = a18mod.grow(5)
FROM12 = a18mod.grow(12)

# Two maps found by a18_other_starts.py, written as 2M (integer entries).
M5 = [[2, 0, 0, 0], [0, 0, 2, 0], [0, -2, 0, 0], [0, 0, 0, 2]]
M12 = [[-1, -1, -1, -1], [1, 1, -1, -1], [-1, 1, -1, 1], [1, -1, -1, 1]]


def image2(M2, S):
    out = []
    for k in S:
        w = [sum(M2[i][j] * ROOTS[k][j] for j in range(4)) for i in range(4)]
        assert all(x % 2 == 0 for x in w)
        out.append(ROOTS.index(tuple(x // 2 for x in w)))
    return sorted(out)


assert image2(M5, A18) == FROM5 and image2(M12, A18) == FROM12


def lst(xs):
    return "[" + ", ".join(str(x) for x in xs) + "]"


def main():
    perms = list(itertools.permutations(range(4)))
    signs = list(itertools.product((1, -1), repeat=4))
    L = []
    L.append("""/-
D4A18.lean: the exact statements about the dense configuration A_18 of
sec:broad-sample of the paper.

  * stab_count: of the 384 signed coordinate permutations (24 permutations
    times 16 sign vectors, all distinct), each maps the 24 roots +-e_i +-e_j
    onto themselves, and exactly 48 map A_18 onto itself; these 48 are the
    ones that fix the coordinate x_1 with its sign (prop:a18-stabiliser).
  * congruent_from5, congruent_from12: the orthogonal maps M5 and M12 (given
    as 2M, with 2M (2M)^T = 4I) map the roots onto the roots and A_18 onto
    the configurations grown from roots 5 and 12 (sec:broad-sample).

Written by gen_a18_lean.py.  No `sorry`, no Mathlib, no native_decide.
-/

namespace D4A18
""")
    L.append("def roots : List (List Int) := [\n  " +
             ",\n  ".join(lst(r) for r in ROOTS) + "]\n")
    L.append("def a18 : List Nat := " + lst(A18))
    L.append("def from5 : List Nat := " + lst(FROM5))
    L.append("def from12 : List Nat := " + lst(FROM12) + "\n")
    L.append("def perms : List (List Nat) := [\n  " +
             ",\n  ".join(lst(p) for p in perms) + "]\n")
    L.append("def signs : List (List Int) := [\n  " +
             ",\n  ".join(lst(s) for s in signs) + "]\n")
    L.append("""def indexIn : List (List Int) -> List Int -> Nat -> Option Nat
  | [], _, _ => none
  | r :: rs, v, i => if r == v then some i else indexIn rs v (i + 1)

/-- the index of the image of each root in the index set `S`, or `none` -/
def imageIdx (f : List Int -> List Int) (S : List Nat) : List (Option Nat) :=
  S.map fun k => indexIn roots (f (roots.getD k [])) 0

def sameSet (I : List (Option Nat)) (T : List Nat) : Bool :=
  I.length == T.length && I.all (fun o => match o with
    | some j => T.contains j
    | none => false) && T.all (fun j => I.contains (some j))

def nodup : List (Option Nat) -> Bool
  | [] => true
  | x :: xs => !(xs.contains x) && nodup xs

def spm (p : List Nat) (s : List Int) (v : List Int) : List Int :=
  (List.range 4).map fun i => s.getD i 0 * v.getD (p.getD i 0) 0

def allRoots : List Nat := List.range 24

theorem perms_ok :
    (perms.length == 24 &&
     perms.all (fun p => p.length == 4 && (List.range 4).all p.contains) &&
     signs.length == 16 &&
     signs.all (fun s => s.length == 4 && s.all (fun x => x == 1 || x == -1))) = true := by
  decide +kernel

theorem perms_distinct :
    (List.range 24).all (fun i => (List.range 24).all (fun j =>
      i == j || perms.getD i [] != perms.getD j [])) = true := by
  decide +kernel

theorem signs_distinct :
    (List.range 16).all (fun i => (List.range 16).all (fun j =>
      i == j || signs.getD i [] != signs.getD j [])) = true := by
  decide +kernel

set_option maxHeartbeats 0 in
theorem signed_perms_preserve_roots :
    perms.all (fun p => signs.all (fun s =>
      let I := imageIdx (spm p s) allRoots
      sameSet I allRoots && nodup I)) = true := by
  decide +kernel

def fixesA18 (p : List Nat) (s : List Int) : Bool :=
  let I := imageIdx (spm p s) a18
  sameSet I a18 && nodup I

set_option maxHeartbeats 0 in
theorem stab_count :
    ((perms.flatMap fun p => signs.filterMap fun s =>
        if fixesA18 p s then some (p, s) else none).length == 48 &&
     perms.all (fun p => signs.all (fun s =>
        fixesA18 p s == (p.getD 1 0 == 1 && s.getD 1 0 == 1)))) = true := by
  decide +kernel
""")
    for name, M2, tgt in (("5", M5, "from5"), ("12", M12, "from12")):
        L.append(f"""def m{name} : List (List Int) := {lst(lst(r) for r in M2)}

def apply{name} (v : List Int) : List Int :=
  (m{name}.map fun row => (List.zipWith (fun a b => a * b) row v).foldl (fun s x => s + x) 0).map
    (fun x => x / 2)

theorem congruent_from{name} :
    ((List.range 4).all (fun i => (List.range 4).all (fun j =>
       ((List.zipWith (fun a b => a * b) (m{name}.getD i []) (m{name}.getD j [])).foldl
          (fun s x => s + x) 0) == (if i == j then 4 else 0))) &&
     roots.all (fun v => (m{name}.map fun row =>
       (List.zipWith (fun a b => a * b) row v).foldl (fun s x => s + x) 0).all
         (fun x => x % 2 == 0)) &&
     sameSet (imageIdx apply{name} allRoots) allRoots &&
     nodup (imageIdx apply{name} allRoots) &&
     sameSet (imageIdx apply{name} a18) {tgt} &&
     nodup (imageIdx apply{name} a18)) = true := by
  decide +kernel
""")
    L.append("end D4A18\n")
    L.append("#print axioms D4A18.stab_count")
    L.append("#print axioms D4A18.congruent_from5")
    L.append("#print axioms D4A18.congruent_from12\n")
    with open(os.path.join(HERE, "D4A18.lean"), "w") as f:
        f.write("\n".join(L))
    print("wrote D4A18.lean")


if __name__ == "__main__":
    main()
