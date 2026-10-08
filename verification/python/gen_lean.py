#!/usr/bin/env python3
"""
gen_lean.py -- write ../lean/R4Check.lean: the integer certificates of
../data/vertices.txt and ../data/edges.txt as Lean strings, together with the
Lean checks that read them.  The data part is regenerated; the checks are
fixed text below.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
OUT = os.path.join(HERE, '..', 'lean', 'R4Check.lean')

HEAD = r'''/-
  R4Check.lean -- Lean 4 (core only, no Mathlib) checks for the paper
  "Densest packings of two translates of a lattice in four dimensions".

  Proved by kernel reasoning:
    * `close_prod_nonneg`, `unbounded_form_nonneg`: for integers w, a and
      l in {-1, 0, 1}, (w - a l)(w - (a + 1) l) >= 0.  This is why every
      direction X_{u,a} of an unbounded edge satisfies X[k] >= 0 on M.

  Proved by evaluation (`native_decide`) on the integer certificates of
  verification/data, embedded below as strings:
    * `vertex_invariants`: the ten vertex representatives have the stated
      det Q and det J;
    * `vertex_minimal_vectors`: every listed minimal vector k of a
      representative has J[k] = 4, and their number is as stated;
    * `bounded_edges_ok`: for each of the 1112 bounded edges (J_i, X, t*),
      the matrix T is in Gamma (last row (0,0,0,0,+-1), det U = +-1) and
      T^T J_j T = J_i + t* X;
    * `unbounded_edges_ok`: for each of the 25 unbounded edges,
      X = p q^T + q p^T with p = (u, -a), q = (u, -(a + 1));
    * `unbounded_orbits_ok`: the 25 unbounded edges fall into the seven
      orbits of Table 5.3: for each, T in Gamma fixes J_i and maps the
      representative direction to it.
-/

def parseInts (s : String) : List Int :=
  (s.split (fun c => c == ' ' || c == '\n')).filterMap String.toInt?

abbrev Mat := List (List Int)

def rows5 (l : List Int) : Mat :=
  [l.take 5, (l.drop 5).take 5, (l.drop 10).take 5, (l.drop 15).take 5, (l.drop 20).take 5]

def entry (A : Mat) (i j : Nat) : Int := (A.getD i []).getD j 0

def matOf (n : Nat) (f : Nat → Nat → Int) : Mat :=
  (List.range n).map fun i => (List.range n).map fun j => f i j

def mul (A B : Mat) (n : Nat) : Mat :=
  matOf n fun i j => (List.range n).foldl (fun s k => s + entry A i k * entry B k j) 0

def transpose (A : Mat) (n : Nat) : Mat := matOf n fun i j => entry A j i

def minor (A : Mat) (j : Nat) : Mat :=
  (A.drop 1).map fun row => (row.enum.filter (fun p => p.1 != j)).map Prod.snd

def detN : Nat → Mat → Int
  | 0, _ => 1
  | n + 1, A =>
    match A with
    | [] => 1
    | row :: _ =>
      (row.enum.map fun (p : Nat × Int) =>
        (if p.1 % 2 == 0 then 1 else -1) * p.2 * detN n (minor A p.1)).foldl (· + ·) 0

/-- Laplace expansion along the first row. -/
def det (A : Mat) : Int := detN A.length A

def qblock (A : Mat) : Mat := (A.take 4).map (·.take 4)

def qf (A : Mat) (k : List Int) : Int :=
  (List.range 5).foldl (fun s i => (List.range 5).foldl (fun t j => t + entry A i j * k.getD i 0 * k.getD j 0) s) 0

def lineRecords (s : String) : List (List Int) :=
  ((s.split (· == '\n')).map parseInts).filter (· ≠ [])

-- vertex record: id, J (25), m, then m vectors of length 5
def vJ (r : List Int) : Mat := rows5 ((r.drop 1).take 25)
def vVecs (r : List Int) : List (List Int) :=
  let m := (r.getD 26 0).toNat
  (List.range m).map fun a => ((r.drop (27 + 5 * a)).take 5)

'''

TAIL = r'''
def vertices : List (List Int) := lineRecords verticesData
def edges : List (List Int) := lineRecords edgesData

def vertexJ (i : Int) : Mat := vJ (vertices.getD i.toNat [])

def invariants : List (Int × Int) := vertices.map fun r => (det (qblock (vJ r)), det (vJ r))

theorem vertex_invariants :
    invariants = [(256, 0), (128, 128), (192, 128), (80, 128), (64, 128),
                  (108, 162), (128, 192), (144, 192), (81, 162), (80, 192)] := by
  native_decide

def minimalOk (r : List Int) : Bool :=
  let J := vJ r
  let ks := vVecs r
  ks.length == (r.getD 26 0).toNat && ks.all (fun k => qf J k == 4)

theorem vertex_minimal_vectors : vertices.length = 10 ∧ vertices.all minimalOk = true := by
  native_decide

-- edge record: id, i, X (25), t, j, then T (25) if j >= 0, or u (4), a if j = -1
def edgeX (r : List Int) : Mat := rows5 ((r.drop 2).take 25)

def boundedOk (r : List Int) : Bool :=
  let i := r.getD 1 0
  let t := r.getD 27 0
  let j := r.getD 28 0
  let T := rows5 ((r.drop 29).take 25)
  let J := vertexJ i
  let X := edgeX r
  let J2 := matOf 5 fun a b => entry J a b + t * entry X a b
  let U := (T.take 4).map (·.take 4)
  let lastRow := T.getD 4 []
  let gammaOk := lastRow.take 4 == [0, 0, 0, 0] && (lastRow.getD 4 0 == 1 || lastRow.getD 4 0 == -1)
    && (det U == 1 || det U == -1)
  gammaOk && mul (mul (transpose T 5) (vertexJ j) 5) T 5 == J2

def unboundedOk (r : List Int) : Bool :=
  let u := (r.drop 29).take 4
  let a := r.getD 33 0
  let p := u ++ [-a]
  let q := u ++ [-(a + 1)]
  let W := matOf 5 fun x y => p.getD x 0 * q.getD y 0 + q.getD x 0 * p.getD y 0
  W == edgeX r

def boundedEdges : List (List Int) := edges.filter fun r => r.getD 28 0 ≥ 0
def unboundedEdges : List (List Int) := edges.filter fun r => r.getD 28 0 < 0

def orbits : List (List Int) := lineRecords orbitsData

def edgeById (e : Int) : List Int := edges.getD e.toNat []

def gammaOk (T : Mat) : Bool :=
  let U := (T.take 4).map (·.take 4)
  let lastRow := T.getD 4 []
  lastRow.take 4 == [0, 0, 0, 0] && (lastRow.getD 4 0 == 1 || lastRow.getD 4 0 == -1)
    && (det U == 1 || det U == -1)

def orbitOk (o : List Int) : Bool :=
  let e := o.getD 0 0
  let i := o.getD 1 0
  let r := o.getD 2 0
  let T := rows5 ((o.drop 3).take 25)
  let Te := mul (transpose T 5) (vertexJ i) 5
  (edgeById e).getD 28 0 == -1 && (edgeById r).getD 28 0 == -1
    && (edgeById e).getD 1 0 == i && (edgeById r).getD 1 0 == i
    && gammaOk T && mul Te T 5 == vertexJ i
    && mul (mul (transpose T 5) (edgeX (edgeById r)) 5) T 5 == edgeX (edgeById e)

def orbitReps : List Int := (orbits.map fun o => o.getD 2 0).eraseDups

theorem unbounded_orbits_ok :
    orbits.length = 25 ∧ orbits.all orbitOk = true ∧ orbitReps = [56, 218, 228, 257, 604, 713, 1077] := by
  native_decide

theorem edge_counts : edges.length = 1137 ∧ boundedEdges.length = 1112 ∧ unboundedEdges.length = 25 := by
  native_decide

theorem bounded_edges_ok : boundedEdges.all boundedOk = true := by
  native_decide

theorem unbounded_edges_ok : unboundedEdges.all unboundedOk = true := by
  native_decide

/-- Two integers that differ by at most one have a nonnegative product. -/
theorem close_prod_nonneg (x y : Int) (h : y = x - 1 ∨ y = x ∨ y = x + 1) : 0 ≤ x * y := by
  cases Int.le_total 0 x with
  | inl hx =>
    cases Int.le_total 0 y with
    | inl hy => exact Int.mul_nonneg hx hy
    | inr hy =>
      have : x = 0 ∨ y = 0 := by omega
      cases this with
      | inl h0 => subst h0; simp
      | inr h0 => subst h0; simp
  | inr hx =>
    cases Int.le_total 0 y with
    | inl hy =>
      have : x = 0 ∨ y = 0 := by omega
      cases this with
      | inl h0 => subst h0; simp
      | inr h0 => subst h0; simp
    | inr hy => exact Int.mul_nonneg_of_nonpos_of_nonpos hx hy

/-- The direction of an unbounded edge is nonnegative on M:
    (w - a l)(w - (a + 1) l) >= 0 for integers w, a and l in {-1, 0, 1}. -/
theorem unbounded_form_nonneg (w a l : Int) (hl : l = -1 ∨ l = 0 ∨ l = 1) :
    0 ≤ (w - a * l) * (w - (a + 1) * l) := by
  apply close_prod_nonneg
  rcases hl with h | h | h <;> subst h <;> omega
'''


def esc(s):
    return s.replace('\\', '\\\\').replace('"', '\\"')


def main():
    v = open(os.path.join(DATA, 'vertices.txt')).read()
    e = open(os.path.join(DATA, 'edges.txt')).read()
    o = open(os.path.join(DATA, 'unbounded_orbits.txt')).read()
    with open(OUT, 'w') as fh:
        fh.write(HEAD)
        fh.write('def verticesData : String := "%s"\n\n' % esc(v))
        fh.write('def edgesData : String := "%s"\n\n' % esc(e))
        fh.write('def orbitsData : String := "%s"\n' % esc(o))
        fh.write(TAIL)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
