#!/usr/bin/env python3
"""
gen_cell600_lean.py -- writes cell600/D4Cell600Enum.lean and
cell600/D4Cell600Main.lean.

The vertex list and the list of the 25 inscribed 24-cells are produced by
multi_cap/cell600_exact.py in exact Z[phi] arithmetic; this script regenerates
them the same way and emits them as Lean literals, together with the theorems
that the Lean kernel checks (D4Cell600Enum.lean) and the three enumeration
theorems that the compiled search settles (D4Cell600Main.lean).  Run once; the
two output files are part of the package and are built with `lake build` in
cell600/.
"""
import sys, os, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'multi_cap'))
from cell600_exact import build_vertices, inner4, enumerate_cliques, popcount

V = build_vertices(); n = len(V)
rootlike = {(-4,0), (-2,0), (0,0), (2,0)}
cadj = [0]*n
for i in range(n):
    for j in range(i+1, n):
        if inner4(V[i], V[j]) in rootlike:
            cadj[i] |= 1 << j; cadj[j] |= 1 << i
cells = set()
for f in range(n):
    for c in enumerate_cliques(cadj, n, 24, fixed=f):
        cells.add(c)
    if len(cells) == 25 and f >= 4: break
cells = sorted(cells)
assert len(cells) == 25

def z(p): return f"({p[0]}, {p[1]})"
vlines = ",\n".join("    (" + ", ".join(z(c) for c in v) + ")" for v in V)
clines = ",\n".join("    [" + ", ".join(str(i) for i in range(n) if c >> i & 1) + "]" for c in cells)

src = r'''/-
D4Cell600Enum.lean

A machine check of the finite statement behind prop:cell600 of
"The Sphere Packing Problem in Dimension 4 and the Twenty-Four-Cell Conjecture": every set of
twenty-three vertices of the 600-cell with pairwise inner products at most
1/2 is an inscribed 24-cell with one vertex removed. This file holds the
data, the definitions, and every theorem the kernel checks by `decide`;
the three enumeration theorems are in D4Cell600Main.lean.

The vertices are the unit icosians. Coordinates are doubled and written in
Z[phi] as pairs (a, b) standing for a + b*phi, so that every coordinate is
one of 0, +-1, +-2, +-phi, +-(phi - 1) and every inner product is computed
exactly. With that scaling 4<u,v> is one of 0, +-2, +-4, +-2phi, +-(2phi-2),
and two vertices are closer than 60 degrees exactly when 4<u,v> = 2phi,
that is (0, 2) below.

Everything about the vertices and the twenty-five cells is settled by
`decide`, so the kernel checks it, and no axiom beyond propositional
extensionality enters. The enumeration,
that the independent 23-sets through a fixed vertex number exactly 115 and
every one of them lies in a cell, is a depth-first search over 64-bit
bitsets defined here and run in D4Cell600Main.lean by `native_decide`,
which evaluates the compiled search and therefore trusts the Lean compiler;
the axiom audit there names that trust explicitly. The same search is
carried out independently in Python (multi_cap/cell600_exact.py) and in C
(multi_cap/cell600_enum.c). There is no `sorry` and no dependence on
Mathlib.
-/

namespace D4Cell600

/-- An element a + b*phi of Z[phi], with phi^2 = phi + 1. -/
abbrev Z := Int × Int

def zmul (x y : Z) : Z := (x.1 * y.1 + x.2 * y.2, x.1 * y.2 + x.2 * y.1 + x.2 * y.2)
def zadd (x y : Z) : Z := (x.1 + y.1, x.2 + y.2)

abbrev Vec := Z × Z × Z × Z

/-- Four times the inner product of two vertices (coordinates doubled). -/
def dot (u v : Vec) : Z :=
  zadd (zadd (zmul u.1 v.1) (zmul u.2.1 v.2.1))
       (zadd (zmul u.2.2.1 v.2.2.1) (zmul u.2.2.2 v.2.2.2))

/-- The 120 vertices of the 600-cell, coordinates doubled, in Z[phi]. -/
def verts : List Vec := [
VLINES
  ]

/-- The twenty-five inscribed 24-cells, as lists of indices into `verts`. -/
def cells : List (List Nat) := [
CLINES
  ]

def vert (i : Nat) : Vec := verts.getD i ((0,0),(0,0),(0,0),(0,0))

/-! ## The vertex set -/

theorem card_verts : verts.length = 120 := by decide

theorem verts_nodup : verts.Nodup := by decide

/-- Each vertex is a unit vector: 4<v,v> = 4. -/
theorem verts_norm : (verts.all fun v => dot v v == (4, 0)) = true := by decide

/-- The nine values that 4<u,v> can take. -/
def allowed : List Z := [(4,0), (-4,0), (0,2), (0,-2), (2,0), (-2,0), (-2,2), (2,-2), (0,0)]

theorem verts_inner :
    (verts.all fun u => verts.all fun v => allowed.contains (dot u v)) = true := by decide +kernel

/-- Two vertices are adjacent (36 degrees apart) when 4<u,v> = 2phi. -/
def adjacent (u v : Vec) : Bool := dot u v == (0, 2)

/-- The edge graph is 12-regular. -/
theorem degree_twelve :
    (verts.all fun u => (verts.filter (adjacent u)).length == 12) = true := by decide +kernel

/-- A pair is root-like when 4<u,v> is one of -4, -2, 0, 2, that is
    <u,v> in {-1, -1/2, 0, 1/2}. -/
def rootlike (u v : Vec) : Bool :=
  dot u v == (-4, 0) || dot u v == (-2, 0) || dot u v == (0, 0) || dot u v == (2, 0)

/-- Each vertex has 71 root-like partners: one antipode, thirty at 90 degrees,
    twenty at 60 and twenty at 120. -/
theorem rootlike_partners :
    (verts.all fun u => (verts.filter fun v => rootlike u v).length == 71) = true := by decide +kernel

/-! ## The twenty-five cells -/

theorem cells_count : cells.length = 25 := by decide

theorem cells_card : (cells.all fun c => c.length == 24) = true := by decide

theorem cells_nodup : (cells.all fun c => c.Nodup) = true := by decide

theorem cells_in_range : (cells.all fun c => c.all fun i => i < 120) = true := by decide

/-- Within a cell every pair is root-like, so each cell is a copy of the
    D4 root system and in particular an independent set of the edge graph. -/
theorem cells_rootlike :
    (cells.all fun c => c.all fun i => c.all fun j =>
      (i == j) || rootlike (vert i) (vert j)) = true := by decide +kernel

/-- Every vertex lies in exactly five of the cells. -/
theorem five_cells_through_each :
    ((List.range 120).all fun i =>
      (cells.filter fun c => c.contains i).length == 5) = true := by decide

/-! ## The enumeration -/

/-- A set of vertices as a pair of 64-bit words: bit `i` of the first word
    for vertices 0..63, bit `i - 64` of the second for 64..119. -/
structure BS where
  lo : UInt64
  hi : UInt64
deriving Inhabited

namespace BS
def empty : BS := ⟨0, 0⟩
def isEmpty (a : BS) : Bool := a.lo == 0 && a.hi == 0
def bit (i : Nat) : BS :=
  if i < 64 then ⟨(1 : UInt64) <<< i.toUInt64, 0⟩ else ⟨0, (1 : UInt64) <<< (i - 64).toUInt64⟩
def union (a b : BS) : BS := ⟨a.lo ||| b.lo, a.hi ||| b.hi⟩
def inter (a b : BS) : BS := ⟨a.lo &&& b.lo, a.hi &&& b.hi⟩
/-- `a` with the bits of `b` removed. -/
def diff (a b : BS) : BS := ⟨a.lo &&& (a.lo ^^^ b.lo), a.hi &&& (a.hi ^^^ b.hi)⟩
def full : BS := ⟨0xFFFFFFFFFFFFFFFF, 0x00FFFFFFFFFFFFFF⟩

/-- Population count of a 64-bit word, by the parallel bit count. -/
def popc64 (x : UInt64) : UInt64 :=
  let x : UInt64 := x - ((x >>> (1 : UInt64)) &&& (0x5555555555555555 : UInt64))
  let x : UInt64 := (x &&& (0x3333333333333333 : UInt64)) + ((x >>> (2 : UInt64)) &&& (0x3333333333333333 : UInt64))
  let x : UInt64 := (x + (x >>> (4 : UInt64))) &&& (0x0F0F0F0F0F0F0F0F : UInt64)
  (x * (0x0101010101010101 : UInt64)) >>> (56 : UInt64)
def card (a : BS) : Nat := (popc64 a.lo + popc64 a.hi).toNat

/-- Index of the lowest set bit of a nonzero word, by binary search. -/
def ctz64 (x : UInt64) : Nat := Id.run do
  let mut x := x
  let mut k := 0
  if x &&& (0xFFFFFFFF : UInt64) == 0 then x := x >>> (32 : UInt64); k := k + 32
  if x &&& (0xFFFF : UInt64) == 0 then x := x >>> (16 : UInt64); k := k + 16
  if x &&& (0xFF : UInt64) == 0 then x := x >>> (8 : UInt64); k := k + 8
  if x &&& (0xF : UInt64) == 0 then x := x >>> (4 : UInt64); k := k + 4
  if x &&& (0x3 : UInt64) == 0 then x := x >>> (2 : UInt64); k := k + 2
  if x &&& (0x1 : UInt64) == 0 then k := k + 1
  return k
def lowbit (a : BS) : Nat := if a.lo != 0 then ctz64 a.lo else 64 + ctz64 a.hi
def subset (a b : BS) : Bool := (a.diff b).isEmpty
end BS

def adjBS (i : Nat) : BS :=
  (List.range 120).foldl (fun m j => if adjacent (vert i) (vert j) then m.union (BS.bit j) else m) BS.empty

def masks : Array BS := (List.range 120).toArray.map adjBS

def cellMasks : Array BS :=
  cells.toArray.map fun c => c.foldl (fun m i => m.union (BS.bit i)) BS.empty

/-- Greedy clique cover of `cand`: an independent set meets each clique at
    most once, so the number of cliques bounds the size of an independent
    set inside `cand`. -/
def coverBound (ms : Array BS) (cand0 : BS) : Nat := Id.run do
  let mut cand := cand0
  let mut k := 0
  for _ in [0:120] do
    if cand.isEmpty then break
    k := k + 1
    let v := cand.lowbit
    cand := cand.diff (BS.bit v)
    let mut cc := cand.inter ms[v]!
    for _ in [0:120] do
      if cc.isEmpty then break
      let u := cc.lowbit
      cand := cand.diff (BS.bit u)
      cc := cc.inter ms[u]!
  return k

def inSomeCell (cms : Array BS) (chosen : BS) : Bool :=
  cms.any fun cm => chosen.subset cm

/-- Depth-first enumeration. Returns (number of independent sets of the
    requested size extending `chosen` inside `cand`, number of those lying
    in no cell). `fuel` bounds the recursion depth. -/
def search (ms cms : Array BS) : Nat → BS → BS → Nat → Nat × Nat
  | 0, _, _, _ => (0, 0)
  | fuel+1, chosen, cand0, need =>
    if need == 0 then (1, if inSomeCell cms chosen then 0 else 1)
    else if cand0.card < need then (0, 0)
    else if coverBound ms cand0 < need then (0, 0)
    else Id.run do
      let mut cand := cand0
      let mut tot := (0, 0)
      for _ in [0:120] do
        if cand.isEmpty then break
        if cand.card < need then break
        let v := cand.lowbit
        cand := cand.diff (BS.bit v)
        let r := search ms cms fuel (chosen.union (BS.bit v)) (cand.diff ms[v]!) (need - 1)
        tot := (tot.1 + r.1, tot.2 + r.2)
      return tot

/-- All independent sets of the given size that contain vertex 0. -/
def throughZero (size : Nat) : Nat × Nat :=
  let cand := (BS.full.diff (BS.bit 0)).diff (masks[0]!)
  search masks cellMasks 130 (BS.bit 0) cand (size - 1)

theorem count_consistent : 115 * 120 = 600 * 23 ∧ 600 = 25 * 24 := by decide

end D4Cell600

/-!
Axiom audit: every theorem in this file must report no axioms.
-/
#print axioms D4Cell600.card_verts
#print axioms D4Cell600.verts_nodup
#print axioms D4Cell600.verts_norm
#print axioms D4Cell600.verts_inner
#print axioms D4Cell600.degree_twelve
#print axioms D4Cell600.rootlike_partners
#print axioms D4Cell600.cells_count
#print axioms D4Cell600.cells_card
#print axioms D4Cell600.cells_nodup
#print axioms D4Cell600.cells_in_range
#print axioms D4Cell600.cells_rootlike
#print axioms D4Cell600.five_cells_through_each
#print axioms D4Cell600.count_consistent
'''
src = src.replace("VLINES", vlines).replace("CLINES", clines)
here = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(here, "cell600"), exist_ok=True)
out = os.path.join(here, "cell600", "D4Cell600Enum.lean")
with open(out, "w") as f:
    f.write(src)
print("wrote", out)

main = """/-
D4Cell600Main.lean

The three enumeration theorems behind prop:cell600: the compiled
depth-first search of D4Cell600Enum.lean, evaluated by `native_decide`.
Building with `lake build` in this directory compiles D4Cell600Enum to
native code first (precompileModules), so the search runs in a few seconds.
The axiom audit at the foot of the file records, for each theorem, the
trust placed in the compiler: besides `propext` and `Quot.sound`, each
depends on its own `native_decide` axiom, which is how Lean 4 names the
compiled evaluation.
-/
import D4Cell600Enum

namespace D4Cell600

/-- No independent set of size 25 contains vertex 0; by vertex transitivity the
    independence number of the edge graph is 24. -/
theorem no_25 : throughZero 25 = (0, 0) := by native_decide

/-- Exactly five independent 24-sets contain vertex 0, all of them cells. -/
theorem five_24 : throughZero 24 = (5, 0) := by native_decide

/-- Exactly 115 independent 23-sets contain vertex 0, and every one of them
    lies in a cell. Since 115 = 5 x 23, they are the five cells through the
    vertex with one of the other twenty-three vertices deleted; by vertex
    transitivity there are 115 x 120 / 23 = 600 = 25 x 24 in all. -/
theorem all_23_in_cells : throughZero 23 = (115, 0) := by native_decide

end D4Cell600

#print axioms D4Cell600.no_25
#print axioms D4Cell600.five_24
#print axioms D4Cell600.all_23_in_cells
"""
out = os.path.join(here, "cell600", "D4Cell600Main.lean")
with open(out, "w") as f:
    f.write(main)
print("wrote", out)
