/-
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
    ((-2, 0), (0, 0), (0, 0), (0, 0)),
    ((-1, 0), (-1, 0), (-1, 0), (-1, 0)),
    ((-1, 0), (-1, 0), (-1, 0), (1, 0)),
    ((-1, 0), (-1, 0), (1, 0), (-1, 0)),
    ((-1, 0), (-1, 0), (1, 0), (1, 0)),
    ((-1, 0), (-1, 1), (0, -1), (0, 0)),
    ((-1, 0), (-1, 1), (0, 1), (0, 0)),
    ((-1, 0), (0, -1), (0, 0), (-1, 1)),
    ((-1, 0), (0, -1), (0, 0), (1, -1)),
    ((-1, 0), (0, 0), (-1, 1), (0, -1)),
    ((-1, 0), (0, 0), (-1, 1), (0, 1)),
    ((-1, 0), (0, 0), (1, -1), (0, -1)),
    ((-1, 0), (0, 0), (1, -1), (0, 1)),
    ((-1, 0), (0, 1), (0, 0), (-1, 1)),
    ((-1, 0), (0, 1), (0, 0), (1, -1)),
    ((-1, 0), (1, -1), (0, -1), (0, 0)),
    ((-1, 0), (1, -1), (0, 1), (0, 0)),
    ((-1, 0), (1, 0), (-1, 0), (-1, 0)),
    ((-1, 0), (1, 0), (-1, 0), (1, 0)),
    ((-1, 0), (1, 0), (1, 0), (-1, 0)),
    ((-1, 0), (1, 0), (1, 0), (1, 0)),
    ((-1, 1), (-1, 0), (0, 0), (0, -1)),
    ((-1, 1), (-1, 0), (0, 0), (0, 1)),
    ((-1, 1), (0, -1), (-1, 0), (0, 0)),
    ((-1, 1), (0, -1), (1, 0), (0, 0)),
    ((-1, 1), (0, 0), (0, -1), (-1, 0)),
    ((-1, 1), (0, 0), (0, -1), (1, 0)),
    ((-1, 1), (0, 0), (0, 1), (-1, 0)),
    ((-1, 1), (0, 0), (0, 1), (1, 0)),
    ((-1, 1), (0, 1), (-1, 0), (0, 0)),
    ((-1, 1), (0, 1), (1, 0), (0, 0)),
    ((-1, 1), (1, 0), (0, 0), (0, -1)),
    ((-1, 1), (1, 0), (0, 0), (0, 1)),
    ((0, -1), (-1, 0), (-1, 1), (0, 0)),
    ((0, -1), (-1, 0), (1, -1), (0, 0)),
    ((0, -1), (-1, 1), (0, 0), (-1, 0)),
    ((0, -1), (-1, 1), (0, 0), (1, 0)),
    ((0, -1), (0, 0), (-1, 0), (-1, 1)),
    ((0, -1), (0, 0), (-1, 0), (1, -1)),
    ((0, -1), (0, 0), (1, 0), (-1, 1)),
    ((0, -1), (0, 0), (1, 0), (1, -1)),
    ((0, -1), (1, -1), (0, 0), (-1, 0)),
    ((0, -1), (1, -1), (0, 0), (1, 0)),
    ((0, -1), (1, 0), (-1, 1), (0, 0)),
    ((0, -1), (1, 0), (1, -1), (0, 0)),
    ((0, 0), (-2, 0), (0, 0), (0, 0)),
    ((0, 0), (-1, 0), (0, -1), (-1, 1)),
    ((0, 0), (-1, 0), (0, -1), (1, -1)),
    ((0, 0), (-1, 0), (0, 1), (-1, 1)),
    ((0, 0), (-1, 0), (0, 1), (1, -1)),
    ((0, 0), (-1, 1), (-1, 0), (0, -1)),
    ((0, 0), (-1, 1), (-1, 0), (0, 1)),
    ((0, 0), (-1, 1), (1, 0), (0, -1)),
    ((0, 0), (-1, 1), (1, 0), (0, 1)),
    ((0, 0), (0, -1), (-1, 1), (-1, 0)),
    ((0, 0), (0, -1), (-1, 1), (1, 0)),
    ((0, 0), (0, -1), (1, -1), (-1, 0)),
    ((0, 0), (0, -1), (1, -1), (1, 0)),
    ((0, 0), (0, 0), (-2, 0), (0, 0)),
    ((0, 0), (0, 0), (0, 0), (-2, 0)),
    ((0, 0), (0, 0), (0, 0), (2, 0)),
    ((0, 0), (0, 0), (2, 0), (0, 0)),
    ((0, 0), (0, 1), (-1, 1), (-1, 0)),
    ((0, 0), (0, 1), (-1, 1), (1, 0)),
    ((0, 0), (0, 1), (1, -1), (-1, 0)),
    ((0, 0), (0, 1), (1, -1), (1, 0)),
    ((0, 0), (1, -1), (-1, 0), (0, -1)),
    ((0, 0), (1, -1), (-1, 0), (0, 1)),
    ((0, 0), (1, -1), (1, 0), (0, -1)),
    ((0, 0), (1, -1), (1, 0), (0, 1)),
    ((0, 0), (1, 0), (0, -1), (-1, 1)),
    ((0, 0), (1, 0), (0, -1), (1, -1)),
    ((0, 0), (1, 0), (0, 1), (-1, 1)),
    ((0, 0), (1, 0), (0, 1), (1, -1)),
    ((0, 0), (2, 0), (0, 0), (0, 0)),
    ((0, 1), (-1, 0), (-1, 1), (0, 0)),
    ((0, 1), (-1, 0), (1, -1), (0, 0)),
    ((0, 1), (-1, 1), (0, 0), (-1, 0)),
    ((0, 1), (-1, 1), (0, 0), (1, 0)),
    ((0, 1), (0, 0), (-1, 0), (-1, 1)),
    ((0, 1), (0, 0), (-1, 0), (1, -1)),
    ((0, 1), (0, 0), (1, 0), (-1, 1)),
    ((0, 1), (0, 0), (1, 0), (1, -1)),
    ((0, 1), (1, -1), (0, 0), (-1, 0)),
    ((0, 1), (1, -1), (0, 0), (1, 0)),
    ((0, 1), (1, 0), (-1, 1), (0, 0)),
    ((0, 1), (1, 0), (1, -1), (0, 0)),
    ((1, -1), (-1, 0), (0, 0), (0, -1)),
    ((1, -1), (-1, 0), (0, 0), (0, 1)),
    ((1, -1), (0, -1), (-1, 0), (0, 0)),
    ((1, -1), (0, -1), (1, 0), (0, 0)),
    ((1, -1), (0, 0), (0, -1), (-1, 0)),
    ((1, -1), (0, 0), (0, -1), (1, 0)),
    ((1, -1), (0, 0), (0, 1), (-1, 0)),
    ((1, -1), (0, 0), (0, 1), (1, 0)),
    ((1, -1), (0, 1), (-1, 0), (0, 0)),
    ((1, -1), (0, 1), (1, 0), (0, 0)),
    ((1, -1), (1, 0), (0, 0), (0, -1)),
    ((1, -1), (1, 0), (0, 0), (0, 1)),
    ((1, 0), (-1, 0), (-1, 0), (-1, 0)),
    ((1, 0), (-1, 0), (-1, 0), (1, 0)),
    ((1, 0), (-1, 0), (1, 0), (-1, 0)),
    ((1, 0), (-1, 0), (1, 0), (1, 0)),
    ((1, 0), (-1, 1), (0, -1), (0, 0)),
    ((1, 0), (-1, 1), (0, 1), (0, 0)),
    ((1, 0), (0, -1), (0, 0), (-1, 1)),
    ((1, 0), (0, -1), (0, 0), (1, -1)),
    ((1, 0), (0, 0), (-1, 1), (0, -1)),
    ((1, 0), (0, 0), (-1, 1), (0, 1)),
    ((1, 0), (0, 0), (1, -1), (0, -1)),
    ((1, 0), (0, 0), (1, -1), (0, 1)),
    ((1, 0), (0, 1), (0, 0), (-1, 1)),
    ((1, 0), (0, 1), (0, 0), (1, -1)),
    ((1, 0), (1, -1), (0, -1), (0, 0)),
    ((1, 0), (1, -1), (0, 1), (0, 0)),
    ((1, 0), (1, 0), (-1, 0), (-1, 0)),
    ((1, 0), (1, 0), (-1, 0), (1, 0)),
    ((1, 0), (1, 0), (1, 0), (-1, 0)),
    ((1, 0), (1, 0), (1, 0), (1, 0)),
    ((2, 0), (0, 0), (0, 0), (0, 0))
  ]

/-- The twenty-five inscribed 24-cells, as lists of indices into `verts`. -/
def cells : List (List Nat) := [
    [16, 19, 20, 27, 28, 30, 41, 42, 44, 54, 55, 59, 60, 64, 65, 75, 77, 78, 89, 91, 92, 99, 100, 103],
    [15, 17, 18, 25, 26, 29, 41, 42, 43, 56, 57, 59, 60, 62, 63, 76, 77, 78, 90, 93, 94, 101, 102, 104],
    [14, 18, 20, 29, 30, 32, 38, 40, 42, 50, 52, 58, 61, 67, 69, 77, 79, 81, 87, 89, 90, 99, 101, 105],
    [13, 17, 19, 29, 30, 31, 37, 39, 41, 51, 53, 58, 61, 66, 68, 78, 80, 82, 88, 89, 90, 100, 102, 106],
    [8, 11, 15, 21, 23, 25, 40, 42, 44, 49, 52, 55, 64, 67, 70, 75, 77, 79, 94, 96, 98, 104, 108, 111],
    [8, 9, 16, 21, 24, 27, 38, 42, 43, 47, 50, 57, 62, 69, 72, 76, 77, 81, 92, 95, 98, 103, 110, 111],
    [7, 12, 15, 22, 23, 26, 39, 41, 44, 48, 53, 54, 65, 66, 71, 75, 78, 80, 93, 96, 97, 104, 107, 112],
    [7, 10, 16, 22, 24, 28, 37, 41, 43, 46, 51, 56, 63, 68, 73, 76, 78, 82, 91, 95, 97, 103, 109, 112],
    [6, 10, 13, 28, 30, 32, 33, 35, 37, 49, 52, 55, 64, 67, 70, 82, 84, 86, 87, 89, 91, 106, 109, 113],
    [6, 9, 14, 27, 30, 31, 33, 36, 38, 48, 53, 54, 65, 66, 71, 81, 83, 86, 88, 89, 92, 105, 110, 113],
    [5, 12, 13, 26, 29, 32, 34, 35, 39, 47, 50, 57, 62, 69, 72, 80, 84, 85, 87, 90, 93, 106, 107, 114],
    [5, 11, 14, 25, 29, 31, 34, 36, 40, 46, 51, 56, 63, 68, 73, 79, 83, 85, 88, 90, 94, 105, 108, 114],
    [4, 12, 20, 22, 28, 32, 34, 40, 44, 45, 46, 49, 70, 73, 74, 75, 79, 85, 87, 91, 97, 99, 107, 115],
    [3, 11, 19, 21, 27, 31, 34, 39, 44, 45, 47, 48, 71, 72, 74, 75, 80, 85, 88, 92, 98, 100, 108, 116],
    [3, 4, 6, 24, 27, 28, 34, 35, 36, 56, 57, 59, 60, 62, 63, 83, 84, 85, 91, 92, 95, 113, 115, 116],
    [2, 10, 18, 22, 26, 32, 33, 38, 43, 45, 47, 48, 71, 72, 74, 76, 81, 86, 87, 93, 97, 101, 109, 117],
    [2, 4, 8, 22, 23, 24, 36, 38, 40, 51, 53, 58, 61, 66, 68, 79, 81, 83, 95, 96, 97, 111, 115, 117],
    [1, 9, 17, 21, 25, 31, 33, 37, 43, 45, 46, 49, 70, 73, 74, 76, 82, 86, 88, 94, 98, 102, 110, 118],
    [1, 3, 7, 21, 23, 24, 35, 37, 39, 50, 52, 58, 61, 67, 69, 80, 82, 84, 95, 96, 98, 112, 116, 118],
    [1, 2, 5, 23, 25, 26, 33, 35, 36, 54, 55, 59, 60, 64, 65, 83, 84, 86, 93, 94, 96, 114, 117, 118],
    [0, 4, 6, 8, 9, 12, 13, 15, 17, 49, 53, 57, 62, 66, 70, 102, 104, 106, 107, 110, 111, 113, 115, 119],
    [0, 3, 6, 7, 10, 11, 14, 15, 18, 48, 52, 56, 63, 67, 71, 101, 104, 105, 108, 109, 112, 113, 116, 119],
    [0, 2, 5, 8, 10, 11, 13, 16, 19, 47, 51, 55, 64, 68, 72, 100, 103, 106, 108, 109, 111, 114, 117, 119],
    [0, 1, 5, 7, 9, 12, 14, 16, 20, 46, 50, 54, 65, 69, 73, 99, 103, 105, 107, 110, 112, 114, 118, 119],
    [0, 1, 2, 3, 4, 17, 18, 19, 20, 45, 58, 59, 60, 61, 74, 99, 100, 101, 102, 115, 116, 117, 118, 119]
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
