/-
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

def roots : List (List Int) := [
  [1, 1, 0, 0],
  [1, -1, 0, 0],
  [-1, 1, 0, 0],
  [-1, -1, 0, 0],
  [1, 0, 1, 0],
  [1, 0, -1, 0],
  [-1, 0, 1, 0],
  [-1, 0, -1, 0],
  [1, 0, 0, 1],
  [1, 0, 0, -1],
  [-1, 0, 0, 1],
  [-1, 0, 0, -1],
  [0, 1, 1, 0],
  [0, 1, -1, 0],
  [0, -1, 1, 0],
  [0, -1, -1, 0],
  [0, 1, 0, 1],
  [0, 1, 0, -1],
  [0, -1, 0, 1],
  [0, -1, 0, -1],
  [0, 0, 1, 1],
  [0, 0, 1, -1],
  [0, 0, -1, 1],
  [0, 0, -1, -1]]

def a18 : List Nat := [0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 20, 21, 22, 23]
def from5 : List Nat := [0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 13, 15, 16, 17, 18, 19, 22, 23]
def from12 : List Nat := [0, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17, 19, 20, 21, 23]

def perms : List (List Nat) := [
  [0, 1, 2, 3],
  [0, 1, 3, 2],
  [0, 2, 1, 3],
  [0, 2, 3, 1],
  [0, 3, 1, 2],
  [0, 3, 2, 1],
  [1, 0, 2, 3],
  [1, 0, 3, 2],
  [1, 2, 0, 3],
  [1, 2, 3, 0],
  [1, 3, 0, 2],
  [1, 3, 2, 0],
  [2, 0, 1, 3],
  [2, 0, 3, 1],
  [2, 1, 0, 3],
  [2, 1, 3, 0],
  [2, 3, 0, 1],
  [2, 3, 1, 0],
  [3, 0, 1, 2],
  [3, 0, 2, 1],
  [3, 1, 0, 2],
  [3, 1, 2, 0],
  [3, 2, 0, 1],
  [3, 2, 1, 0]]

def signs : List (List Int) := [
  [1, 1, 1, 1],
  [1, 1, 1, -1],
  [1, 1, -1, 1],
  [1, 1, -1, -1],
  [1, -1, 1, 1],
  [1, -1, 1, -1],
  [1, -1, -1, 1],
  [1, -1, -1, -1],
  [-1, 1, 1, 1],
  [-1, 1, 1, -1],
  [-1, 1, -1, 1],
  [-1, 1, -1, -1],
  [-1, -1, 1, 1],
  [-1, -1, 1, -1],
  [-1, -1, -1, 1],
  [-1, -1, -1, -1]]

def indexIn : List (List Int) -> List Int -> Nat -> Option Nat
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

def m5 : List (List Int) := [[2, 0, 0, 0], [0, 0, 2, 0], [0, -2, 0, 0], [0, 0, 0, 2]]

def apply5 (v : List Int) : List Int :=
  (m5.map fun row => (List.zipWith (fun a b => a * b) row v).foldl (fun s x => s + x) 0).map
    (fun x => x / 2)

theorem congruent_from5 :
    ((List.range 4).all (fun i => (List.range 4).all (fun j =>
       ((List.zipWith (fun a b => a * b) (m5.getD i []) (m5.getD j [])).foldl
          (fun s x => s + x) 0) == (if i == j then 4 else 0))) &&
     roots.all (fun v => (m5.map fun row =>
       (List.zipWith (fun a b => a * b) row v).foldl (fun s x => s + x) 0).all
         (fun x => x % 2 == 0)) &&
     sameSet (imageIdx apply5 allRoots) allRoots &&
     nodup (imageIdx apply5 allRoots) &&
     sameSet (imageIdx apply5 a18) from5 &&
     nodup (imageIdx apply5 a18)) = true := by
  decide +kernel

def m12 : List (List Int) := [[-1, -1, -1, -1], [1, 1, -1, -1], [-1, 1, -1, 1], [1, -1, -1, 1]]

def apply12 (v : List Int) : List Int :=
  (m12.map fun row => (List.zipWith (fun a b => a * b) row v).foldl (fun s x => s + x) 0).map
    (fun x => x / 2)

theorem congruent_from12 :
    ((List.range 4).all (fun i => (List.range 4).all (fun j =>
       ((List.zipWith (fun a b => a * b) (m12.getD i []) (m12.getD j [])).foldl
          (fun s x => s + x) 0) == (if i == j then 4 else 0))) &&
     roots.all (fun v => (m12.map fun row =>
       (List.zipWith (fun a b => a * b) row v).foldl (fun s x => s + x) 0).all
         (fun x => x % 2 == 0)) &&
     sameSet (imageIdx apply12 allRoots) allRoots &&
     nodup (imageIdx apply12 allRoots) &&
     sameSet (imageIdx apply12 a18) from12 &&
     nodup (imageIdx apply12 a18)) = true := by
  decide +kernel

end D4A18

#print axioms D4A18.stab_count
#print axioms D4A18.congruent_from5
#print axioms D4A18.congruent_from12
