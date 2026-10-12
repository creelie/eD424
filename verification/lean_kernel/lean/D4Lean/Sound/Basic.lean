import D4Lean.Defs

/-!
# Soundness of the list functions

`sumN`, `prodN`, `allN`, `dot`, `qf` and the box enumeration, in terms of finite sums over `Fin n`.
-/

open Matrix

namespace D4

theorem sumN_eq (n : Nat) (f : Nat → Int) : sumN n f = ∑ i : Fin n, f i := by
  induction n with
  | zero => simp [sumN]
  | succ n ih => rw [sumN, ih, Fin.sum_univ_castSucc]; simp

theorem prodN_eq (n : Nat) (f : Nat → Int) : prodN n f = ∏ i : Fin n, f i := by
  induction n with
  | zero => simp [prodN]
  | succ n ih => rw [prodN, ih, Fin.prod_univ_castSucc]; simp

theorem allN_iff (n : Nat) (p : Nat → Bool) : allN n p = true ↔ ∀ i < n, p i = true := by
  induction n with
  | zero => simp [allN]
  | succ n ih =>
    rw [allN, Bool.and_eq_true, ih]
    constructor
    · rintro ⟨h1, h2⟩ i hi
      rcases Nat.lt_succ_iff_lt_or_eq.mp hi with h | h
      · exact h1 i h
      · exact h ▸ h2
    · intro h
      exact ⟨fun i hi => h i (Nat.lt_succ_of_lt hi), h n (Nat.lt_succ_self n)⟩

theorem vg_cons_zero (a : Int) (l : List Int) : vg (a :: l) 0 = a := rfl

theorem vg_cons_succ (a : Int) (l : List Int) (i : Nat) : vg (a :: l) (i + 1) = vg l i := rfl

theorem dot_eq (a b : List Int) (n : Nat) (ha : a.length = n) (hb : b.length = n) :
    dot a b = ∑ i : Fin n, vg a i * vg b i := by
  induction a generalizing b n with
  | nil =>
    subst ha
    simp [dot]
  | cons x xs ih =>
    cases b with
    | nil => simp at hb; subst hb; simp at ha
    | cons y ys =>
      cases n with
      | zero => simp at ha
      | succ m =>
        simp only [List.length_cons, Nat.add_right_cancel_iff] at ha hb
        rw [dot, ih ys m ha hb, Fin.sum_univ_succ]
        simp [vg_cons_zero, vg_cons_succ]

theorem shape_iff (A : Mat) (n : Nat) :
    shape A n = true ↔ A.length = n ∧ ∀ r ∈ A, r.length = n := by
  simp [shape]

theorem row_length (A : Mat) (n : Nat) (h : shape A n = true) (i : Nat) (hi : i < n) :
    (A.getD i []).length = n := by
  rw [shape_iff] at h
  rw [List.getD_eq_getElem _ _ (by omega)]
  exact h.2 _ (List.getElem_mem _)

theorem vg_map_dot (A : Mat) (k : List Int) (i : Nat) :
    vg (A.map fun r => dot r k) i = dot (A.getD i []) k := by
  have h := List.getD_map A [] (n := i) (fun r => dot r k)
  simp only [dot] at h
  exact h

theorem qf_eq (A : Mat) (k : List Int) (hA : shape A 5 = true) (hk : k.length = 5) :
    qf A k = ∑ i : Fin 5, ∑ j : Fin 5, vg k i * ent A i j * vg k j := by
  have hm : (A.map fun r => dot r k).length = 5 := by
    rw [List.length_map]; exact ((shape_iff A 5).mp hA).1
  rw [qf, dot_eq k _ 5 hk hm]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [vg_map_dot, dot_eq _ k 5 (row_length A 5 hA i i.2) hk, Finset.mul_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  simp only [ent, vg]
  ring

theorem vg_ofFn (k : Fin 5 → ℤ) (i : Fin 5) : vg (List.ofFn k) i = k i := by
  fin_cases i <;> rfl

/-- `qf` at a point given as a function. -/
theorem qf_ofFn (A : Mat) (hA : shape A 5 = true) (k : Fin 5 → ℤ) :
    qf A (List.ofFn k) = ∑ i : Fin 5, ∑ j : Fin 5, k i * ent A i j * k j := by
  rw [qf_eq A _ hA (by simp)]
  simp only [vg_ofFn]

theorem qform_toR (A : Mat) (k : Fin 5 → ℤ) :
    qform (toR A) k = ((∑ i : Fin 5, ∑ j : Fin 5, k i * ent A i j * k j : ℤ) : ℝ) := by
  simp [qform, toR, toZ]

theorem qform_neg (J : M5) (k : Fin 5 → ℤ) : qform J (-k) = qform J k := by
  simp [qform]

/-! ### The box -/

theorem rangeAll_sound (p : Int → Bool) (lo : Int) (n : Nat) (h : rangeAll p lo n = true)
    (v : Int) (h1 : lo ≤ v) (h2 : v < lo + n) : p v = true := by
  induction n generalizing lo with
  | zero => simp at h2; omega
  | succ n ih =>
    rw [rangeAll, Bool.and_eq_true] at h
    rcases eq_or_lt_of_le h1 with h3 | h3
    · exact h3 ▸ h.1
    · exact ih (lo + 1) h.2 (by omega) (by push_cast at h2; omega)

theorem boxAll_sound (bx : List (Int × Nat)) (p : List Int → Bool) (h : boxAll bx p = true)
    (xs : List Int) (hx : List.Forall₂ (fun (b : Int × Nat) (x : Int) => b.1 ≤ x ∧ x < b.1 + b.2)
      bx xs) : p xs = true := by
  induction hx generalizing p with
  | nil => simpa [boxAll] using h
  | @cons b x bs xs hb _ ih =>
    rw [boxAll] at h
    exact ih (fun ys => p (x :: ys)) (rangeAll_sound _ _ _ h x hb.1 hb.2)

end D4
