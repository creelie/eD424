import D4Lean.Sound.Det
import D4Lean.Sound.Poly

/-!
# The determinant polynomials of an edge

`det (J + t Y)` is a polynomial of degree at most `5` in `t` and `det Q(J + t Y)` one of degree at
most `4`. If both agree with `pJ`, `pQ` at six distinct points, they agree everywhere.
-/

open Matrix Polynomial

namespace D4

/-- The polynomial with coefficient list `p`, lowest degree first. -/
noncomputable def toPoly : List Int → ℝ[X]
  | [] => 0
  | a :: r => C (a : ℝ) + X * toPoly r

theorem eval_toPoly (p : List Int) (t : ℝ) : (toPoly p).eval t = evalL p t := by
  induction p with
  | nil => simp [toPoly, evalL]
  | cons a r ih => simp [toPoly, evalL, ih]

theorem natDegree_toPoly (p : List Int) (n : Nat) (h : p.length ≤ n + 1) :
    (toPoly p).natDegree ≤ n := by
  induction p generalizing n with
  | nil => simp [toPoly]
  | cons a r ih =>
    cases n with
    | zero =>
      have hr : r.length = 0 := by simp only [List.length_cons] at h; omega
      rw [List.eq_nil_of_length_eq_zero hr]
      simp [toPoly]
    | succ m =>
      have h1 := ih m (by simp only [List.length_cons] at h; omega)
      have h2 := natDegree_mul_le (p := (X : ℝ[X])) (q := toPoly r)
      have h3 := natDegree_X_le (R := ℝ)
      calc (toPoly (a :: r)).natDegree
          ≤ max (C (a : ℝ)).natDegree (X * toPoly r).natDegree := natDegree_add_le _ _
        _ ≤ m + 1 := max_le (by simp) (by omega)

/-- `det (X • A + B)` evaluated at `t`. -/
theorem eval_det_lin {n : Type} [Fintype n] [DecidableEq n] (A B : Matrix n n ℝ) (t : ℝ) :
    (det ((X : ℝ[X]) • A.map C + B.map C)).eval t = (B + t • A).det := by
  have := RingHom.map_det (Polynomial.evalRingHom t) ((X : ℝ[X]) • A.map C + B.map C)
  rw [Polynomial.coe_evalRingHom] at this
  rw [this]
  congr 1
  ext i j
  simp [RingHom.mapMatrix_apply]
  ring

theorem sorted_lt (l : List Int) (h : sorted l = true) (i j : Nat) (hij : i < j)
    (hj : j < l.length) : vg l i < vg l j := by
  induction l generalizing i j with
  | nil => simp at hj
  | cons a t ih =>
    cases t with
    | nil => simp at hj; omega
    | cons b r =>
      simp only [sorted, Bool.and_eq_true, decide_eq_true_eq] at h
      obtain ⟨hab, h⟩ := h
      cases j with
      | zero => omega
      | succ j =>
        rw [vg_cons_succ]
        cases i with
        | zero =>
          rw [vg_cons_zero]
          cases j with
          | zero => exact hab
          | succ j =>
            have := ih h 0 (j + 1) (by omega) (by simp at hj ⊢; omega)
            rw [vg_cons_zero] at this
            omega
        | succ i =>
          rw [vg_cons_succ]
          exact ih h i j (by omega) (by simp at hj ⊢; omega)

/-- The entries of the list matrix built from a function. -/
theorem ent_range (f : Nat → Nat → Int) (i j : Nat) (hi : i < 5) (hj : j < 5) :
    ent ((List.range 5).map fun a => (List.range 5).map fun b => f a b) i j = f i j := by
  simp [ent, List.getD_eq_getElem?_getD, hi, hj]

theorem qb_add_smul (A B : M5) (t : ℝ) : Qb (A + t • B) = Qb A + t • Qb B := by
  ext i j; simp [Qb]

theorem qb_smul (A : M5) (c : ℝ) : Qb (c • A) = c • Qb A := by
  ext i j; simp [Qb]

/-- The two determinant polynomials of an edge. -/
theorem samples_sound (J Y : Mat) (pJ pQ : List Int) (sd : Int) (sn : List Int) (Es : List Mat)
    (h : samplesOK J Y pJ pQ sd sn Es = true) (hpJ : pJ.length ≤ 6) (hpQ : pQ.length ≤ 5)
    (t : ℝ) :
    (toR J + t • toR Y).det = evalL pJ t ∧ (Qb (toR J + t • toR Y)).det = evalL pQ t := by
  simp only [samplesOK, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq, allN_iff] at h
  obtain ⟨⟨⟨⟨hsd, hsn⟩, hEs⟩, hsort⟩, hall⟩ := h
  have hsdR : (0 : ℝ) < sd := by exact_mod_cast hsd
  have hsd0 : (sd : ℝ) ≠ 0 := hsdR.ne'
  -- the sample points
  set f : Fin 6 → ℝ := fun j => (vg sn j : ℝ) / sd with hf
  have hinj : Function.Injective f := by
    apply StrictMono.injective
    intro i j hij
    have := sorted_lt sn hsort i j hij (by rw [hsn]; exact j.2)
    simp only [hf]
    exact div_lt_div_of_pos_right (by exact_mod_cast this) hsdR
  -- the values at the sample points
  have hval : ∀ j : Fin 6, (toR J + f j • toR Y).det = evalL pJ (f j) ∧
      (Qb (toR J + f j • toR Y)).det = evalL pQ (f j) := by
    intro j
    set s := vg sn j
    set A : Mat := (List.range 5).map fun a => (List.range 5).map fun b =>
      sd * ent J a b + s * ent Y a b with hA
    have hd := detOK_sound A (Es.getD j []) _ _ (hall j j.2)
    have hAR : toR A = (sd : ℝ) • (toR J + f j • toR Y) := by
      ext a b
      simp only [toR, toZ, Matrix.map_apply, Matrix.of_apply, Matrix.smul_apply, Matrix.add_apply,
        smul_eq_mul, hA, ent_range _ a b a.2 b.2, hf]
      push_cast
      field_simp
      simp only [s]
      ring
    constructor
    · have h1 := det_toR A
      rw [hd.1, hAR, Matrix.det_smul, hom_eq pJ s sd 5 hsd (by omega)] at h1
      simp only [Fintype.card_fin] at h1
      have h2 : (sd : ℝ) ^ 5 ≠ 0 := pow_ne_zero _ hsd0
      exact mul_left_cancel₀ h2 h1
    · have h1 := detQ_toR A
      rw [hd.2, hAR, qb_smul, Matrix.det_smul, hom_eq pQ s sd 4 hsd (by omega)] at h1
      simp only [Fintype.card_fin] at h1
      have h2 : (sd : ℝ) ^ 4 ≠ 0 := pow_ne_zero _ hsd0
      exact mul_left_cancel₀ h2 h1
  -- interpolation
  constructor
  · have hP := eq_of_natDegree_lt_card_of_eval_eq
      (det ((X : ℝ[X]) • (toR Y).map C + (toR J).map C)) (toPoly pJ) hinj
      (fun j => by rw [eval_det_lin, eval_toPoly]; exact (hval j).1)
      (by
        have h1 := natDegree_det_X_add_C_le (toR Y) (toR J)
        have h2 := natDegree_toPoly pJ 5 (by omega)
        simp only [Fintype.card_fin] at h1 ⊢
        omega)
    have := congrArg (Polynomial.eval t) hP
    rwa [eval_det_lin, eval_toPoly] at this
  · have hP := eq_of_natDegree_lt_card_of_eval_eq
      (det ((X : ℝ[X]) • (Qb (toR Y)).map C + (Qb (toR J)).map C)) (toPoly pQ) hinj
      (fun j => by rw [eval_det_lin, eval_toPoly, ← qb_add_smul]; exact (hval j).2)
      (by
        have h1 := natDegree_det_X_add_C_le (Qb (toR Y)) (Qb (toR J))
        have h2 := natDegree_toPoly pQ 4 (by omega)
        simp only [Fintype.card_fin] at h1 ⊢
        omega)
    have := congrArg (Polynomial.eval t) hP
    rwa [eval_det_lin, eval_toPoly, ← qb_add_smul] at this

end D4
