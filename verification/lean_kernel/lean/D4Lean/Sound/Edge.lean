import D4Lean.Sound.Samples
import D4Lean.Sound.Mem

/-!
# Soundness of the edge checks

The sign certificate, the end point of a bounded edge, and the direction of an unbounded edge.
-/

open Matrix

namespace D4

/-- What the sign certificate proves at a point `t`, in terms of the coefficient lists `pJ` of
`det J` and `g` of `det Q - 256`. -/
def Good (pJ g : List Int) (wd : Int) (eqs : List Int) (t : ℝ) : Prop :=
  (0 < evalL pJ t ∨ 0 ≤ evalL g t) ∧
    (evalL pJ t = 0 → evalL g t = 0 → ∃ u ∈ eqs, t = (u : ℝ) / wd)

theorem pointOK_sound (pJ g : List Int) (wd : Int) (eqs : List Int) (u : Int) (hwd : 0 < wd)
    (hpJ : pJ.length ≤ 6) (hg : g.length ≤ 5) (h : pointOK pJ g wd eqs u = true) :
    Good pJ g wd eqs ((u : ℝ) / wd) := by
  have ha := hom_eq pJ u wd 5 hwd (by omega)
  have hb := hom_eq g u wd 4 hwd (by omega)
  have hw : (0 : ℝ) < wd := by exact_mod_cast hwd
  have hw6 : (0 : ℝ) < (wd : ℝ) ^ 5 := by positivity
  have hw5 : (0 : ℝ) < (wd : ℝ) ^ 4 := by positivity
  simp only [pointOK, Bool.and_eq_true] at h
  obtain ⟨h1, h2⟩ := h
  constructor
  · simp only [Bool.or_eq_true, decide_eq_true_eq] at h1
    rcases h1 with h1 | h1
    · left
      have : (0 : ℝ) < (hom pJ u wd 5 : ℤ) := by exact_mod_cast h1
      rw [ha] at this
      exact pos_of_mul_pos_right this hw6.le
    · right
      have : (0 : ℝ) ≤ (hom g u wd 4 : ℤ) := by exact_mod_cast h1
      rw [hb] at this
      exact nonneg_of_mul_nonneg_right this hw5
  · intro hA hB
    have hA' : hom pJ u wd 5 = 0 := by
      have : ((hom pJ u wd 5 : ℤ) : ℝ) = 0 := by rw [ha, hA, mul_zero]
      exact_mod_cast this
    have hB' : hom g u wd 4 = 0 := by
      have : ((hom g u wd 4 : ℤ) : ℝ) = 0 := by rw [hb, hB, mul_zero]
      exact_mod_cast this
    rw [hA', hB'] at h2
    simp at h2
    exact ⟨u, h2, rfl⟩

theorem piece_aux (P : List Int) (n : Nat) (wd u v : Int) (hwd : 0 < wd)
    (hl : P.length ≤ n + 1)
    (hp : posBound (comp P u (v - u) wd n) = true ∨ posBound (comp P v (u - v) wd n) = true)
    (t : ℝ) (h1 : (u : ℝ) / wd < t) (h2 : t < (v : ℝ) / wd) : 0 < evalL P t := by
  have hw : (0 : ℝ) < wd := by exact_mod_cast hwd
  have hw0 : (wd : ℝ) ≠ 0 := hw.ne'
  rw [div_lt_iff₀ hw] at h1
  rw [lt_div_iff₀ hw] at h2
  have hvu : (0 : ℝ) < (v : ℝ) - u := by linarith
  have hvu0 : (v : ℝ) - u ≠ 0 := hvu.ne'
  rcases hp with hp | hp
  · have h0 : 0 < (t * wd - u) / (v - u) := div_pos (by linarith) hvu
    have h1' : (t * wd - u) / (v - u) ≤ 1 := by rw [div_le_one hvu]; linarith
    have := comp_pos P u (v - u) wd n hwd hl hp _ h0 h1'
    convert this using 2
    push_cast
    field_simp
    ring
  · have h0 : 0 < (v - t * wd) / (v - u) := div_pos (by linarith) hvu
    have h1' : (v - t * wd) / (v - u) ≤ 1 := by rw [div_le_one hvu]; linarith
    have := comp_pos P v (u - v) wd n hwd hl hp _ h0 h1'
    convert this using 2
    push_cast
    field_simp
    ring

theorem pieceOK_sound (pJ g : List Int) (wd : Int) (m d : Bool) (u v : Int) (hwd : 0 < wd)
    (hpJ : pJ.length ≤ 6) (hg : g.length ≤ 5) (h : pieceOK pJ g wd m d u v = true) (t : ℝ)
    (h1 : (u : ℝ) / wd < t) (h2 : t < (v : ℝ) / wd) :
    0 < evalL (if m then pJ else g) t := by
  cases m <;> cases d <;> simp only [pieceOK, ↓reduceIte, Bool.false_eq_true] at h ⊢
  · exact piece_aux g 4 wd u v hwd (by omega) (Or.inr h) t h1 h2
  · exact piece_aux g 4 wd u v hwd (by omega) (Or.inl h) t h1 h2
  · exact piece_aux pJ 5 wd u v hwd (by omega) (Or.inr h) t h1 h2
  · exact piece_aux pJ 5 wd u v hwd (by omega) (Or.inl h) t h1 h2

theorem good_of_pos (pJ g : List Int) (wd : Int) (eqs : List Int) (m : Bool) (t : ℝ)
    (h : 0 < evalL (if m then pJ else g) t) : Good pJ g wd eqs t := by
  cases m
  · simp only [Bool.false_eq_true, ↓reduceIte] at h
    exact ⟨Or.inr h.le, fun _ h2 => absurd h2 h.ne'⟩
  · simp only [↓reduceIte] at h
    exact ⟨Or.inl h, fun h1 _ => absurd h1 h.ne'⟩

/-- Breakpoints and pieces cover `[u₀ / wd, u_last / wd]`. -/
theorem pieces_good (pJ g : List Int) (wd : Int) (eqs : List Int) (hwd : 0 < wd)
    (hpJ : pJ.length ≤ 6) (hg : g.length ≤ 5) (us : List Int) :
    ∀ ps : List (Bool × Bool), piecesOK pJ g wd us ps = true →
    (∀ x ∈ us, pointOK pJ g wd eqs x = true) → ∀ u0 ul : Int, us.head? = some u0 →
    us.getLast? = some ul → ∀ t : ℝ, (u0 : ℝ) / wd ≤ t → t ≤ (ul : ℝ) / wd →
    Good pJ g wd eqs t := by
  induction us with
  | nil => intro ps _ _ u0 ul h0; simp at h0
  | cons u rest ih =>
    intro ps hps hpt u0 ul h0 hl t ht0 htl
    simp only [List.head?_cons, Option.some.injEq] at h0
    subst h0
    cases rest with
    | nil =>
      simp only [List.getLast?_singleton, Option.some.injEq] at hl
      subst hl
      have : t = (u : ℝ) / wd := le_antisymm htl ht0
      subst this
      exact pointOK_sound pJ g wd eqs u hwd hpJ hg (hpt u (by simp))
    | cons v r =>
      cases ps with
      | nil => simp [piecesOK] at hps
      | cons p ps =>
        obtain ⟨m, d⟩ := p
        simp only [piecesOK, Bool.and_eq_true] at hps
        obtain ⟨hpc, hps⟩ := hps
        rw [List.getLast?_cons_cons] at hl
        by_cases htv : t ≤ (v : ℝ) / wd
        · rcases eq_or_lt_of_le ht0 with h | h
          · rw [← h]; exact pointOK_sound pJ g wd eqs u hwd hpJ hg (hpt u (by simp))
          · rcases eq_or_lt_of_le htv with h' | h'
            · rw [h']; exact pointOK_sound pJ g wd eqs v hwd hpJ hg (hpt v (by simp))
            · exact good_of_pos pJ g wd eqs m t
                (pieceOK_sound pJ g wd m d u v hwd hpJ hg hpc t h h')
        · exact ih ps hps (fun x hx => hpt x (by simp [hx])) v ul rfl hl t
            (le_of_lt (not_le.mp htv)) htl

theorem tail_aux (P : List Int) (n : Nat) (wd u : Int) (hwd : 0 < wd) (hl : P.length ≤ n + 1)
    (hp : posInf (comp P u 1 wd n) = true) (t : ℝ) (h1 : (u : ℝ) / wd < t) :
    0 < evalL P t := by
  have hw : (0 : ℝ) < wd := by exact_mod_cast hwd
  have hw0 : (wd : ℝ) ≠ 0 := hw.ne'
  rw [div_lt_iff₀ hw] at h1
  have := comp_posInf P u wd n hwd hl hp (t * wd - u) (by linarith)
  convert this using 2
  field_simp
  ring

/-! ### Quadratic forms -/

theorem qform_add_smul (A B : M5) (t : ℝ) (k : Fin 5 → ℤ) :
    qform (A + t • B) k = qform A k + t * qform B k := by
  simp only [qform, Matrix.add_apply, Matrix.smul_apply, smul_eq_mul]
  rw [Finset.mul_sum, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [Finset.mul_sum, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun j _ => ?_
  ring

theorem ofFn_vg (l : List Int) (hl : l.length = 5) :
    List.ofFn (fun i : Fin 5 => vg l i) = l := by
  apply List.ext_getElem (by simp [hl])
  intro n h1 h2
  simp only [List.getElem_ofFn, vg]
  exact List.getD_eq_getElem _ _ h2

/-- `λ n² + 2 μ n + s ≥ 0` on `ℤ` from its values at the two integers around `-μ / λ`. -/
theorem quad_nonneg (lam mu s n0 : Int) (hlam : 0 ≤ lam)
    (h : (if lam == 0 then mu == 0 && decide (0 ≤ s) else
      decide (lam * n0 ≤ -mu) && decide (-mu < lam * (n0 + 1)) &&
      decide (0 ≤ lam * n0 * n0 + 2 * mu * n0 + s) &&
      decide (0 ≤ lam * (n0 + 1) * (n0 + 1) + 2 * mu * (n0 + 1) + s)) = true) (n : Int) :
    0 ≤ lam * n * n + 2 * mu * n + s := by
  by_cases hl : lam = 0
  · simp only [hl, beq_self_eq_true, ↓reduceIte, Bool.and_eq_true, beq_iff_eq,
      decide_eq_true_eq] at h
    rw [hl, h.1]; simpa using h.2
  · have hl' : (lam == 0) = false := by simpa using hl
    simp only [hl', Bool.false_eq_true, ↓reduceIte, Bool.and_eq_true, decide_eq_true_eq] at h
    obtain ⟨⟨⟨h1, h2⟩, h3⟩, h4⟩ := h
    rcases le_or_gt n n0 with hn | hn
    · have hm : lam * n ≤ lam * n0 := mul_le_mul_of_nonneg_left hn hlam
      have hc : lam * (n + n0) + 2 * mu ≤ 0 := by linarith
      nlinarith [mul_nonneg (sub_nonneg.2 hn) (neg_nonneg.2 hc)]
    · have hn' : n0 + 1 ≤ n := hn
      have hm : lam * (n0 + 1) ≤ lam * n := mul_le_mul_of_nonneg_left hn' hlam
      have hc : 0 ≤ lam * (n + n0 + 1) + 2 * mu := by linarith
      nlinarith [mul_nonneg (sub_nonneg.2 hn') hc]

/-- The quadratic form of a direction `Y = [[λ w wᵀ, μ w], [μ wᵀ, s]]`. -/
theorem unbounded_Qv (Y : Mat) (lam mu s : Int) (w : List Int)
    (hab : ∀ a b : Fin 4, ent Y a b = lam * vg w a * vg w b)
    (ha5 : ∀ a : Fin 4, ent Y a 4 = mu * vg w a) (h5a : ∀ a : Fin 4, ent Y 4 a = mu * vg w a)
    (h55 : ent Y 4 4 = s) (k : Fin 5 → ℤ) :
    Qv Y k = lam * (∑ a : Fin 4, vg w a * k a.castSucc) * (∑ a : Fin 4, vg w a * k a.castSucc) +
      2 * mu * (∑ a : Fin 4, vg w a * k a.castSucc) * k (Fin.last 4) +
      s * k (Fin.last 4) * k (Fin.last 4) := by
  simp only [Qv, Fin.sum_univ_castSucc (n := 4), Fin.val_castSucc, Fin.val_last, hab, ha5, h5a,
    h55, Finset.sum_add_distrib]
  have e1 : ∑ i : Fin 4, ∑ j : Fin 4, k i.castSucc * (lam * vg w i * vg w j) * k j.castSucc =
      lam * (∑ a : Fin 4, vg w a * k a.castSucc) * (∑ a : Fin 4, vg w a * k a.castSucc) := by
    simp only [Finset.mul_sum, Finset.sum_mul]
    refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
    ring
  have e2 : ∑ i : Fin 4, k i.castSucc * (mu * vg w i) * k (Fin.last 4) =
      mu * (∑ a : Fin 4, vg w a * k a.castSucc) * k (Fin.last 4) := by
    simp only [Finset.mul_sum, Finset.sum_mul]
    refine Finset.sum_congr rfl fun i _ => ?_
    ring
  have e3 : ∑ j : Fin 4, k (Fin.last 4) * (mu * vg w j) * k j.castSucc =
      mu * (∑ a : Fin 4, vg w a * k a.castSucc) * k (Fin.last 4) := by
    simp only [Finset.mul_sum, Finset.sum_mul]
    refine Finset.sum_congr rfl fun i _ => ?_
    ring
  rw [e1, e2, e3]
  ring

end D4
