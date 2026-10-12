import D4Lean.Sound.Basic

/-!
# Membership in `R`

A bound certificate `C (cd J + cn E₄₄) = ∑ e_r m_r m_rᵀ` with `e_r ≥ 0` and `m₀` the unit vector
`eᵢ` gives `C (cd J[k] + cn k₄²) ≥ e₀ kᵢ²`, so `J[k] > 4` when `|kᵢ| > bᵢ` and `k₄ ∈ {0, 1}`.
The remaining points lie in the box, where `ptOK` was checked.
-/

open Matrix

namespace D4

/-- The value `J[k]` of a list matrix at an integral vector. -/
def Qv (J : Mat) (k : Fin 5 → ℤ) : ℤ := ∑ i : Fin 5, ∑ j : Fin 5, k i * ent J i j * k j

/-- The entry `(a, c)` of `∑ e_r m_r m_rᵀ`. -/
def FS (terms : List (Int × List Int)) (a c : Nat) : Int :=
  (terms.map fun t => t.1 * vg t.2 a * vg t.2 c).foldr (· + ·) 0

/-- `∑ e_r (m_r · k)²`. -/
def GS (terms : List (Int × List Int)) (k : Fin 5 → ℤ) : Int :=
  (terms.map fun t => t.1 * (∑ a : Fin 5, vg t.2 a * k a) ^ 2).foldr (· + ·) 0

theorem quad_FS (terms : List (Int × List Int)) (k : Fin 5 → ℤ) :
    ∑ a : Fin 5, ∑ c : Fin 5, k a * FS terms a c * k c = GS terms k := by
  induction terms with
  | nil => simp [FS, GS]
  | cons t ts ih =>
    have e1 : ∀ a c : Fin 5, FS (t :: ts) a c = t.1 * vg t.2 a * vg t.2 c + FS ts a c := by
      intro a c; simp [FS]
    have e2 : GS (t :: ts) k = t.1 * (∑ a : Fin 5, vg t.2 a * k a) ^ 2 + GS ts k := by
      simp [GS]
    simp only [e1, e2, ← ih, mul_add, add_mul, Finset.sum_add_distrib]
    congr 1
    simp only [Fin.sum_univ_five]
    ring

theorem GS_nonneg (terms : List (Int × List Int)) (h : ∀ t ∈ terms, 0 ≤ t.1) (k : Fin 5 → ℤ) :
    0 ≤ GS terms k := by
  induction terms with
  | nil => simp [GS]
  | cons t ts ih =>
    have e2 : GS (t :: ts) k = t.1 * (∑ a : Fin 5, vg t.2 a * k a) ^ 2 + GS ts k := by
      simp [GS]
    rw [e2]
    have := h t (by simp)
    have := ih (fun s hs => h s (by simp [hs]))
    positivity

theorem bound_sound (J : Mat) (cd cn C : Int) (i : Fin 4) (b : Nat)
    (terms : List (Int × List Int)) (h : boundOK J cd cn C i b terms = true)
    (hcd : 0 < cd) (hcn : 0 ≤ cn) (hC : 0 < C) (k : Fin 5 → ℤ) (h5 : k 4 = 0 ∨ k 4 = 1)
    (hk : (b : ℤ) < |k i.castSucc|) : 4 < Qv J k := by
  cases terms with
  | nil => simp [boundOK] at h
  | cons t rest =>
    obtain ⟨e, m⟩ := t
    simp only [boundOK, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq, List.all_eq_true,
      allN_iff] at h
    obtain ⟨⟨⟨⟨_, hm⟩, hb⟩, hpos⟩, hid⟩ := h
    -- the identity in matrix form, contracted with k
    have hsum : C * (cd * Qv J k + cn * k 4 ^ 2) = GS ((e, m) :: rest) k := by
      rw [← quad_FS]
      have : ∀ a c : Fin 5, FS ((e, m) :: rest) a c =
          C * (cd * ent J a c + (if (a : Nat) = 4 ∧ (c : Nat) = 4 then cn else 0)) := by
        intro a c
        have := hid a a.2 c c.2
        rw [FS, ← this]
      simp only [this, Qv, Fin.sum_univ_five]
      norm_num
      ring
    have hhead : GS ((e, m) :: rest) k ≥ e * (k i.castSucc) ^ 2 := by
      have e2 : GS ((e, m) :: rest) k = e * (∑ a : Fin 5, vg m a * k a) ^ 2 + GS rest k := by
        simp [GS]
      have hmk : ∑ a : Fin 5, vg m a * k a = k i.castSucc := by
        have hm' : ∀ a : Fin 5, vg m a = if (a : Nat) = i then 1 else 0 := by
          intro a; have := hm a a.2; simpa using this
        simp only [hm']
        fin_cases i <;> simp [Fin.sum_univ_five]
      have := GS_nonneg rest (fun s hs => (hpos s (by simp [hs])).1) k
      rw [e2, hmk]
      linarith
    have hki : ((b : ℤ) + 1) ^ 2 ≤ (k i.castSucc) ^ 2 := by
      have h0 : (0 : ℤ) ≤ (b : ℤ) + 1 := by positivity
      have h1 : (b : ℤ) + 1 ≤ |k i.castSucc| := by omega
      calc ((b : ℤ) + 1) ^ 2 ≤ |k i.castSucc| ^ 2 := by gcongr
        _ = (k i.castSucc) ^ 2 := sq_abs _
    have he : 0 ≤ e := (hpos (e, m) (by simp)).1
    have h55 : k 4 ^ 2 ≤ 1 := by rcases h5 with h5 | h5 <;> simp [h5]
    have : C * (4 * cd + cn) < C * (cd * Qv J k + cn * k 4 ^ 2) := by
      calc C * (4 * cd + cn) < e * ((b : ℤ) + 1) * ((b : ℤ) + 1) := hb
        _ = e * ((b : ℤ) + 1) ^ 2 := by ring
        _ ≤ e * (k i.castSucc) ^ 2 := by gcongr
        _ ≤ _ := by rw [hsum]; exact hhead
    have h2 : C * cn * k 4 ^ 2 ≤ C * cn := by
      have := mul_nonneg hC.le hcn
      nlinarith
    have h3 : C * cd * 4 < C * cd * Qv J k := by nlinarith
    exact lt_of_mul_lt_mul_left h3 (by positivity)

theorem isZero_ofFn (k : Fin 5 → ℤ) (h : isZero (List.ofFn k) = true) : k = 0 := by
  simp only [isZero, List.all_eq_true, List.mem_ofFn, beq_iff_eq] at h
  funext i
  exact h _ ⟨i, rfl⟩

theorem negV_ofFn (k : Fin 5 → ℤ) : negV (List.ofFn k) = List.ofFn (-k) := by
  simp only [negV, List.map_ofFn]
  rfl

theorem Qv_neg (J : Mat) (k : Fin 5 → ℤ) : Qv J (-k) = Qv J k := by
  simp [Qv]

theorem memOK_sound (J : Mat) (mins : List (List Int)) (cd cn C : Int) (bs : List Nat)
    (certs : List (List (Int × List Int))) (hJ : shape J 5 = true)
    (h : memOK J mins cd cn C bs certs = true) (k : Fin 5 → ℤ) (h5 : k 4 = 0 ∨ k 4 = 1)
    (hk : k ≠ 0) :
    4 < Qv J k ∨ (Qv J k = 4 ∧ (List.ofFn k ∈ mins ∨ List.ofFn (-k) ∈ mins)) := by
  simp only [memOK, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq, allN_iff] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨hcd, hcn⟩, hC⟩, hbl⟩, _⟩, hb⟩, _⟩, hbox⟩ := h
  by_cases hout : ∃ i : Fin 4, ((bs.getD i 0 : Nat) : ℤ) < |k i.castSucc|
  · obtain ⟨i, hi⟩ := hout
    left
    exact bound_sound J cd cn C i _ _ (hb i i.2) hcd hcn hC k h5 hi
  · push Not at hout
    have hpt : ptOK J mins (List.ofFn k) = true := by
      apply boxAll_sound _ _ hbox
      match bs, hbl with
      | [b0, b1, b2, b3], _ =>
        have h0 := abs_le.mp (hout 0)
        have h1 := abs_le.mp (hout 1)
        have h2 := abs_le.mp (hout 2)
        have h3 := abs_le.mp (hout 3)
        simp at h0 h1 h2 h3
        simp [boxOf]
        refine ⟨⟨by omega, by omega⟩, ⟨by omega, by omega⟩, ⟨by omega, by omega⟩,
          ⟨by omega, by omega⟩, ?_⟩
        rcases h5 with h5 | h5 <;> simp_all
    simp only [ptOK, Bool.or_eq_true, decide_eq_true_eq, Bool.and_eq_true, beq_iff_eq,
      List.contains_iff_mem, negV_ofFn, qf_ofFn J hJ] at hpt
    rcases hpt with (hz | hgt) | ⟨heq, hm⟩
    · exact absurd (isZero_ofFn k hz) hk
    · left; exact hgt
    · right; exact ⟨heq, hm⟩

theorem minOK_sound (J : Mat) (hJ : shape J 5 = true) (k : Fin 5 → ℤ)
    (h : minOK J (List.ofFn k) = true) : Qv J k = 4 := by
  simp only [minOK, Bool.and_eq_true, beq_iff_eq, qf_ofFn J hJ] at h
  exact h.2

theorem mstar_norm (k : Fin 5 → ℤ) (hk : k ∈ Mstar) :
    ∃ k' : Fin 5 → ℤ, (k' = k ∨ k' = -k) ∧ (k' 4 = 0 ∨ k' 4 = 1) ∧ k' ≠ 0 := by
  rcases hk with ⟨h5, hk⟩ | h5 | h5
  · exact ⟨k, Or.inl rfl, Or.inl h5, hk⟩
  · refine ⟨k, Or.inl rfl, Or.inr h5, fun h => ?_⟩
    rw [h] at h5; simp at h5
  · refine ⟨-k, Or.inr rfl, Or.inr (by simp [h5]), fun h => ?_⟩
    have := congrFun h 4
    simp [h5] at this

theorem qform_Qv (J : Mat) (k : Fin 5 → ℤ) : qform (toR J) k = (Qv J k : ℝ) := by
  rw [qform_toR]; rfl

theorem symm_sound (A : Mat) (h : symm A = true) : (toR A).IsSymm := by
  simp only [symm, allN_iff, beq_iff_eq] at h
  apply Matrix.IsSymm.ext
  intro i j
  simp only [toR, toZ, Matrix.map_apply, Matrix.of_apply]
  rw [h j j.2 i i.2]

/-- Membership in `R` and the minimal vectors of a vertex. -/
theorem vertex_mem (J : Mat) (mins : List (List Int)) (cd cn C : Int) (bs : List Nat)
    (certs : List (List (Int × List Int))) (hJ : shape J 5 = true) (hs : symm J = true)
    (h : memOK J mins cd cn C bs certs = true) :
    toR J ∈ Rset ∧
    ∀ k ∈ Mstar, qform (toR J) k = 4 ↔ (List.ofFn k ∈ mins ∨ List.ofFn (-k) ∈ mins) := by
  have hmins : ∀ x ∈ mins, minOK J x = true := by
    have := h
    simp only [memOK, Bool.and_eq_true, List.all_eq_true] at this
    exact this.1.2
  have key : ∀ k ∈ Mstar, 4 < Qv J k ∨ (Qv J k = 4 ∧ (List.ofFn k ∈ mins ∨ List.ofFn (-k) ∈ mins)) := by
    intro k hk
    obtain ⟨k', hk', h5, h0⟩ := mstar_norm k hk
    have := memOK_sound J mins cd cn C bs certs hJ h k' h5 h0
    rcases hk' with rfl | rfl
    · exact this
    · rw [Qv_neg, neg_neg] at this
      rcases this with h1 | ⟨h1, h2⟩
      · exact Or.inl h1
      · exact Or.inr ⟨h1, h2.symm⟩
  refine ⟨⟨symm_sound J hs, fun k hk => ?_⟩, fun k hk => ?_⟩
  · rw [qform_Qv]
    rcases key k hk with h1 | ⟨h1, _⟩
    · exact_mod_cast h1.le
    · exact_mod_cast h1.ge
  · rw [qform_Qv]
    constructor
    · intro h4
      have h4' : Qv J k = 4 := by exact_mod_cast h4
      rcases key k hk with h1 | ⟨_, h2⟩
      · omega
      · exact h2
    · intro hm
      rcases hm with hm | hm
      · have := minOK_sound J hJ k (hmins _ hm)
        exact_mod_cast this
      · have := minOK_sound J hJ (-k) (hmins _ hm)
        rw [Qv_neg] at this
        exact_mod_cast this

end D4
