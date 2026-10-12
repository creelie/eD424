import D4Lean.Sound.Edge

/-!
# Soundness of the vertex and edge checks

`Vtx.ok v = true` gives `VFacts v`, and `Edge.ok v.J w.J e = true` gives `EFacts v w e`.
-/

open Matrix

namespace D4

/-- The end point of a bounded edge and the minimal vector that leaves `R` after it. -/
theorem boundedOK_sound (J Y Jto T Tinv : Mat) (tn td : Int) (kmax : List Int)
    (hJ : shape J 5 = true) (h : boundedOK J Y Jto T Tinv tn td kmax = true) :
    (toR Y).IsSymm ∧ 0 < tn ∧ 0 < td ∧ toZ T ∈ Gamma ∧
    toR J + ((tn : ℝ) / td) • toR Y = ((toZ T).map Int.cast)ᵀ * toR Jto * (toZ T).map Int.cast ∧
    ∃ k ∈ Mstar, ∀ t : ℝ, (tn : ℝ) / td < t → qform (toR J + t • toR Y) k < 4 := by
  simp only [boundedOK, Bool.and_eq_true] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨hYs, hYsym⟩, -⟩, -⟩, htn⟩, htd⟩, hcong⟩, hinv⟩, hrow⟩, h55⟩, hkl⟩,
    hk5⟩, hkz⟩, hkq⟩, hkneg⟩ := h
  simp only [decide_eq_true_eq] at htn htd hkneg
  simp only [allN_iff, beq_iff_eq] at hcong hinv hrow
  simp only [Bool.or_eq_true, beq_iff_eq] at h55 hk5
  simp only [beq_iff_eq] at hkl hkq
  have htdR : (0 : ℝ) < td := by exact_mod_cast htd
  have htd0 : (td : ℝ) ≠ 0 := htdR.ne'
  refine ⟨symm_sound Y hYsym, htn, htd, ?_, ?_, ?_⟩
  · -- `T ∈ Γ`
    have hTT : toZ T * toZ Tinv = 1 := by
      ext i j
      rw [toZ_mul_apply, hinv i i.2 j j.2, Matrix.one_apply]
      simp [Fin.ext_iff]
    have hdet := congrArg Matrix.det hTT
    rw [Matrix.det_mul, Matrix.det_one] at hdet
    refine ⟨IsUnit.of_mul_eq_one _ hdet, fun j => ?_, ?_⟩
    · simpa [toZ_apply] using hrow j j.2
    · simpa [toZ_apply] using h55
  · -- the end point
    ext i j
    have h2 : (((td * ent J i j + tn * ent Y i j : ℤ)) : ℝ) = ((td * congE T Jto i j : ℤ) : ℝ) := by
      rw [hcong i i.2 j j.2]
    simp only [congE, sumN_eq] at h2
    push_cast at h2
    simp only [Matrix.add_apply, Matrix.smul_apply, smul_eq_mul, Matrix.mul_apply,
      Matrix.transpose_apply, Matrix.map_apply, toR, toZ, Matrix.of_apply, Finset.sum_mul]
    rw [Finset.sum_comm]
    field_simp
    linarith
  · -- the minimal vector `kmax`
    set k : Fin 5 → ℤ := fun i => vg kmax i with hk
    have hofn : List.ofFn k = kmax := ofFn_vg kmax hkl
    have hk0 : k ≠ 0 := by
      intro h0
      have : isZero kmax = true := by rw [← hofn, h0]; decide
      simp [this] at hkz
    have hkM : k ∈ Mstar := by
      rcases hk5 with h5 | h5
      · exact Or.inl ⟨by simpa [hk] using h5, hk0⟩
      · exact Or.inr (Or.inl (by simpa [hk] using h5))
    have hQJ : Qv J k = qf J kmax := by rw [← hofn, qf_ofFn J hJ k]; rfl
    have hQY : Qv Y k = qf Y kmax := by rw [← hofn, qf_ofFn Y hYs k]; rfl
    refine ⟨k, hkM, fun t ht => ?_⟩
    rw [qform_add_smul, qform_Qv, qform_Qv, hQJ, hQY]
    have hq : (td : ℝ) * qf J kmax + tn * qf Y kmax = 4 * td := by exact_mod_cast hkq
    have hneg : (qf Y kmax : ℝ) < 0 := by exact_mod_cast hkneg
    rw [div_lt_iff₀ htdR] at ht
    have hp : 0 < (t * td - tn) * (-(qf Y kmax : ℝ)) := mul_pos (by linarith) (by linarith)
    by_contra hc
    push Not at hc
    have := mul_le_mul_of_nonneg_left hc htdR.le
    nlinarith

/-- The direction of an unbounded edge has `Y[k] ≥ 0` on `M*`. -/
theorem unboundedOK_sound (Y : Mat) (lam mu s n0 : Int) (w : List Int)
    (h : unboundedOK Y lam mu s n0 w = true) :
    (toR Y).IsSymm ∧ ∀ k ∈ Mstar, 0 ≤ qform (toR Y) k := by
  simp only [unboundedOK, Bool.and_eq_true] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨-, hYsym⟩, -⟩, hab⟩, ha5⟩, h55⟩, hlam⟩, hq⟩ := h
  simp only [allN_iff, beq_iff_eq] at hab ha5
  simp only [beq_iff_eq] at h55
  simp only [decide_eq_true_eq] at hlam
  have hsy : ∀ i < 5, ∀ j < 5, ent Y i j = ent Y j i := by
    simpa [symm, allN_iff] using hYsym
  refine ⟨symm_sound Y hYsym, fun k hk => ?_⟩
  rw [qform_Qv, unbounded_Qv Y lam mu s w (fun a b => hab a a.2 b b.2) (fun a => ha5 a a.2)
    (fun a => by rw [hsy 4 (by omega) a (by omega)]; exact ha5 a a.2) h55 k]
  set m := ∑ a : Fin 4, vg w a * k a.castSucc
  have hk5 : k (Fin.last 4) = k 4 := rfl
  rw [hk5]
  have hquad := quad_nonneg lam mu s n0 hlam hq
  have hint : 0 ≤ lam * m * m + 2 * mu * m * k 4 + s * k 4 * k 4 := by
    rcases hk with ⟨h5, -⟩ | h5 | h5 <;> rw [h5]
    · nlinarith [mul_nonneg hlam (mul_self_nonneg m)]
    · linarith [hquad m]
    · have := hquad (-m)
      nlinarith
  exact_mod_cast hint

/-- Soundness of the check of a vertex. -/
theorem vertex_sound (v : Vtx) (h : v.ok = true) : VFacts v := by
  simp only [Vtx.ok, Bool.and_eq_true] at h
  obtain ⟨⟨⟨⟨⟨hs, hsy⟩, hm⟩, hd⟩, hb⟩, heq⟩ := h
  have hmem := vertex_mem v.J v.mins v.cd v.cn v.C v.bs v.certs hs hsy hm
  have hdet := detOK_sound v.J v.E v.dJ v.dQ hd
  refine ⟨hmem.1, hmem.2, ?_, ?_, ?_, ?_⟩
  · rw [det_toR, hdet.1]
  · rw [detQ_toR, hdet.2]
  · simpa using hb
  · rw [beq_iff_eq] at heq
    rw [← heq]
    simp

/-- Soundness of the check of an edge `e` at the vertex `v`, ending at `w` if it is bounded. -/
theorem edge_sound (v w : Vtx) (e : Edge) (hJ : shape v.J 5 = true)
    (h : Edge.ok v.J w.J e = true) : EFacts v w e := by
  simp only [Edge.ok, Bool.and_eq_true] at h
  obtain ⟨⟨hs, hsg⟩, hb⟩ := h
  simp only [signOK, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq, List.all_eq_true] at hsg
  obtain ⟨⟨⟨⟨⟨⟨hwd, hpJ⟩, hpQ⟩, hhead⟩, -⟩, hpts⟩, hpcs⟩ := hsg
  have hg : (gOf e.pQ).length ≤ 5 := by rw [length_gOf]; omega
  have hdet := samples_sound v.J e.Y e.pJ e.pQ e.sd e.sn e.Es hs hpJ hpQ
  have hgood : ∀ t : ℝ, Good e.pJ (gOf e.pQ) e.wd e.eqs t →
      (0 < (seg v e t).det ∨ 256 ≤ (Qb (seg v e t)).det) ∧
      ((seg v e t).det = 0 → (Qb (seg v e t)).det = 256 → ∃ u ∈ e.eqs, t = (u : ℝ) / e.wd) := by
    intro t ⟨h1, h2⟩
    rw [evalL_gOf] at h1 h2
    simp only [seg, (hdet t).1, (hdet t).2]
    refine ⟨?_, fun a b => h2 a (by rw [b]; ring)⟩
    rcases h1 with h1 | h1
    · exact Or.inl h1
    · exact Or.inr (by linarith)
  have hcov := pieces_good e.pJ (gOf e.pQ) e.wd e.eqs hwd hpJ hg e.us e.ps hpcs hpts 0
  cases hbd : e.bounded
  · -- an unbounded edge
    simp only [hbd, Bool.false_eq_true, ↓reduceIte, Bool.and_eq_true] at hb
    obtain ⟨hu, htail⟩ := hb
    have hun := unboundedOK_sound e.Y e.lam e.mu e.s e.n0 e.w hu
    cases hl : e.us.getLast? with
    | none => simp [tailOK, hl] at htail
    | some ul =>
      simp only [tailOK, hl] at htail
      have key : ∀ t : ℝ, 0 ≤ t → Good e.pJ (gOf e.pQ) e.wd e.eqs t := by
        intro t ht
        by_cases htl : t ≤ (ul : ℝ) / e.wd
        · exact hcov ul hhead hl t (by simpa using ht) htl
        · have ht' : (ul : ℝ) / e.wd < t := not_le.mp htl
          cases hm : e.tailJ
          · simp only [hm, Bool.false_eq_true, ↓reduceIte] at htail
            exact good_of_pos _ _ _ _ false t
              (by simpa using tail_aux _ 4 e.wd ul hwd (by omega) htail t ht')
          · simp only [hm, ↓reduceIte] at htail
            exact good_of_pos _ _ _ _ true t
              (by simpa using tail_aux _ 5 e.wd ul hwd (by omega) htail t ht')
      exact {
        symm := hun.1
        sign := fun t ht _ => (hgood t (key t ht)).1
        eq := fun t ht _ => (hgood t (key t ht)).2
        bounded := fun h => absurd h (by simp [hbd])
        unbounded := fun _ => hun.2 }
  · -- a bounded edge
    simp only [hbd, ↓reduceIte, Bool.and_eq_true, beq_iff_eq] at hb
    obtain ⟨⟨hbo, hlast⟩, hdiv⟩ := hb
    obtain ⟨hsym, htn, htd, hT, hM, hk⟩ := boundedOK_sound v.J e.Y w.J e.T e.Tinv e.tn e.td e.kmax hJ hbo
    have htd0 : (e.td : ℝ) ≠ 0 := by exact_mod_cast htd.ne'
    have hwd0 : (e.wd : ℝ) ≠ 0 := by exact_mod_cast hwd.ne'
    have hul : (((e.tn * e.wd / e.td : ℤ) : ℝ)) / e.wd = (e.tn : ℝ) / e.td := by
      rw [Int.cast_div (Int.dvd_of_emod_eq_zero hdiv) htd0]
      push_cast
      field_simp
    have key : ∀ t : ℝ, 0 ≤ t → t ≤ (e.tn : ℝ) / e.td → Good e.pJ (gOf e.pQ) e.wd e.eqs t :=
      fun t ht htl => hcov _ hhead hlast t (by simpa using ht) (by rw [hul]; exact htl)
    exact {
      symm := hsym
      sign := fun t ht htl => (hgood t (key t ht (htl hbd))).1
      eq := fun t ht htl => (hgood t (key t ht (htl hbd))).2
      bounded := fun _ => ⟨div_pos (by exact_mod_cast htn) (by exact_mod_cast htd),
        ⟨toZ e.T, hT, by simpa [seg] using hM⟩, by simpa [seg] using hk⟩
      unbounded := fun h => absurd h (by simp [hbd]) }

end D4
