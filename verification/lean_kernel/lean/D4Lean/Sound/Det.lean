import D4Lean.Sound.Basic

/-!
# Determinants by elimination

If `E` is lower triangular with nonzero diagonal and `E A` is upper triangular, then
`det E · det A = ∏ (E A)ᵢᵢ`, and the leading `4 × 4` block of `E A` is `E₄ A₄`.
-/

open Matrix

namespace D4

theorem toZ_apply (A : Mat) (i j : Fin 5) : toZ A i j = ent A i j := rfl

theorem toZ_mul_apply (E A : Mat) (i j : Fin 5) : (toZ E * toZ A) i j = mulE 5 E A i j := by
  rw [Matrix.mul_apply, mulE, sumN_eq]
  rfl

theorem detOK_sound (A E : Mat) (d dq : Int) (h : detOK A E d dq = true) :
    (toZ A).det = d ∧ ((toZ A).submatrix Fin.castSucc Fin.castSucc).det = dq := by
  simp only [detOK, Bool.and_eq_true, allN_iff, Bool.or_eq_true, decide_eq_true_eq, beq_iff_eq,
    bne_iff_ne, ne_eq] at h
  obtain ⟨⟨⟨⟨hL, hU⟩, hE0⟩, hd⟩, hdq⟩ := h
  have hLow : ∀ i j : Fin 5, i < j → toZ E i j = 0 := by
    intro i j hij
    rcases hL i i.2 j j.2 with h1 | h1
    · exact absurd h1 (by simpa using hij)
    · exact h1
  have hUp : ∀ i j : Fin 5, j < i → (toZ E * toZ A) i j = 0 := by
    intro i j hij
    rw [toZ_mul_apply]
    rcases hU i i.2 j j.2 with h1 | h1
    · exact absurd h1 (by simpa using hij)
    · exact h1
  have hEdet : (toZ E).det = ∏ i : Fin 5, ent E i i := by
    rw [Matrix.det_of_isLowerTriangular _ (fun i j hij => hLow i j (by simpa using hij))]
    rfl
  have hEAdet : (toZ E * toZ A).det = ∏ i : Fin 5, mulE 5 E A i i := by
    rw [Matrix.det_of_isUpperTriangular (fun i j hij => hUp i j hij)]
    simp only [toZ_mul_apply]
  simp only [prodN_eq] at hE0 hd hdq
  constructor
  · have h1 : (toZ E).det * (toZ A).det = d * (toZ E).det := by
      rw [← Matrix.det_mul, hEAdet, hd, hEdet]
    rw [hEdet] at h1
    have h2 : (∏ i : Fin 5, ent E i i) * ((toZ A).det - d) = 0 := by rw [mul_sub]; linarith
    rcases mul_eq_zero.mp h2 with h3 | h3
    · exact absurd h3 hE0
    · linarith
  · set S : Fin 4 → Fin 5 := Fin.castSucc
    have hsub : (toZ E * toZ A).submatrix S S = (toZ E).submatrix S S * (toZ A).submatrix S S := by
      ext i j
      simp only [Matrix.submatrix_apply, Matrix.mul_apply, S]
      rw [Fin.sum_univ_castSucc, hLow _ _ (Fin.castSucc_lt_last i), zero_mul, add_zero]
    have hE5 : ((toZ E).submatrix S S).det = ∏ i : Fin 4, ent E i i := by
      rw [Matrix.det_of_isLowerTriangular ((toZ E).submatrix S S) (fun i j hij => hLow _ _ (by
        simpa [S, Fin.castSucc_lt_castSucc_iff] using hij))]
      rfl
    have hEA5 : ((toZ E * toZ A).submatrix S S).det = ∏ i : Fin 4, mulE 5 E A i i := by
      rw [Matrix.det_of_isUpperTriangular (M := (toZ E * toZ A).submatrix S S)
        (fun i j hij => hUp _ _ (by simpa [S, Fin.castSucc_lt_castSucc_iff] using hij))]
      simp only [Matrix.submatrix_apply, toZ_mul_apply, S, Fin.val_castSucc]
    have hE50 : (∏ i : Fin 4, ent E i i) ≠ 0 := by
      intro h0
      apply hE0
      rw [Fin.prod_univ_castSucc]
      simp only [Fin.val_castSucc]
      rw [h0, zero_mul]
    have h1 : ((toZ E).submatrix S S).det * ((toZ A).submatrix S S).det =
        dq * ((toZ E).submatrix S S).det := by
      rw [← Matrix.det_mul, ← hsub, hEA5, hE5]
      simpa using hdq
    rw [hE5] at h1
    have h2 : (∏ i : Fin 4, ent E i i) * (((toZ A).submatrix S S).det - dq) = 0 := by
      rw [mul_sub]; linarith
    rcases mul_eq_zero.mp h2 with h3 | h3
    · exact absurd h3 hE50
    · linarith

/-- Real determinants of an integral matrix. -/
theorem det_map_cast {n : Type} [Fintype n] [DecidableEq n] (M : Matrix n n ℤ) :
    (M.map (Int.cast : ℤ → ℝ)).det = (M.det : ℝ) := by
  have := RingHom.map_det (Int.castRingHom ℝ) M
  rw [RingHom.mapMatrix_apply, Int.coe_castRingHom] at this
  exact this.symm

theorem det_toR (A : Mat) : (toR A).det = ((toZ A).det : ℝ) := det_map_cast _

theorem detQ_toR (A : Mat) :
    (Qb (toR A)).det = (((toZ A).submatrix Fin.castSucc Fin.castSucc).det : ℝ) :=
  det_map_cast ((toZ A).submatrix Fin.castSucc Fin.castSucc)

end D4
