import Mathlib

/-!
# Generators of a pointed polyhedral cone from its hyperplane flats

Let `a₁, …, a_m` be vectors in `ℝ^d` that span the dual space, so that the cone
`K = {x | aᵢ ⬝ x ≥ 0 for all i}` is pointed. Suppose that for every set `s` of `d - 1` indices
there is a nonzero `y` with `aᵢ ⬝ y = 0` for `i ∈ s` such that either `y` takes both signs on the
`aᵢ`, or `y` or `-y` lies in a set `Y ⊆ K`. Then `K` is the cone spanned by `Y`.

The proof moves a point `x ∈ K` inside `K` until `d - 1` independent constraints are tight, keeping
a list `S` of independent tight constraints: a tight constraint outside the span of `S` is added
to `S`; otherwise the kernel of `S` has dimension at least two and contains a direction `z ≠ 0`
orthogonal to `∑ aᵢ`, along which `x` splits into two points of `K` with more tight constraints.
When `S` has `d - 1` elements, its kernel is the line through `x` and through the `y` given for
`S`, and `x` is a nonnegative multiple of an element of `Y`.
-/

open Matrix Module

namespace D4
namespace Cone

variable {d m : ℕ} (a : Fin m → Fin d → ℝ)

/-- The rows `a (S i)` as a matrix. -/
def subA {r : ℕ} (S : Fin r → Fin m) : Matrix (Fin r) (Fin d) ℝ := Matrix.of fun i => a (S i)

lemma mem_ker_subA {r : ℕ} (S : Fin r → Fin m) (z : Fin d → ℝ) :
    z ∈ LinearMap.ker (subA a S).mulVecLin ↔ ∀ i, a (S i) ⬝ᵥ z = 0 := by
  simp only [LinearMap.mem_ker, Matrix.mulVecLin_apply, funext_iff, Matrix.mulVec, subA,
    Matrix.of_apply, Pi.zero_apply]

lemma finrank_ker_subA {r : ℕ} (S : Fin r → Fin m)
    (h : LinearIndependent ℝ (fun i => a (S i))) :
    finrank ℝ (LinearMap.ker (subA a S).mulVecLin) = d - r := by
  have h1 := LinearMap.finrank_range_add_finrank_ker (subA a S).mulVecLin
  have h2 : (subA a S).rank = r := by
    have := LinearIndependent.rank_matrix (M := subA a S) h
    simpa using this
  rw [Matrix.rank] at h2
  rw [h2, Module.finrank_fin_fun] at h1
  omega

/-- The constraints that are tight at `x`. -/
noncomputable def tight (x : Fin d → ℝ) : Finset (Fin m) :=
  Finset.univ.filter fun i => a i ⬝ᵥ x = 0

lemma mem_tight {x : Fin d → ℝ} {i : Fin m} : i ∈ tight a x ↔ a i ⬝ᵥ x = 0 := by
  simp [tight]

/-- Moving from `x ∈ K` along `z`, which is orthogonal to the constraints tight at `x` and
negative on some constraint, until a new constraint becomes tight. -/
lemma step (x z : Fin d → ℝ) (hx : ∀ i, 0 ≤ a i ⬝ᵥ x)
    (hz : ∀ i, a i ⬝ᵥ x = 0 → a i ⬝ᵥ z = 0) (hq : ∃ q, a q ⬝ᵥ z < 0) :
    ∃ t : ℝ, 0 < t ∧ (∀ i, 0 ≤ a i ⬝ᵥ (x + t • z)) ∧ tight a x ⊂ tight a (x + t • z) := by
  classical
  set N := Finset.univ.filter fun i => a i ⬝ᵥ z < 0 with hNdef
  have hN : N.Nonempty := by
    obtain ⟨q, hq⟩ := hq
    exact ⟨q, by simp [N, hq]⟩
  obtain ⟨i0, hi0, hmin⟩ := N.exists_min_image (fun i => (a i ⬝ᵥ x) / -(a i ⬝ᵥ z)) hN
  have hi0' : a i0 ⬝ᵥ z < 0 := by simpa [N] using hi0
  have hx0 : 0 < a i0 ⬝ᵥ x := by
    rcases (hx i0).lt_or_eq with h | h
    · exact h
    · exact absurd (hz i0 h.symm) (ne_of_lt hi0')
  have hpos0 : 0 < -(a i0 ⬝ᵥ z) := neg_pos.mpr hi0'
  set t := (a i0 ⬝ᵥ x) / -(a i0 ⬝ᵥ z) with ht
  have htpos : 0 < t := div_pos hx0 hpos0
  refine ⟨t, htpos, ?_, ?_⟩
  · intro i
    rw [dotProduct_add, dotProduct_smul, smul_eq_mul]
    by_cases hi : a i ⬝ᵥ z < 0
    · have h1 := hmin i (by simp [N, hi])
      have hpos : 0 < -(a i ⬝ᵥ z) := neg_pos.mpr hi
      rw [le_div_iff₀ hpos] at h1
      nlinarith
    · push Not at hi
      have := hx i
      nlinarith [mul_nonneg htpos.le hi]
  · have hsub : tight a x ⊆ tight a (x + t • z) := by
      intro i hi
      rw [mem_tight] at hi ⊢
      rw [dotProduct_add, dotProduct_smul, hi, hz i hi, smul_zero, add_zero]
    rw [Finset.ssubset_iff_of_subset hsub]
    refine ⟨i0, ?_, ?_⟩
    · rw [mem_tight, dotProduct_add, dotProduct_smul, smul_eq_mul, ht]
      have hne : a i0 ⬝ᵥ z ≠ 0 := hi0'.ne
      field_simp
      ring
    · rw [mem_tight]
      exact hx0.ne'

lemma dot_eq_zero_of_mem_span {r : ℕ} (S : Fin r → Fin m) (z v : Fin d → ℝ)
    (hz : ∀ j, a (S j) ⬝ᵥ z = 0) (hv : v ∈ Submodule.span ℝ (Set.range fun j => a (S j))) :
    v ⬝ᵥ z = 0 := by
  induction hv using Submodule.span_induction with
  | mem w hw =>
    obtain ⟨j, rfl⟩ := hw
    exact hz j
  | zero => simp
  | add u w _ _ hu hw => rw [add_dotProduct, hu, hw, add_zero]
  | smul c u _ hu => rw [smul_dotProduct, hu, smul_zero]

lemma sum_dot (z : Fin d → ℝ) : (∑ i, a i) ⬝ᵥ z = ∑ i, a i ⬝ᵥ z := by
  simp only [dotProduct, Finset.sum_apply, Finset.sum_mul]
  rw [Finset.sum_comm]

section main

variable (Y : Set (Fin d → ℝ))
  (hpt : ∀ z, (∀ i, a i ⬝ᵥ z = 0) → z = 0)
  (hY : ∀ y ∈ Y, ∀ i, 0 ≤ a i ⬝ᵥ y)
  (hcert : ∀ s : Finset (Fin m), s.card + 1 = d → ∃ y : Fin d → ℝ, y ≠ 0 ∧
    (∀ i ∈ s, a i ⬝ᵥ y = 0) ∧ ((∃ p q, 0 < a p ⬝ᵥ y ∧ a q ⬝ᵥ y < 0) ∨ y ∈ Y ∨ -y ∈ Y))
include hpt hY hcert

lemma mem_span_aux : ∀ n (x : Fin d → ℝ) (r : ℕ) (S : Fin r → Fin m), (∀ i, 0 ≤ a i ⬝ᵥ x) →
    LinearIndependent ℝ (fun i => a (S i)) → (∀ i, a (S i) ⬝ᵥ x = 0) → r + 1 ≤ d →
    m - (tight a x).card + (d - r) = n → x ∈ Submodule.span NNReal Y := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
  intro x r S hx hS hSx hr hn
  classical
  -- `x` is a nonnegative multiple of an element of `Y`
  have key : ∀ w ∈ Y, ∀ c : ℝ, c • w = x → x ∈ Submodule.span NNReal Y := by
    intro w hw c hc
    by_cases hc0 : 0 ≤ c
    · rw [← hc]
      have : (⟨c, hc0⟩ : NNReal) • w ∈ Submodule.span NNReal Y :=
        Submodule.smul_mem _ _ (Submodule.subset_span hw)
      simpa [NNReal.smul_def] using this
    · push Not at hc0
      have hw0 : ∀ i, a i ⬝ᵥ w = 0 := by
        intro i
        have h1 := hY w hw i
        have h2 := hx i
        rw [← hc, dotProduct_smul, smul_eq_mul] at h2
        nlinarith
      rw [← hc, hpt w hw0, smul_zero]
      exact Submodule.zero_mem _
  by_cases hrd : r + 1 = d
  · -- `d - 1` independent tight constraints: the kernel is a line
    have hinj : Function.Injective S := by
      intro i j hij
      exact hS.injective (by simp only [hij])
    set s := Finset.univ.image S with hsdef
    have hs : s.card = r := by
      rw [hsdef, Finset.card_image_of_injective _ hinj, Finset.card_univ, Fintype.card_fin]
    obtain ⟨y, hy0, hys, hkind⟩ := hcert s (by omega)
    have hfr := finrank_ker_subA a S hS
    rw [show d - r = 1 by omega] at hfr
    have hyK : y ∈ LinearMap.ker (subA a S).mulVecLin :=
      (mem_ker_subA a S y).2 fun i => hys _ (by simp [hsdef])
    have hxK : x ∈ LinearMap.ker (subA a S).mulVecLin := (mem_ker_subA a S x).2 hSx
    have hyne : (⟨y, hyK⟩ : LinearMap.ker (subA a S).mulVecLin) ≠ 0 := by
      intro h
      exact hy0 (congrArg Subtype.val h)
    obtain ⟨c, hc⟩ := (finrank_eq_one_iff_of_nonzero' _ hyne).1 hfr ⟨x, hxK⟩
    have hcx : c • y = x := congrArg Subtype.val hc
    rcases hkind with ⟨p, q, hp, hq⟩ | hyY | hyY
    · rcases lt_trichotomy c 0 with hc0 | hc0 | hc0
      · have := hx p
        rw [← hcx, dotProduct_smul, smul_eq_mul] at this
        nlinarith
      · rw [← hcx, hc0, zero_smul]
        exact Submodule.zero_mem _
      · have := hx q
        rw [← hcx, dotProduct_smul, smul_eq_mul] at this
        nlinarith
    · exact key y hyY c hcx
    · exact key (-y) hyY (-c) (by rw [smul_neg, neg_smul, neg_neg]; exact hcx)
  · by_cases hext : ∃ i, a i ⬝ᵥ x = 0 ∧ a i ∉ Submodule.span ℝ (Set.range fun j => a (S j))
    · -- add a tight constraint outside the span of `S`
      obtain ⟨i, hi, hni⟩ := hext
      have hfun : (fun j => a ((Fin.cons i S : Fin (r + 1) → Fin m) j)) =
          Fin.cons (a i) (fun j => a (S j)) := by
        funext j
        refine Fin.cases ?_ ?_ j <;> simp
      have hS' : LinearIndependent ℝ (fun j => a ((Fin.cons i S : Fin (r + 1) → Fin m) j)) := by
        rw [hfun]
        exact linearIndependent_finCons.2 ⟨hS, hni⟩
      refine ih (m - (tight a x).card + (d - (r + 1))) (by omega) x (r + 1) (Fin.cons i S) hx hS'
        ?_ (by omega) rfl
      intro j
      refine Fin.cases ?_ ?_ j
      · simpa using hi
      · intro j
        simpa using hSx j
    · -- every tight constraint lies in the span of `S`: split `x`
      push Not at hext
      have hfr := finrank_ker_subA a S hS
      set Kr := LinearMap.ker (subA a S).mulVecLin with hKr
      let f : (Fin d → ℝ) →ₗ[ℝ] ℝ :=
        { toFun := fun z => (∑ i, a i) ⬝ᵥ z
          map_add' := fun u v => dotProduct_add _ u v
          map_smul' := fun c u => by simp [dotProduct_smul] }
      let g := f.domRestrict Kr
      have hg := LinearMap.finrank_range_add_finrank_ker g
      have hrange : finrank ℝ (LinearMap.range g) ≤ 1 := by
        have := Submodule.finrank_le (LinearMap.range g)
        simpa using this
      have hker : 0 < finrank ℝ (LinearMap.ker g) := by omega
      have hne : LinearMap.ker g ≠ ⊥ := by
        intro h
        rw [h, finrank_bot] at hker
        omega
      obtain ⟨⟨z, hzK⟩, hzg, hz0⟩ := Submodule.exists_mem_ne_zero_of_ne_bot hne
      have hz0' : z ≠ 0 := by
        intro h
        exact hz0 (Subtype.ext (by simpa using h))
      have hzS : ∀ j, a (S j) ⬝ᵥ z = 0 := (mem_ker_subA a S z).1 hzK
      have hzc : (∑ i, a i) ⬝ᵥ z = 0 := by
        simpa [g, f, LinearMap.mem_ker] using hzg
      have hzt : ∀ i, a i ⬝ᵥ x = 0 → a i ⬝ᵥ z = 0 :=
        fun i hi => dot_eq_zero_of_mem_span a S z (a i) hzS (hext i hi)
      have hsum : ∑ i, a i ⬝ᵥ z = 0 := by rw [← sum_dot]; exact hzc
      have hnz : ∃ i, a i ⬝ᵥ z ≠ 0 := by
        by_contra h
        push Not at h
        exact hz0' (hpt z h)
      have hneg : ∃ q, a q ⬝ᵥ z < 0 := by
        by_contra h
        push Not at h
        obtain ⟨i, hi⟩ := hnz
        have := (Finset.sum_eq_zero_iff_of_nonneg (fun j _ => h j)).1 hsum i (Finset.mem_univ _)
        exact hi this
      have hpos : ∃ p, a p ⬝ᵥ (-z) < 0 := by
        by_contra h
        push Not at h
        obtain ⟨i, hi⟩ := hnz
        have h' : ∀ j, a j ⬝ᵥ z ≤ 0 := fun j => by
          have := h j
          rw [dotProduct_neg] at this
          linarith
        have := (Finset.sum_eq_zero_iff_of_nonpos (fun j _ => h' j)).1 hsum i (Finset.mem_univ _)
        exact hi this
      obtain ⟨t1, ht1, hx1, hss1⟩ := step a x z hx hzt hneg
      obtain ⟨t2, ht2, hx2, hss2⟩ := step a x (-z) hx
        (fun i hi => by rw [dotProduct_neg, hzt i hi, neg_zero]) hpos
      have hc1 := Finset.card_lt_card hss1
      have hc2 := Finset.card_lt_card hss2
      have hm1 := Finset.card_le_univ (tight a (x + t1 • z))
      have hm2 := Finset.card_le_univ (tight a (x + t2 • -z))
      rw [Fintype.card_fin] at hm1 hm2
      have hmx := Finset.card_le_univ (tight a x)
      rw [Fintype.card_fin] at hmx
      have hm1' := ih _ (by omega) (x + t1 • z) r S hx1 hS
        (fun j => by rw [dotProduct_add, dotProduct_smul, hSx j, hzS j, smul_zero, add_zero])
        hr rfl
      have hm2' := ih _ (by omega) (x + t2 • -z) r S hx2 hS
        (fun j => by
          rw [dotProduct_add, dotProduct_smul, dotProduct_neg, hSx j, hzS j, neg_zero, smul_zero,
            add_zero])
        hr rfl
      have hsum12 : 0 < t1 + t2 := by linarith
      have hdecomp : x = (t2 / (t1 + t2)) • (x + t1 • z) + (t1 / (t1 + t2)) • (x + t2 • -z) := by
        ext i
        simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul, Pi.neg_apply]
        field_simp
        ring
      rw [hdecomp]
      have e1 : (t2 / (t1 + t2)) • (x + t1 • z) =
          (⟨t2 / (t1 + t2), (div_pos ht2 hsum12).le⟩ : NNReal) • (x + t1 • z) := by
        simp
      have e2 : (t1 / (t1 + t2)) • (x + t2 • -z) =
          (⟨t1 / (t1 + t2), (div_pos ht1 hsum12).le⟩ : NNReal) • (x + t2 • -z) := by
        simp
      rw [e1, e2]
      exact Submodule.add_mem _ (Submodule.smul_mem _ _ hm1') (Submodule.smul_mem _ _ hm2')

/-- Every point of the pointed cone `K = {x | aᵢ ⬝ x ≥ 0}` is a nonnegative combination of
elements of `Y`. -/
theorem mem_span (hd : 0 < d) (x : Fin d → ℝ) (hx : ∀ i, 0 ≤ a i ⬝ᵥ x) :
    x ∈ Submodule.span NNReal Y :=
  mem_span_aux a Y hpt hY hcert _ x 0 Fin.elim0 hx (linearIndependent_empty_type)
    (fun i => i.elim0) (by omega) rfl

end main

end Cone
end D4
