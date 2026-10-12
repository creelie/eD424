import D4Lean.Sound.Facts
import D4Lean.Sound.Cone

/-!
# The cone at a vertex

`ConeOK` gives two facts about a vertex `J` with minimal vectors `mins` and listed edge directions
`Y_e`. First, `J` is a vertex of `R`: the only symmetric `X` with `X[k] = 0` at every minimal
vector is `0`. Second, every symmetric `X` with `X[k] ≥ 0` at every minimal vector, that is every
direction of the tangent cone of `R` at `J`, is a nonnegative combination of the `Y_e`; so every
edge of `R` at `J` points along one of the listed directions. The second fact is
`D4.Cone.mem_span` applied to the rows `symcL k` in the coordinates `psiL`.
-/

open Matrix

namespace D4

/-- Row and column of the coordinate `v` of the upper triangle. -/
def pI : Fin 15 → Fin 5 := ![0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4]
def pJ : Fin 15 → Fin 5 := ![0, 1, 2, 3, 4, 1, 2, 3, 4, 2, 3, 4, 3, 4, 4]

/-- The coordinate of the entry `(i, j)` of a symmetric matrix. -/
def pIdx : Fin 5 → Fin 5 → Fin 15 :=
  ![![0, 1, 2, 3, 4], ![1, 5, 6, 7, 8], ![2, 6, 9, 10, 11], ![3, 7, 10, 12, 13],
    ![4, 8, 11, 13, 14]]

/-- A list of integers as a real vector of length `15`. -/
def vR (l : List Int) : Fin 15 → ℝ := fun v => (vg l v : ℝ)

/-- The upper triangle of a real matrix. -/
def psiR (X : M5) : Fin 15 → ℝ := fun v => X (pI v) (pJ v)

/-- The symmetric matrix with upper triangle `y`. -/
def phiR (y : Fin 15 → ℝ) : M5 := Matrix.of fun i j => y (pIdx i j)

theorem vg_symcL (k : List Int) (v : Fin 15) :
    vg (symcL k) v = (if pI v = pJ v then 1 else 2) * vg k (pI v) * vg k (pJ v) := by
  fin_cases v <;> rfl

theorem vg_psiL (Y : Mat) (v : Fin 15) : vg (psiL Y) v = ent Y (pI v) (pJ v) := by
  fin_cases v <;> rfl

theorem length_symcL (k : List Int) : (symcL k).length = 15 := rfl

theorem length_psiL (Y : Mat) : (psiL Y).length = 15 := rfl

theorem phiR_psiR (X : M5) (hX : X.IsSymm) : phiR (psiR X) = X := by
  ext i j
  have h := hX.apply
  fin_cases i <;> fin_cases j <;> simp only [phiR, psiR, Matrix.of_apply] <;>
    first | rfl | exact h _ _

theorem phiR_add (y z : Fin 15 → ℝ) : phiR (y + z) = phiR y + phiR z := by
  ext i j; simp [phiR]

theorem phiR_smul (c : NNReal) (y : Fin 15 → ℝ) : phiR (c • y) = c • phiR y := by
  ext i j; simp [phiR, NNReal.smul_def]

theorem phiR_zero : phiR 0 = 0 := by
  ext i j; simp [phiR]

theorem phiR_psiL (Y : Mat) (hY : symm Y = true) : phiR (vR (psiL Y)) = toR Y := by
  have h := (symm_sound Y hY).apply
  rw [← phiR_psiR (toR Y) (symm_sound Y hY)]
  ext i j
  simp only [phiR, vR, psiR, Matrix.of_apply, vg_psiL, toR, toZ, Matrix.map_apply]

theorem qform_eq_dot (X : M5) (hX : X.IsSymm) (k : List Int) :
    vR (symcL k) ⬝ᵥ psiR X = qform X (fun j => vg k j) := by
  have h := hX.apply
  simp only [dotProduct, vR, vg_symcL, psiR, qform, Fin.sum_univ_succ, Fin.sum_univ_zero]
  simp [pI, pJ]
  rw [h 0 1, h 0 2, h 0 3, h 0 4, h 1 2, h 1 3, h 1 4, h 2 3, h 2 4, h 3 4]
  ring

theorem vR_dot (l1 l2 : List Int) (h1 : l1.length = 15) (h2 : l2.length = 15) :
    vR l1 ⬝ᵥ vR l2 = (dot l1 l2 : ℝ) := by
  rw [dot_eq l1 l2 15 h1 h2]
  simp [dotProduct, vR]

theorem vg_negV : ∀ (l : List Int) (i : Nat), vg (negV l) i = -vg l i
  | [], i => by simp [vg, negV]
  | x :: l, 0 => rfl
  | x :: l, i + 1 => by
    rw [show negV (x :: l) = -x :: negV l from rfl, vg_cons_succ, vg_cons_succ, vg_negV l i]

theorem vR_negV (l : List Int) : vR (negV l) = -vR l := by
  funext v; simp [vR, vg_negV]

theorem vR_ne_zero (l : List Int) (hl : l.length = 15) (h : isZero l = false) : vR l ≠ 0 := by
  intro h0
  have h1 : isZero l = true := by
    simp only [isZero, List.all_eq_true, beq_iff_eq]
    intro x hx
    obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem hx
    have h2 := congrFun h0 ⟨j, by omega⟩
    simp only [vR, Pi.zero_apply, Int.cast_eq_zero, vg] at h2
    rwa [List.getD_eq_getElem _ _ hj] at h2
  rw [h1] at h
  exact absurd h (by decide)

/-! ### The rank certificate -/

theorem rank_sound (rows : List (List Int)) (idx : List Nat) (B : Mat) (c : Int)
    (h : rankOK rows idx B c = true) (z : Fin 15 → ℝ)
    (hz : ∀ i : Fin rows.length, vR (rows.getD i []) ⬝ᵥ z = 0) : z = 0 := by
  simp only [rankOK, Bool.and_eq_true, bne_iff_ne, ne_eq, beq_iff_eq, List.all_eq_true,
    decide_eq_true_eq, allN_iff] at h
  obtain ⟨⟨⟨hc, hlen⟩, hidx⟩, hB⟩ := h
  funext u
  have hrow : ∀ t : Fin 15, vR (rows.getD (idx.getD t 0) []) ⬝ᵥ z = 0 := by
    intro t
    have ht : idx.getD t 0 < rows.length := by
      apply hidx
      rw [List.getD_eq_getElem _ _ (by omega)]
      exact List.getElem_mem _
    exact hz ⟨_, ht⟩
  have key : (c : ℝ) * z u = 0 := by
    have e1 : (c : ℝ) * z u =
        ∑ v : Fin 15, (((if (u : Nat) = v then c else 0 : ℤ)) : ℝ) * z v := by
      rw [Finset.sum_eq_single u]
      · simp
      · intro b _ hb
        have : (u : Nat) ≠ b := fun h' => hb (Fin.ext h').symm
        simp [this]
      · simp
    have e2 : ∀ v : Fin 15, (if (u : Nat) = v then c else 0 : ℤ) =
        ∑ t : Fin 15, ent B u t * vg (rows.getD (idx.getD t 0) []) v := by
      intro v
      rw [← hB u u.2 v v.2, sumN_eq]
    rw [e1]
    simp only [e2, Int.cast_sum, Int.cast_mul, Finset.sum_mul]
    rw [Finset.sum_comm]
    have e3 : ∀ t : Fin 15, ∑ v : Fin 15,
        (ent B u t : ℝ) * (vg (rows.getD (idx.getD t 0) []) v : ℝ) * z v =
        (ent B u t : ℝ) * (vR (rows.getD (idx.getD t 0) []) ⬝ᵥ z) := by
      intro t
      rw [dotProduct, Finset.mul_sum]
      refine Finset.sum_congr rfl fun v _ => ?_
      simp only [vR]
      ring
    simp only [e3, hrow, mul_zero, Finset.sum_const_zero]
  have hc' : (c : ℝ) ≠ 0 := by exact_mod_cast hc
  simpa [hc'] using key

/-! ### Flats and sets of rows -/

theorem split_mem {α : Type} {x : α} : ∀ l : List α, (x ∈ (split l).1 → x ∈ l) ∧ (x ∈ (split l).2 → x ∈ l)
  | [] => by simp [split]
  | [y] => by simp [split]
  | y :: z :: r => by
    have ih := split_mem (x := x) r
    simp only [split, List.mem_cons]
    exact ⟨fun h => h.elim Or.inl fun h => Or.inr (Or.inr (ih.1 h)),
      fun h => h.elim (fun h => Or.inr (Or.inl h)) fun h => Or.inr (Or.inr (ih.2 h))⟩

/-- Every flat of the tree of a list is in the list. -/
theorem FTree.get_mem : ∀ (k : Nat) (l : List Flat) (n : Nat) (f : Flat),
    (FTree.ofList k l).get n = some f → f ∈ l
  | 0, l, n, f, h => by simp [FTree.ofList, FTree.get] at h
  | k + 1, [], n, f, h => by simp [FTree.ofList, FTree.get] at h
  | k + 1, g :: gs, 0, f, h => by
    simp only [FTree.ofList, FTree.get, Option.some.injEq] at h
    exact h ▸ List.mem_cons_self
  | k + 1, g :: gs, n + 1, f, h => by
    simp only [FTree.ofList, FTree.get] at h
    split at h
    · exact List.mem_cons_of_mem _ ((split_mem gs).1 (FTree.get_mem k _ _ f h))
    · exact List.mem_cons_of_mem _ ((split_mem gs).2 (FTree.get_mem k _ _ f h))

theorem testBit_of_land {a b i : Nat} (h : (a &&& b == a) = true) (hi : a.testBit i = true) :
    b.testBit i = true := by
  rw [beq_iff_eq] at h
  have := congrArg (fun n => n.testBit i) h
  simp only [Nat.testBit_land, hi, Bool.true_and] at this
  exact this

/-- What `subsOK` shows: each `k`-element sublist `S` of `l`, with the rows of `acc`, lies in a
flat of `t`. -/
theorem subsOK_sound (t : FTree) : ∀ (k : Nat) (l : List Nat) (acc : Nat) (js rest : List Nat),
    subsOK t k l acc js = some rest → ∀ S ∈ combs k l, ∃ j f, t.get j = some f ∧
      (∀ i ∈ S, f.mask.testBit i = true) ∧ ∀ i, acc.testBit i = true → f.mask.testBit i = true := by
  intro k
  induction k with
  | zero =>
    intro l acc js rest h S hS
    have hS' : S = [] := by cases l <;> simpa [combs] using hS
    subst hS'
    cases js with
    | nil => cases l <;> simp [subsOK] at h
    | cons j js =>
      cases hg : t.get j with
      | none => cases l <;> simp [subsOK, hg] at h
      | some f =>
        have hm : (acc &&& f.mask == acc) = true := by
          cases l <;> simp only [subsOK, hg] at h <;> split at h <;> simp_all
        exact ⟨j, f, hg, by simp, fun i hi => testBit_of_land hm hi⟩
  | succ k ihk =>
    intro l
    induction l with
    | nil => intro acc js rest h S hS; simp [combs] at hS
    | cons x xs ihl =>
      intro acc js rest h S hS
      simp only [subsOK] at h
      cases h1 : subsOK t k xs (acc ||| 2 ^ x) js with
      | none => simp [h1] at h
      | some js' =>
        rw [h1] at h
        simp only [combs, List.mem_append, List.mem_map] at hS
        rcases hS with ⟨S', hS', rfl⟩ | hS
        · obtain ⟨j, f, hg, hb, ha⟩ := ihk xs _ js js' h1 S' hS'
          refine ⟨j, f, hg, ?_, fun i hi => ha i (by simp [hi])⟩
          intro i hi
          rcases List.mem_cons.1 hi with rfl | hi
          · exact ha i (by simp [Nat.testBit_two_pow_self])
          · exact hb i hi
        · exact ihl acc js' rest h S hS

theorem mem_combs : ∀ (k : Nat) (xs l : List Nat), List.Sublist l xs → l.length = k → l ∈ combs k xs
  | 0, xs, l, _, hk => by
    rw [List.length_eq_zero_iff] at hk
    subst hk
    cases xs <;> simp [combs]
  | k + 1, [], l, hs, hk => by
    rw [List.sublist_nil] at hs
    subst hs
    simp at hk
  | k + 1, x :: xs, l, hs, hk => by
    simp only [combs, List.mem_append, List.mem_map]
    cases hs with
    | cons _ h => exact Or.inr (mem_combs (k + 1) xs l h hk)
    | cons_cons _ h => exact Or.inl ⟨_, mem_combs k xs _ h (by simpa using hk), rfl⟩

theorem exists_sublist {m : Nat} (s : Finset (Fin m)) : ∃ l : List Nat, List.Sublist l (List.range m) ∧
    l.length = s.card ∧ ∀ i : Fin m, i ∈ s → (i : Nat) ∈ l := by
  classical
  refine ⟨(List.range m).filter (fun i => decide (∃ x ∈ s, (x : Nat) = i)),
    List.filter_sublist, ?_, ?_⟩
  · have hnd : ((List.range m).filter (fun i => decide (∃ x ∈ s, (x : Nat) = i))).Nodup :=
      List.nodup_range.filter _
    rw [← List.toFinset_card_of_nodup hnd]
    rw [show ((List.range m).filter (fun i => decide (∃ x ∈ s, (x : Nat) = i))).toFinset =
      s.map Fin.valEmbedding from ?_, Finset.card_map]
    ext i
    simp only [List.mem_toFinset, List.mem_filter, List.mem_range, decide_eq_true_eq,
      Finset.mem_map, Fin.valEmbedding_apply]
    constructor
    · rintro ⟨_, h⟩; exact h
    · rintro ⟨x, hx, rfl⟩; exact ⟨x.2, x, hx, rfl⟩
  · intro i hi
    simp only [List.mem_filter, List.mem_range, decide_eq_true_eq]
    exact ⟨i.2, i, hi, rfl⟩

theorem flatOK_sound (rows rays : List (List Int)) (f : Flat) (h : flatOK rows rays f = true) :
    (flatY rays f).length = 15 ∧ isZero (flatY rays f) = false ∧
    (∀ i < rows.length, f.mask.testBit i = true → dot (rows.getD i []) (flatY rays f) = 0) ∧
    ((∃ y p q, f.kind = .mixed y p q ∧ p < rows.length ∧ q < rows.length ∧
        0 < dot (rows.getD p []) (flatY rays f) ∧ dot (rows.getD q []) (flatY rays f) < 0) ∨
      ∃ r neg, f.kind = .ray r neg ∧ r < rays.length) := by
  obtain ⟨mask, kind⟩ := f
  have hz : ∀ (y : List Int), (∀ i < rows.length, (!mask.testBit i || dot (rows.getD i []) y == 0) = true) →
      ∀ i < rows.length, mask.testBit i = true → dot (rows.getD i []) y = 0 := by
    intro y hy i hi hb
    have := hy i hi
    rw [hb] at this
    simpa using this
  cases kind with
  | mixed y p q =>
    simp only [flatOK, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq, allN_iff] at h
    obtain ⟨⟨⟨hlen, hnz⟩, hzero⟩, ⟨⟨⟨hp, hq⟩, hpp⟩, hqq⟩⟩ := h
    refine ⟨hlen, by simpa using hnz, hz _ hzero, Or.inl ⟨y, p, q, rfl, hp, hq, hpp, hqq⟩⟩
  | ray r neg =>
    simp only [flatOK, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq, allN_iff] at h
    obtain ⟨⟨⟨hlen, hnz⟩, hzero⟩, hr⟩ := h
    exact ⟨hlen, by simpa using hnz, hz _ hzero, Or.inr ⟨r, neg, rfl, hr⟩⟩

/-! ### The cone of the rows -/

/-- The checks on the rows, the rays and the flats: the rows span, so the cone they cut out is
pointed, and the rays span that cone. -/
theorem rows_cone (rows rays : List (List Int)) (idx : List Nat) (B : Mat) (c : Int)
    (flats : FTree) (hrl : ∀ r ∈ rows, r.length = 15)
    (hrank : rankOK rows idx B c = true)
    (hrays : raysOK rows rays = true)
    (hflats : ∀ j f, flats.get j = some f → flatOK rows rays f = true)
    (hsub : ∃ js, subsOK flats 14 (List.range rows.length) 0 js = some []) :
    (∀ z : Fin 15 → ℝ, (∀ i : Fin rows.length, vR (rows.getD i []) ⬝ᵥ z = 0) → z = 0) ∧
    ∀ x : Fin 15 → ℝ, (∀ i : Fin rows.length, 0 ≤ vR (rows.getD i []) ⬝ᵥ x) →
      x ∈ Submodule.span NNReal {y | ∃ r ∈ rays, y = vR r} := by
  have hpt := rank_sound rows idx B c hrank
  refine ⟨hpt, fun x hx => ?_⟩
  have hrow : ∀ i : Fin rows.length, (rows.getD i []).length = 15 := by
    intro i
    rw [List.getD_eq_getElem _ _ i.2]
    exact hrl _ (List.getElem_mem _)
  obtain ⟨js, hsub⟩ := hsub
  simp only [raysOK, List.all_eq_true, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hrays
  refine Cone.mem_span (fun i : Fin rows.length => vR (rows.getD i [])) _ hpt ?_ ?_
    (by norm_num) x hx
  · rintro y ⟨r, hr, rfl⟩ i
    rw [vR_dot _ _ (hrow i) (hrays r hr).1]
    have := (hrays r hr).2 (rows.getD i [])
      (by rw [List.getD_eq_getElem _ _ i.2]; exact List.getElem_mem _)
    exact_mod_cast this
  · intro s hs
    obtain ⟨l, hl_sub, hl_len, hl_mem⟩ := exists_sublist s
    obtain ⟨j, f, hget, hbits, -⟩ :=
      subsOK_sound flats 14 _ 0 js [] hsub l (mem_combs 14 _ l hl_sub (by omega))
    have hf := flatOK_sound rows rays _ (hflats j f hget)
    obtain ⟨hlen, hnz, hzero, hkind⟩ := hf
    refine ⟨vR (flatY rays f), vR_ne_zero _ hlen hnz, ?_, ?_⟩
    · intro i hi
      rw [vR_dot _ _ (hrow i) hlen]
      exact_mod_cast hzero i i.2 (hbits i (hl_mem i hi))
    · rcases hkind with ⟨y, p, q, _, hp, hq, hpp, hqq⟩ | ⟨r, neg, hk, hr⟩
      · left
        refine ⟨⟨p, hp⟩, ⟨q, hq⟩, ?_, ?_⟩
        · rw [vR_dot _ _ (hrow _) hlen]; exact_mod_cast hpp
        · rw [vR_dot _ _ (hrow _) hlen]; exact_mod_cast hqq
      · have hr' : rays.getD r [] ∈ rays := by
          rw [List.getD_eq_getElem _ _ hr]; exact List.getElem_mem _
        right
        cases neg with
        | false => left; exact ⟨_, hr', by simp [flatY, hk]⟩
        | true =>
          right
          refine ⟨_, hr', ?_⟩
          simp [flatY, hk, vR_negV]

/-- What `ConeOK` shows at a vertex `v` with edges `es`: `v.J` is a vertex of `R` (no nonzero
symmetric `X` has `X[k] = 0` at every minimal vector), and every symmetric `X` with `X[k] ≥ 0` at
every minimal vector is a nonnegative combination of the directions of the edges. -/
structure CFacts (v : Vtx) (es : List Edge) : Prop where
  vertex : ∀ X : M5, X.IsSymm → (∀ k : Fin 5 → ℤ, List.ofFn k ∈ v.mins → qform X k = 0) → X = 0
  cone : ∀ X : M5, X.IsSymm → (∀ k : Fin 5 → ℤ, List.ofFn k ∈ v.mins → 0 ≤ qform X k) →
    X ∈ Submodule.span NNReal {Y | ∃ e ∈ es, Y = toR e.Y}

theorem phiR_span (es : List Edge) (hsym : ∀ e ∈ es, symm e.Y = true) (y : Fin 15 → ℝ)
    (hy : y ∈ Submodule.span NNReal {y | ∃ r ∈ es.map (fun e => psiL e.Y), y = vR r}) :
    phiR y ∈ Submodule.span NNReal {Y | ∃ e ∈ es, Y = toR e.Y} := by
  induction hy using Submodule.span_induction with
  | mem w hw =>
    obtain ⟨r, hr, rfl⟩ := hw
    obtain ⟨e, he, rfl⟩ := List.mem_map.1 hr
    exact Submodule.subset_span ⟨e, he, phiR_psiL e.Y (hsym e he)⟩
  | zero => rw [phiR_zero]; exact Submodule.zero_mem _
  | add u w _ _ hu hw => rw [phiR_add]; exact Submodule.add_mem _ hu hw
  | smul c u _ hu => rw [phiR_smul]; exact Submodule.smul_mem _ c hu

theorem cone_sound (v : Vtx) (es : List Edge) (cd : ConeData)
    (hmins : ∀ k ∈ v.mins, k.length = 5) (hsym : ∀ e ∈ es, symm e.Y = true)
    (h : ConeOK v.mins (es.map fun e => psiL e.Y) cd) : CFacts v es := by
  obtain ⟨hrank, hrays, hflats, hsub⟩ := h
  have hflats' : ∀ j f, cd.tree.get j = some f →
      flatOK (v.mins.map symcL) (es.map fun e => psiL e.Y) f = true := by
    intro j f hg
    obtain ⟨n, hn, rfl⟩ := List.mem_map.1 (FTree.get_mem 20 _ j f hg)
    simp only [flatsOK, List.all_eq_true] at hflats
    exact hflats n hn
  have hrl : ∀ r ∈ v.mins.map symcL, r.length = 15 := by
    intro r hr
    obtain ⟨k, _, rfl⟩ := List.mem_map.1 hr
    exact length_symcL k
  obtain ⟨hpt, hspan⟩ := rows_cone _ _ cd.idx cd.B cd.c cd.tree hrl hrank hrays hflats' hsub
  -- the row `i` evaluated at `psiR X` is `X[k]` for the minimal vector `k = mins[i]`
  have hval : ∀ X : M5, X.IsSymm → ∀ i : Fin (v.mins.map symcL).length,
      ∃ k : Fin 5 → ℤ, List.ofFn k ∈ v.mins ∧
        vR ((v.mins.map symcL).getD i []) ⬝ᵥ psiR X = qform X k := by
    intro X hX ⟨n, hn⟩
    have hi : n < v.mins.length := by simpa using hn
    have hk : v.mins[n] ∈ v.mins := List.getElem_mem _
    refine ⟨fun j => vg v.mins[n] j, ?_, ?_⟩
    · rw [ofFn_vg _ (hmins _ hk)]; exact hk
    · have : (v.mins.map symcL).getD n [] = symcL v.mins[n] := by
        rw [List.getD_eq_getElem _ _ hn, List.getElem_map]
      show vR ((v.mins.map symcL).getD n []) ⬝ᵥ psiR X = _
      rw [this, qform_eq_dot X hX]
  constructor
  · intro X hX h0
    have hz := hpt (psiR X) fun i => by
      obtain ⟨k, hk, he⟩ := hval X hX i
      rw [he, h0 k hk]
    rw [← phiR_psiR X hX, hz, phiR_zero]
  · intro X hX h0
    have hx := hspan (psiR X) fun i => by
      obtain ⟨k, hk, he⟩ := hval X hX i
      rw [he]; exact h0 k hk
    rw [← phiR_psiR X hX]
    exact phiR_span es hsym _ hx

/-- The minimal vectors of a checked vertex have length 5. -/
theorem mins_length (v : Vtx) (h : v.ok = true) : ∀ k ∈ v.mins, k.length = 5 := by
  simp only [Vtx.ok, Bool.and_eq_true] at h
  have hm := h.1.1.1.2
  simp only [memOK, Bool.and_eq_true, List.all_eq_true] at hm
  intro k hk
  have := hm.1.2 k hk
  simp only [minOK, Bool.and_eq_true, beq_iff_eq] at this
  exact this.1.1.1

/-- The direction of a checked edge is symmetric. -/
theorem edge_symm (J Jto : Mat) (e : Edge) (h : Edge.ok J Jto e = true) : symm e.Y = true := by
  cases hs : symm e.Y
  · simp [Edge.ok, boundedOK, unboundedOK, hs] at h
  · rfl

end D4
