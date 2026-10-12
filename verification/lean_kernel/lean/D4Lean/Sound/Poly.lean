import D4Lean.Sound.Basic

/-!
# Polynomials given by coefficient lists

`evalL p t = p₀ + p₁ t + p₂ t² + ⋯`. `hom p y w n = wⁿ p(y / w)`, `comp p u h w n` is the list of
coefficients in `x` of `hom p (u + h x) w n`, and `posBound`, `posInf` show positivity on `(0, 1]`
and on `(0, ∞)`.
-/

namespace D4

/-- `evalL p t = p₀ + p₁ t + p₂ t² + ⋯`. -/
def evalL : List Int → ℝ → ℝ
  | [], _ => 0
  | a :: r, t => a + t * evalL r t

/-- `homR p y w n = ∑ pᵢ yⁱ wⁿ⁻ⁱ` over the reals. -/
def homR : List Int → ℝ → ℝ → Nat → ℝ
  | [], _, _, _ => 0
  | a :: r, y, w, n => a * w ^ n + y * homR r y w (n - 1)

theorem powI_eq (w : Int) (n : Nat) : powI w n = w ^ n := by
  induction n with
  | zero => simp [powI]
  | succ n ih => rw [powI, ih, pow_succ]

theorem hom_cast (p : List Int) (y w : Int) (n : Nat) :
    ((hom p y w n : Int) : ℝ) = homR p y w n := by
  induction p generalizing n with
  | nil => simp [hom, homR]
  | cons a r ih => simp [hom, homR, ih, powI_eq]

theorem homR_eq (p : List Int) (y w : ℝ) (n : Nat) (hw : w ≠ 0) (hp : p.length ≤ n + 1) :
    homR p y w n = w ^ n * evalL p (y / w) := by
  induction p generalizing n with
  | nil => simp [homR, evalL]
  | cons a r ih =>
    cases n with
    | zero =>
      have hr : r.length = 0 := by simp only [List.length_cons] at hp; omega
      have : r = [] := List.eq_nil_of_length_eq_zero hr
      subst this
      simp [homR, evalL]
    | succ m =>
      rw [homR, Nat.add_sub_cancel, ih m (by simp at hp; omega), evalL]
      field_simp
      ring

theorem evalL_addP (p q : List Int) (x : ℝ) : evalL (addP p q) x = evalL p x + evalL q x := by
  induction p generalizing q with
  | nil => simp [addP, evalL]
  | cons a p ih =>
    cases q with
    | nil => simp [addP, evalL]
    | cons b q => simp [addP, evalL, ih]; ring

theorem evalL_map_mul (u : Int) (c : List Int) (x : ℝ) :
    evalL (c.map (u * ·)) x = u * evalL c x := by
  induction c with
  | nil => simp [evalL]
  | cons a c ih => simp [evalL, ih]; ring

theorem evalL_mulLin (u h : Int) (c : List Int) (x : ℝ) :
    evalL (mulLin u h c) x = (u + h * x) * evalL c x := by
  simp only [mulLin, evalL_addP, evalL, evalL_map_mul]
  push_cast
  ring

theorem evalL_comp (p : List Int) (u h w : Int) (n : Nat) (x : ℝ) :
    evalL (comp p u h w n) x = homR p (u + h * x) w n := by
  induction p generalizing n with
  | nil => simp [comp, evalL, homR]
  | cons a r ih =>
    simp only [comp, evalL_addP, evalL_mulLin, ih, homR, evalL, powI_eq]
    push_cast
    ring

theorem negPart_nonpos (r : List Int) : negPart r ≤ 0 := by
  induction r with
  | nil => simp [negPart]
  | cons a r ih => simp only [negPart]; have := min_le_right a 0; omega

theorem evalL_ge_negPart (r : List Int) (x : ℝ) (h0 : 0 ≤ x) (h1 : x ≤ 1) :
    (negPart r : ℝ) ≤ evalL r x := by
  induction r with
  | nil => simp [negPart, evalL]
  | cons a r ih =>
    simp only [negPart, evalL]
    push_cast
    have hn : (negPart r : ℝ) ≤ 0 := by exact_mod_cast negPart_nonpos r
    have hm : (min (a : ℝ) 0) ≤ a := min_le_left _ _
    have h2 : x * (negPart r : ℝ) ≤ x * evalL r x := mul_le_mul_of_nonneg_left ih h0
    have h3 : (negPart r : ℝ) ≤ x * (negPart r : ℝ) := by nlinarith
    linarith

theorem posBound_sound (c : List Int) (h : posBound c = true) (x : ℝ) (h0 : 0 < x)
    (h1 : x ≤ 1) : 0 < evalL c x := by
  induction c with
  | nil => simp [posBound] at h
  | cons a r ih =>
    simp only [posBound] at h
    split_ifs at h with ha
    · have ha' : a = 0 := by simpa using ha
      subst ha'
      simp only [evalL, Int.cast_zero, zero_add]
      exact mul_pos h0 (ih h)
    · have h2 : (0 : ℝ) < a + negPart r := by exact_mod_cast (decide_eq_true_iff.mp h)
      have h3 := evalL_ge_negPart r x h0.le h1
      have hn : (negPart r : ℝ) ≤ 0 := by exact_mod_cast negPart_nonpos r
      simp only [evalL]
      nlinarith

theorem evalL_nonneg (r : List Int) (h : ∀ c ∈ r, 0 ≤ c) (x : ℝ) (h0 : 0 ≤ x) :
    0 ≤ evalL r x := by
  induction r with
  | nil => simp [evalL]
  | cons a r ih =>
    simp only [evalL]
    have ha : (0 : ℝ) ≤ a := by exact_mod_cast h a (by simp)
    have := ih (fun c hc => h c (by simp [hc]))
    positivity

theorem posInf_sound (c : List Int) (h : posInf c = true) (x : ℝ) (h0 : 0 < x) :
    0 < evalL c x := by
  induction c with
  | nil => simp [posInf] at h
  | cons a r ih =>
    simp only [posInf] at h
    split_ifs at h with ha
    · have ha' : a = 0 := by simpa using ha
      subst ha'
      simp only [evalL, Int.cast_zero, zero_add]
      exact mul_pos h0 (ih h)
    · simp only [Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at h
      have h2 : (0 : ℝ) < a := by exact_mod_cast h.1
      have h3 := evalL_nonneg r h.2 x h0.le
      simp only [evalL]
      positivity

theorem evalL_gOf (p : List Int) (t : ℝ) : evalL (gOf p) t = evalL p t - 256 := by
  cases p with
  | nil => simp [gOf, evalL]
  | cons a r => simp [gOf, evalL]; ring

theorem length_gOf (p : List Int) : (gOf p).length = max p.length 1 := by
  cases p with
  | nil => simp [gOf]
  | cons a r => simp [gOf]

/-- Positivity of `P` at `(a + h x) / wd` for `x ∈ (0, 1]` from the coefficient test. -/
theorem comp_pos (P : List Int) (a h wd : Int) (n : Nat) (hwd : 0 < wd) (hl : P.length ≤ n + 1)
    (hp : posBound (comp P a h wd n) = true) (x : ℝ) (h0 : 0 < x) (h1 : x ≤ 1) :
    0 < evalL P ((a + h * x) / wd) := by
  have hw : (wd : ℝ) ≠ 0 := by exact_mod_cast hwd.ne'
  have hw' : (0 : ℝ) < wd := by exact_mod_cast hwd
  have := posBound_sound _ hp x h0 h1
  rw [evalL_comp, homR_eq P _ _ n hw hl] at this
  exact pos_of_mul_pos_right this (by positivity)

/-- Positivity of `P` at `(a + x) / wd` for `x > 0` from the coefficient test. -/
theorem comp_posInf (P : List Int) (a wd : Int) (n : Nat) (hwd : 0 < wd) (hl : P.length ≤ n + 1)
    (hp : posInf (comp P a 1 wd n) = true) (x : ℝ) (h0 : 0 < x) :
    0 < evalL P ((a + x) / wd) := by
  have hw : (wd : ℝ) ≠ 0 := by exact_mod_cast hwd.ne'
  have hw' : (0 : ℝ) < wd := by exact_mod_cast hwd
  have := posInf_sound _ hp x h0
  rw [evalL_comp, homR_eq P _ _ n hw hl] at this
  simp only [Int.cast_one, one_mul] at this
  exact pos_of_mul_pos_right this (by positivity)

/-- The value of `hom` at a break point. -/
theorem hom_eq (P : List Int) (u wd : Int) (n : Nat) (hwd : 0 < wd) (hl : P.length ≤ n + 1) :
    ((hom P u wd n : Int) : ℝ) = (wd : ℝ) ^ n * evalL P ((u : ℝ) / wd) := by
  have hw : (wd : ℝ) ≠ 0 := by exact_mod_cast hwd.ne'
  rw [hom_cast, homR_eq P _ _ n hw hl]

end D4
