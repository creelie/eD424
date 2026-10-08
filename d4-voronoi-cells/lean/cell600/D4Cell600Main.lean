/-
D4Cell600Main.lean

The three enumeration theorems behind prop:cell600: the compiled
depth-first search of D4Cell600Enum.lean, evaluated by `native_decide`.
Building with `lake build` in this directory compiles D4Cell600Enum to
native code first (precompileModules), so the search runs in a few seconds.
The axiom audit at the foot of the file records, for each theorem, the
trust placed in the compiler: besides `propext` and `Quot.sound`, each
depends on its own `native_decide` axiom, which is how Lean 4 names the
compiled evaluation.
-/
import D4Cell600Enum

namespace D4Cell600

/-- No independent set of size 25 contains vertex 0; by vertex transitivity the
    independence number of the edge graph is 24. -/
theorem no_25 : throughZero 25 = (0, 0) := by native_decide

/-- Exactly five independent 24-sets contain vertex 0, all of them cells. -/
theorem five_24 : throughZero 24 = (5, 0) := by native_decide

/-- Exactly 115 independent 23-sets contain vertex 0, and every one of them
    lies in a cell. Since 115 = 5 x 23, they are the five cells through the
    vertex with one of the other twenty-three vertices deleted; by vertex
    transitivity there are 115 x 120 / 23 = 600 = 25 x 24 in all. -/
theorem all_23_in_cells : throughZero 23 = (115, 0) := by native_decide

end D4Cell600

#print axioms D4Cell600.no_25
#print axioms D4Cell600.five_24
#print axioms D4Cell600.all_23_in_cells
