/-
D4SecondCode.lean: the second 24-point code of sec:remains of the paper.  At slack
849/50000 = 0.01698 the directions alone no longer place 24 points near a root
system.

The 24 points below have rational coordinates (from multi_cap/second_code/
code24_exact.py, which rounds the best code found by code24_second.py to
points of S^3 by inverse stereographic projection).  Proved here by exact
rational arithmetic:

  * every point has norm 1;
  * every two distinct points have inner product at most 1/2 + 849/50000;
  * the inner product of points 8 and 10 is at distance at least 57/250 from
    each of -1, -1/2, 0, 1/2, 1.

If the set were within d of the normalised root system (d(W) of thm:local-uniqueness: after an orthogonal map and a relabelling the points move by e_i with
sum e_i^2 = d^2), every inner product would be within e_i + e_j <= 2d of one
of those five values.  So d(W) >= 57/500 > 1/48: the set has slack below
0.017 and is far from the root system.  That step is the paper's.

No `sorry`, no Mathlib; settled by native_decide.
-/

namespace D4SecondCode

def pts : List (List Rat) := [
  [(-366185000000 / 569290473223 : Rat), (48486000000 / 569290473223 : Rat), (-46245000000 / 569290473223 : Rat), (430709526777 / 569290473223 : Rat)],
  [(462792798371 / 537207201629 : Rat), (184624000000 / 537207201629 : Rat), (199051000000 / 537207201629 : Rat), (-26591000000 / 537207201629 : Rat)],
  [(33206000000 / 208748679009 : Rat), (191251320991 / 208748679009 : Rat), (27061600000 / 208748679009 : Rat), (-23953600000 / 69582893003 : Rat)],
  [(-742994000000 / 1203383157659 : Rat), (796616842341 / 1203383157659 : Rat), (487842000000 / 1203383157659 : Rat), (153306000000 / 1203383157659 : Rat)],
  [(-197131000000 / 531096681389 : Rat), (-468903318611 / 531096681389 : Rat), (62729000000 / 531096681389 : Rat), (139276000000 / 531096681389 : Rat)],
  [(-268325000000 / 564079660139 : Rat), (179138000000 / 564079660139 : Rat), (155147000000 / 564079660139 : Rat), (-435920339861 / 564079660139 : Rat)],
  [(37904000000 / 254739049833 : Rat), (17828000000 / 84913016611 : Rat), (-245260950167 / 254739049833 : Rat), (21019000000 / 254739049833 : Rat)],
  [(-342379000000 / 588997067583 : Rat), (-228319000000 / 588997067583 : Rat), (-411002932417 / 588997067583 : Rat), (30986000000 / 196332355861 : Rat)],
  [(-478742000000 / 1164451485001 : Rat), (-557552000000 / 1164451485001 : Rat), (-343144000000 / 1164451485001 : Rat), (-835548514999 / 1164451485001 : Rat)],
  [(-491255846963 / 508744153037 : Rat), (-96607000000 / 508744153037 : Rat), (74416000000 / 508744153037 : Rat), (-51163000000 / 508744153037 : Rat)],
  [(407418311281 / 592581688719 : Rat), (44227000000 / 197527229573 : Rat), (-195469000000 / 592581688719 : Rat), (359654000000 / 592581688719 : Rat)],
  [(10784000000 / 101857897991 : Rat), (363148000000 / 1120436877901 : Rat), (-579482000000 / 1120436877901 : Rat), (-879563122099 / 1120436877901 : Rat)],
  [(33045000000 / 524115410099 : Rat), (173757000000 / 524115410099 : Rat), (475884589901 / 524115410099 : Rat), (-130182000000 / 524115410099 : Rat)],
  [(403112000000 / 1080424226553 : Rat), (-919575773447 / 1080424226553 : Rat), (1096000000 / 154346318079 : Rat), (-132974000000 / 360141408851 : Rat)],
  [(270458000000 / 574125873659 : Rat), (-261785000000 / 574125873659 : Rat), (425874126341 / 574125873659 : Rat), (81073000000 / 574125873659 : Rat)],
  [(243374000000 / 1222836216501 : Rat), (208736000000 / 407612072167 : Rat), (663308000000 / 1222836216501 : Rat), (777163783499 / 1222836216501 : Rat)],
  [(36826400000 / 211266465221 : Rat), (-77762800000 / 211266465221 : Rat), (5732000000 / 30180923603 : Rat), (188733534779 / 211266465221 : Rat)],
  [(90666524591 / 109333475409 : Rat), (-13174600000 / 109333475409 : Rat), (-50316200000 / 109333475409 : Rat), (-10688000000 / 36444491803 : Rat)],
  [(234488000000 / 543899032737 : Rat), (-9407000000 / 181299677579 : Rat), (178933000000 / 543899032737 : Rat), (-456100967263 / 543899032737 : Rat)],
  [(-239933000000 / 562048637159 : Rat), (-98890000000 / 562048637159 : Rat), (437951362841 / 562048637159 : Rat), (238223000000 / 562048637159 : Rat)],
  [(-345880000000 / 1212960563349 : Rat), (-210388000000 / 404320187783 : Rat), (787039436651 / 1212960563349 : Rat), (-577790000000 / 1212960563349 : Rat)],
  [(-811776104127 / 1188223895873 : Rat), (581344000000 / 1188223895873 : Rat), (-587016000000 / 1188223895873 : Rat), (-37890000000 / 169746270839 : Rat)],
  [(-17139000000 / 184451655359 : Rat), (446645033923 / 553354966077 : Rat), (-185027000000 / 553354966077 : Rat), (264256000000 / 553354966077 : Rat)],
  [(13027250000 / 38391391053 : Rat), (-24108608947 / 38391391053 : Rat), (-7537250000 / 12797130351 : Rat), (14549000000 / 38391391053 : Rat)]
]

def dot (a b : List Rat) : Rat := (List.zipWith (· * ·) a b).foldl (· + ·) 0

def rabs (x : Rat) : Rat := if x < 0 then -x else x

theorem on_sphere : (pts.length == 24 && pts.all fun p => p.length == 4 && dot p p == 1) = true := by
  native_decide

theorem slack_bound :
    ((List.range 24).all fun i => (List.range 24).all fun j =>
      i == j || dot pts[i]! pts[j]! <= 1 / 2 + 849 / 50000) = true := by
  native_decide

theorem far_from_roots :
    ([(-1 : Rat), -1 / 2, 0, 1 / 2, 1].all fun v => rabs (dot pts[8]! pts[10]! - v) >= 57 / 250) = true := by
  native_decide

end D4SecondCode

#print axioms D4SecondCode.on_sphere
#print axioms D4SecondCode.slack_bound
#print axioms D4SecondCode.far_from_roots
