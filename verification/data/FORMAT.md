# Data formats

All files except the JSON file hold whitespace-separated integers, one record
per line. A 5 x 5 symmetric matrix is written as its 25 entries row by row.
Matrices J encode a two-periodic set as in Section 2 of the paper:
`J = [[Q, r], [r^T, s]]`, the Gram matrix of a basis a_1, ..., a_4 of the
lattice and the translation vector b.

## `vertices.txt`

One line per vertex class J_0, ..., J_9 (Table 5.1):

    id  J(25)  m  k_1(5) ... k_m(5)

`m` is the number of minimal vectors up to sign, and `k_1, ..., k_m` are
those vectors `k = (n, l)` in M = Z^4 x {-1, 0, 1} with `J[k] = 4`.

## `edges.txt`

One line per edge of R(4) at a vertex representative (1137 lines):

    id  i  X(25)  t  j  ...

`i` is the vertex the edge starts from and `X` is the integer direction.

* Bounded edge (`j >= 0`, 1112 lines): `t` is `t*`, the end point is
  `J_i + t X`, and the record ends with `T(25)`, a matrix in Gamma with
  `T^T J_j T = J_i + t X`.
* Unbounded edge (`j = -1`, 25 lines): `t = 0`, and the record ends with
  `u(4) a`, such that `X = p q^T + q p^T` with `p = (u, -a)` and
  `q = (u, -(a + 1))` (Definition 5.2).

## `unbounded_orbits.txt`

One line per unbounded edge (25 lines):

    e  i  r  T(25)

Edge `e` at vertex `i` lies in the orbit of the representative edge `r`
(Table 5.3) under the stabiliser of `J_i` in Gamma: `T^T J_i T = J_i` and
`T^T X_r T = X_e`.

## `r4_vertices_edges.json`

The output of `python/ryshkov2.py`, in exact rationals written as strings:
`vertices` (each with `J` and its minimal vectors `active`) and `edges` (each
with `from`, the direction `X`, `t` = `t*` or `null` for an unbounded edge,
and `to`, the class of the end point, or `null`). `python/certify.py` turns it into the
three tables above, and `python/edges.py` reads it directly.
