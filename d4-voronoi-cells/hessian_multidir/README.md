# hessian_multidir/

The joint-Hessian computations of sec:hessian-technique to sec:broad-sample of the paper. Apart
from the two exact group computations marked below, every result here
is numerical and the paper labels it as a numerical observation. None of
it is used in the proof of any theorem: the multi-direction case is
settled by cor:conj-resolved through the classification of thm:m24.

    extend_hessian.py, c2_constant_test.py, c2_hessian_relation_check2.py,
    triality_search.py, triality_search2.py, triality_exact.py,
    chain_length.py, m3_hessian.py
                              Exploratory Hessian scans and the chain
                              configurations of Numerical observation 15.3.

    joint_hessian_closed_form.py
                              The closed form of the cross-Hessian of two
                              adjacent directions, obtained by fitting
                              finite differences (to about 1e-6), and its
                              minimum eigenvalue 1/3 (Numerical
                              observation, sec:dense-configs-new).

    vertex_degeneracy_check.py
                              Exact vertex enumeration for sec:dense-configs-new: two
                              adjacent facets of the 24-cell share 3 of
                              their 6 vertices, and after one facet is
                              perturbed, 4 of the 10 completions to a
                              determining facet set remain feasible.

    multidir_exact_zero_hessian_hp.py
                              The configuration A_18 in 35- to 45-digit
                              arithmetic: the second-derivative estimates
                              along near-null directions decrease by a
                              factor of about 4 per halving of the step,
                              and the quartic coefficient is positive
                              (sec:exact-sing).

    multidir_nullspace_broad_sample.py, nullspace_broad_sample_A18.log
                              The scaling test that finds a 4-dimensional
                              near-null subspace at A_18 (Numerical
                              observation 17.2) and the sampling of that
                              subspace along 36 directions at two step
                              sizes (Numerical observation 18.1). The log
                              is the run quoted in the paper.

    multidir_nullspace_broad_sample_m20.py
                              The same at A_20: a 5-dimensional near-null
                              subspace sampled along 30 directions
                              (Numerical observations 18.2 and 18.3).

    quartic_sample_dense.py, quartic_dense_A18_data.tsv,
    quartic_dense_A18_run.log
                              166 estimates of the quartic coefficient at
                              A_18, the data of the least-squares fit of
                              sec:broad-sample. The script skips directions
                              already present in its output file.

    gram_sos_lib.py           The Gram-matrix semidefinite test for a sum-
                              of-squares representation of a quartic form
                              (cvxpy). Run directly, it checks itself on a
                              perfect square, which must pass, and on the
                              Choi-Lam form, which must fail.

    quartic_fit_and_check.py  Fits the 35 coefficients of a general quartic
                              form at A_18 from quartic_dense_A18_data.tsv
                              and applies gram_sos_lib.py to the fitted
                              form (Numerical observation 18.4).

    multidir_chain_hessian_extended.py, hp_volume.py
                              Root system, tangent bases and the
                              high-precision volume routine of sec:hp-volume-method,
                              used by the scripts above (copied from core/
                              so that this directory runs on its own).

    a18_stabilizer_group.py   Exact: within the 384 signed coordinate
                              permutations preserving the D4 roots, the
                              stabiliser of A_18 has order 48 (prop:a18-stabiliser).

    a18_other_starts.py       Exact: the configurations grown from roots 5
                              and 12 are images of A_18 under orthogonal
                              maps preserving the D4 roots (sec:broad-sample).

    a18_symmetry_representation_check.py
                              Checks numerically that the induced
                              54-dimensional representation of that group
                              commutes with the finite-difference Hessian
                              (to about 4e-6) and leaves the quartic
                              coefficient invariant (to about 8e-5), and
                              computes its character on the near-null
                              subspace and the commutant dimension, 2.

    a18_exact_character_table.py
                              Exact character table of the stabiliser (10
                              classes, 10 irreducible characters built from
                              its natural representations, checked by the
                              orthogonality relations) and the decomposition
                              of the 54-dimensional tangent representation,
                              whose multiplicities are non-negative integers
                              summing to 54.

    a18_nullspace_irrep_match.py
                              Compares the character of the near-null
                              subspace, class by class, with the two
                              candidate decompositions that share its trace
                              distribution, and finds eps_perm + (std3 x
                              det) (Numerical observation 18.6).

    a18_invariant_quartic_basis.py
                              Exact: the quartics invariant under that
                              representation form a space of dimension 5,
                              by power-sum character formulas, with the
                              basis {x0^4, x0^2 S2, x0 P3, S4, S22} found
                              independently by Reynolds projection
                              (prop:a18-invariant-dim).

    a18_fit_and_sos_check.py  Fits the five invariant coefficients from
                              seven directions at two step sizes (40-digit
                              volumes) and applies the sum-of-squares test
                              to the three fits (Numerical observation
                              18.8). Double-precision fits disagree by 100
                              to 700% between step sizes and are not used.
