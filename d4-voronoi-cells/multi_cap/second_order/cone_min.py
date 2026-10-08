"""
m = min xi^T H xi over the first-order feasible cone, normalised by sum delta = 1:

    (Lambda tau)_ij <= (eta_i + eta_j)/2      for the 96 tight pairs   (packing, first order)
    eta_i >= 0                                                       (centres at distance >= 2)
    sum_i 2 eta_i = 1                                                (sum delta = 1)

Then the second-order model of the cell volume reads
    vol - 8  =  (2/3) S + (m/2) S^2 + O(S^3),   S = sum delta,
and it stays positive up to S* = 4 / (3 |m|).

Two computations: many local minimisations (SLSQP) for an upper bound on m
(a feasible point), and the Shor SDP relaxation with RLT products for a lower
bound.  If they agree, m is known.
"""
import numpy as np, sys
from scipy.optimize import minimize
sys.path.insert(0, '.')
import hessian as Hm

H = np.load('H.npy')
U, TIGHT, BASIS = Hm.U, Hm.TIGHT, Hm.BASIS

# linear map xi -> (Lambda tau)_ij - (eta_i + eta_j)/2   (96 rows)
A = np.zeros((96, 96))
for r, (i, j) in enumerate(TIGHT):
    A[r, 3 * j:3 * j + 3] += BASIS[j].T @ U[i]       # <u_i, tau_j>
    A[r, 3 * i:3 * i + 3] += BASIS[i].T @ U[j]       # <u_j, tau_i>
    A[r, 72 + i] -= 0.5
    A[r, 72 + j] -= 0.5
c_sum = np.zeros(96); c_sum[72:] = 2.0                # sum delta = 2 sum eta


def value(xi):
    return xi @ H @ xi


def local_min(x0):
    cons = [{'type': 'ineq', 'fun': lambda x: -(A @ x)},          # A x <= 0
            {'type': 'ineq', 'fun': lambda x: x[72:]},            # eta >= 0
            {'type': 'eq', 'fun': lambda x: c_sum @ x - 1.0}]
    res = minimize(value, x0, jac=lambda x: 2 * H @ x, constraints=cons,
                   method='SLSQP', options=dict(maxiter=2000, ftol=1e-13))
    return res


if __name__ == '__main__':
  rng = np.random.default_rng(0)
  best = np.inf; bestx = None
  starts = []
  # structured starts: one centre out; all out evenly; random
  x = np.zeros(96); x[72] = 0.5; starts.append(x)
  x = np.zeros(96); x[72:] = 1 / 48; starts.append(x)
  for _ in range(60):
      x = rng.standard_normal(96) * 0.05; x[72:] = np.abs(rng.standard_normal(24)); x[72:] /= 2 * x[72:].sum()
      starts.append(x)
  for k, x0 in enumerate(starts):
      r = local_min(x0)
      feas = max((A @ r.x).max(), -(r.x[72:]).min(), abs(c_sum @ r.x - 1))
      if feas < 1e-7 and r.fun < best:
          best, bestx = r.fun, r.x.copy()
  print(f"upper bound on m (best feasible point over {len(starts)} starts): {best:.6f}")
  tau, eta = Hm.unpack(bestx)
  print(f"   at it: ||tau|| = {np.linalg.norm(tau):.4f}, delta = 2 eta, max delta = {2*eta.max():.4f}, "
        f"number of centres with delta > 1e-3: {(2*eta > 1e-3).sum()}")
  print(f"   one-centre ray value: {value(starts[0]):.6f};  even ray value: {value(starts[1]):.6f}")

  # second-order model crossovers
  for name, mm in (("worst found", best), ("one-centre ray", value(starts[0]))):
      print(f"   S* = 4/(3|m|) with m = {mm:.4f} ({name}): {4/(3*abs(mm)):.4f}")

  # ---------------------------------------------------------------- Shor + RLT lower bound
  try:
      import cvxpy as cp
      n = 96
      X = cp.Variable((n, n), PSD=True)
      x = cp.Variable(n)
      M = cp.bmat([[X, cp.reshape(x, (n, 1), order='C')], [cp.reshape(x, (1, n), order='C'), np.ones((1, 1))]])
      cons = [M >> 0, A @ x <= 0, x[72:] >= 0, c_sum @ x == 1]
      # RLT: products of the constraint (1 - c_sum x) = 0 with everything, and pairwise products of the
      # homogeneous inequalities  (-A x)_r (-A x)_s >= 0, (-A x)_r eta_i >= 0, eta_i eta_j >= 0
      Bm = np.vstack([-A, np.eye(96)[72:]])      # 120 rows, all >= 0 on the cone
      cons += [Bm @ X @ Bm.T >= 0]
      cons += [X @ c_sum == x]                   # (c_sum x = 1) * x
      prob = cp.Problem(cp.Minimize(cp.trace(H @ X)), cons)
      prob.solve(solver='CLARABEL', tol_gap_abs=1e-7, tol_gap_rel=1e-7, tol_feas=1e-8, max_iter=300,
                 static_regularization_constant=1e-7)
      print(f"\nShor+RLT lower bound on m: {prob.value:.6f}   ({prob.status})")
  except Exception as e:
      print("SDP relaxation failed:", e)
  np.save('best_xi.npy', bestx)