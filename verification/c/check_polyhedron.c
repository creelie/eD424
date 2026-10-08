/*
 * check_polyhedron.c -- independent exact check of the vertex and edge data
 * of R(4) in ../data/vertices.txt and ../data/edges.txt.
 *
 * All arithmetic is on integers (__int128 with overflow traps); floating point
 * is used only to choose enumeration boxes, which are then widened by 2.
 *
 * Checks, for every vertex representative J_i:
 *   (V1) min_{k in M, k != 0} J_i[k] = 4 and the minimal vectors are the
 *        listed ones;
 *   (V2) the functionals X -> X[k] (k minimal) have rank 15, so J_i is a
 *        vertex;
 *   (V3) the extreme rays of C_i = {X : X[k] >= 0, k minimal} are exactly the
 *        listed edge directions at J_i: every extreme ray is the kernel of
 *        14 independent minimal vectors, so all 14-subsets are tried.
 * and for every edge (J_i, X):
 *   (E1) bounded, t* in {1, 2}: J' = J_i + t* X has minimum 4, its minimal
 *        vectors have rank 15, and J' = T^T J_j T for the listed T in Gamma;
 *   (E2) unbounded: X[(n, l)] = 2 (u.n - a l)(u.n - (a + 1) l) identically;
 *   (E3) unbounded edges at J_i fall into the orbits of Table 5.3 under the
 *        stabiliser of J_i: ../data/unbounded_orbits.txt gives, for each, a
 *        representative edge r and T in Gamma with T^T J_i T = J_i and
 *        T^T X_r T = X.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef __int128 i128;
typedef long long ll;

#define N 5
#define D 4
#define NC 15
#define MAXV 64
#define MAXE 4096
#define MAXK 64

static void fail(const char *msg, int a, int b)
{
    fprintf(stderr, "FAIL: %s (%d, %d)\n", msg, a, b);
    exit(1);
}

static i128 mul(i128 a, i128 b)
{
    i128 r;
    if (__builtin_mul_overflow(a, b, &r))
        fail("overflow", 0, 0);
    return r;
}

static i128 sub(i128 a, i128 b)
{
    i128 r;
    if (__builtin_sub_overflow(a, b, &r))
        fail("overflow", 0, 0);
    return r;
}

/* fraction-free Gaussian elimination: rank of an m x n integer matrix */
static int rank_int(i128 *A, int m, int n)
{
    i128 prev = 1;
    int r = 0;
    for (int c = 0; c < n && r < m; c++) {
        int p = -1;
        for (int i = r; i < m; i++)
            if (A[i * n + c] != 0) { p = i; break; }
        if (p < 0)
            continue;
        if (p != r)
            for (int j = 0; j < n; j++) { i128 t = A[r * n + j]; A[r * n + j] = A[p * n + j]; A[p * n + j] = t; }
        for (int i = r + 1; i < m; i++) {
            for (int j = c + 1; j < n; j++)
                A[i * n + j] = sub(mul(A[r * n + c], A[i * n + j]), mul(A[i * n + c], A[r * n + j])) / prev;
            A[i * n + c] = 0;
        }
        prev = A[r * n + c];
        r++;
    }
    return r;
}

/* determinant of an n x n integer matrix (Bareiss) */
static i128 det_int(i128 *A, int n)
{
    i128 prev = 1;
    int sign = 1;
    for (int c = 0; c < n; c++) {
        int p = -1;
        for (int i = c; i < n; i++)
            if (A[i * n + c] != 0) { p = i; break; }
        if (p < 0)
            return 0;
        if (p != c) {
            for (int j = 0; j < n; j++) { i128 t = A[c * n + j]; A[c * n + j] = A[p * n + j]; A[p * n + j] = t; }
            sign = -sign;
        }
        for (int i = c + 1; i < n; i++) {
            for (int j = c + 1; j < n; j++)
                A[i * n + j] = sub(mul(A[c * n + c], A[i * n + j]), mul(A[i * n + c], A[c * n + j])) / prev;
            A[i * n + c] = 0;
        }
        prev = A[c * n + c];
    }
    return sign * A[(n - 1) * n + n - 1];
}

static int IDXI[NC], IDXJ[NC];

static void coords(const int *k, i128 *row)
{
    for (int c = 0; c < NC; c++) {
        int i = IDXI[c], j = IDXJ[c];
        row[c] = (i128)k[i] * k[j] * (i == j ? 1 : 2);
    }
}

static ll qf(const ll J[N][N], const int *k)
{
    ll s = 0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            s += J[i][j] * k[i] * k[j];
    return s;
}

/* inverse diagonal of the Q-block and q = Q^{-1} r, sigma, in long double, for boxes */
static void boxdata(const ll J[N][N], long double qinvdiag[D], long double q[D], long double *sigma)
{
    long double A[D][D + 1 + D];
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < D; j++)
            A[i][j] = J[i][j];
        A[i][D] = J[i][D];
        for (int j = 0; j < D; j++)
            A[i][D + 1 + j] = (i == j);
    }
    for (int c = 0; c < D; c++) {
        int p = c;
        for (int i = c; i < D; i++)
            if (fabsl(A[i][c]) > fabsl(A[p][c])) p = i;
        for (int j = 0; j < 2 * D + 1; j++) { long double t = A[c][j]; A[c][j] = A[p][j]; A[p][j] = t; }
        for (int i = 0; i < D; i++)
            if (i != c) {
                long double f = A[i][c] / A[c][c];
                for (int j = 0; j < 2 * D + 1; j++)
                    A[i][j] -= f * A[c][j];
            }
    }
    long double s = J[D][D];
    for (int i = 0; i < D; i++) {
        q[i] = A[i][D] / A[i][i];
        qinvdiag[i] = A[i][D + 1 + i] / A[i][i];
        s -= J[i][D] * q[i];
    }
    *sigma = s;
}

/* all k = (n, l), l in {0, 1}, one of each +-pair, with J[k] <= 4; returns count, and the minimum */
static int small_vectors(const ll J[N][N], int out[][N], ll *minval)
{
    long double qid[D], q[D], sigma;
    boxdata(J, qid, q, &sigma);
    for (int i = 0; i < D; i++)
        if (!(qid[i] > 0))
            fail("Q-block not positive definite", 0, 0);
    int cnt = 0;
    *minval = 1LL << 60;
    for (int l = 0; l <= 1; l++) {
        long double rad = 4.0L - (l ? sigma : 0.0L);
        if (rad < -1e-9L)
            continue;
        if (rad < 0)
            rad = 0;
        int lo[D], hi[D];
        for (int i = 0; i < D; i++) {
            long double c = l ? -q[i] : 0.0L, w = sqrtl(rad * qid[i]);
            lo[i] = (int)floorl(c - w) - 2;
            hi[i] = (int)ceill(c + w) + 2;
        }
        int k[N];
        k[D] = l;
        for (k[0] = lo[0]; k[0] <= hi[0]; k[0]++)
            for (k[1] = lo[1]; k[1] <= hi[1]; k[1]++)
                for (k[2] = lo[2]; k[2] <= hi[2]; k[2]++)
                    for (k[3] = lo[3]; k[3] <= hi[3]; k[3]++) {
                        if (l == 0) {
                            int f = 0;
                            for (int i = 0; i < D && !f; i++)
                                if (k[i]) f = k[i] > 0 ? 1 : -1;
                            if (f <= 0)
                                continue; /* zero, or the negative of a listed vector */
                        }
                        ll v = qf(J, k);
                        if (v < *minval) *minval = v;
                        if (v <= 4) {
                            if (cnt >= MAXK) fail("too many small vectors", cnt, 0);
                            memcpy(out[cnt++], k, sizeof k);
                        }
                    }
    }
    return cnt;
}

static int rank_of(int vecs[][N], int m)
{
    i128 A[MAXK * NC];
    for (int i = 0; i < m; i++)
        coords(vecs[i], A + i * NC);
    return rank_int(A, m, NC);
}

static ll gcdll(ll a, ll b)
{
    if (a < 0) a = -a;
    if (b < 0) b = -b;
    while (b) { ll t = a % b; a = b; b = t; }
    return a;
}

/* ---------------------------------------------------------------- data */
static int nv, ne;
static ll VJ[MAXV][N][N];
static int VK[MAXV][MAXK][N], VNK[MAXV];
static int Efrom[MAXE], Et[MAXE], Eto[MAXE];
static ll EX[MAXE][N][N];
static ll ET[MAXE][N][N];
static int Eu[MAXE][D], Ea[MAXE];

static void read_data(const char *dir)
{
    char path[1024];
    snprintf(path, sizeof path, "%s/vertices.txt", dir);
    FILE *f = fopen(path, "r");
    if (!f) fail("cannot open vertices.txt", 0, 0);
    int id;
    while (fscanf(f, "%d", &id) == 1) {
        if (id != nv) fail("vertex numbering", id, nv);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                if (fscanf(f, "%lld", &VJ[nv][i][j]) != 1) fail("read", nv, 0);
        if (fscanf(f, "%d", &VNK[nv]) != 1) fail("read", nv, 1);
        for (int a = 0; a < VNK[nv]; a++)
            for (int i = 0; i < N; i++)
                if (fscanf(f, "%d", &VK[nv][a][i]) != 1) fail("read", nv, 2);
        nv++;
    }
    fclose(f);
    snprintf(path, sizeof path, "%s/edges.txt", dir);
    f = fopen(path, "r");
    if (!f) fail("cannot open edges.txt", 0, 0);
    while (fscanf(f, "%d", &id) == 1) {
        if (id != ne) fail("edge numbering", id, ne);
        if (fscanf(f, "%d", &Efrom[ne]) != 1) fail("read", ne, 0);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                if (fscanf(f, "%lld", &EX[ne][i][j]) != 1) fail("read", ne, 1);
        if (fscanf(f, "%d %d", &Et[ne], &Eto[ne]) != 2) fail("read", ne, 2);
        if (Eto[ne] >= 0) {
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    if (fscanf(f, "%lld", &ET[ne][i][j]) != 1) fail("read", ne, 3);
        } else {
            for (int i = 0; i < D; i++)
                if (fscanf(f, "%d", &Eu[ne][i]) != 1) fail("read", ne, 4);
            if (fscanf(f, "%d", &Ea[ne]) != 1) fail("read", ne, 5);
        }
        ne++;
    }
    fclose(f);
}

/* normalized 15-vector of a symmetric matrix */
static void tovec(const ll X[N][N], ll v[NC])
{
    for (int c = 0; c < NC; c++)
        v[c] = X[IDXI[c]][IDXJ[c]];
}

static int same_set_as_vertex(int k[N], int vi)
{
    for (int a = 0; a < VNK[vi]; a++) {
        int eq = 1, neg = 1;
        for (int i = 0; i < N; i++) {
            if (VK[vi][a][i] != k[i]) eq = 0;
            if (VK[vi][a][i] != -k[i]) neg = 0;
        }
        if (eq || neg)
            return 1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    const char *dir = argc > 1 ? argv[1] : "../data";
    int c = 0;
    for (int i = 0; i < N; i++) { IDXI[c] = i; IDXJ[c] = i; c++; }
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++) { IDXI[c] = i; IDXJ[c] = j; c++; }
    read_data(dir);
    printf("read %d vertex representatives and %d edges\n", nv, ne);

    long total_rays = 0;
    for (int vi = 0; vi < nv; vi++) {
        int ks[MAXK][N];
        ll mv;
        int m = small_vectors(VJ[vi], ks, &mv);
        if (mv != 4) fail("vertex minimum is not 4", vi, (int)mv);
        if (m != VNK[vi]) fail("number of minimal vectors differs", vi, m);
        for (int a = 0; a < m; a++)
            if (!same_set_as_vertex(ks[a], vi)) fail("minimal vector not listed", vi, a);
        if (rank_of(ks, m) != NC) fail("not a vertex", vi, 0);

        /* (V3): all extreme rays from 14-subsets */
        static ll rays[200000][NC];
        int nr = 0;
        i128 rows[MAXK][NC];
        for (int a = 0; a < m; a++)
            coords(ks[a], rows[a]);
        int idx[14];
        for (int i = 0; i < 14; i++) idx[i] = i;
        while (1) {
            i128 A[14 * NC];
            for (int i = 0; i < 14; i++)
                memcpy(A + i * NC, rows[idx[i]], sizeof(i128) * NC);
            i128 B[14 * NC];
            memcpy(B, A, sizeof A);
            if (rank_int(B, 14, NC) == 14) {
                ll x[NC];
                ll g = 0;
                for (int j = 0; j < NC; j++) {
                    i128 M[14 * 14];
                    for (int i = 0; i < 14; i++) {
                        int cc = 0;
                        for (int jj = 0; jj < NC; jj++)
                            if (jj != j) M[i * 14 + cc++] = A[i * NC + jj];
                    }
                    i128 d = det_int(M, 14);
                    if (j % 2) d = -d;
                    /* store reduced later; bound check */
                    if (d > ((i128)1 << 62) || d < -((i128)1 << 62)) fail("minor too large", vi, j);
                    x[j] = (ll)d;
                    g = gcdll(g, x[j]);
                }
                for (int j = 0; j < NC; j++) x[j] /= g;
                /* sign: X[k] >= 0 for all minimal k, or <= 0 */
                int pos = 1, neg = 1;
                for (int a = 0; a < m; a++) {
                    i128 s = 0;
                    for (int j = 0; j < NC; j++) s += rows[a][j] * x[j];
                    if (s < 0) pos = 0;
                    if (s > 0) neg = 0;
                }
                if (pos || neg) {
                    if (!pos)
                        for (int j = 0; j < NC; j++) x[j] = -x[j];
                    int seen = 0;
                    for (int r = 0; r < nr && !seen; r++)
                        if (!memcmp(rays[r], x, sizeof x)) seen = 1;
                    if (!seen) {
                        if (nr >= 200000) fail("too many rays", vi, nr);
                        memcpy(rays[nr++], x, sizeof x);
                    }
                }
            }
            int i = 13;
            while (i >= 0 && idx[i] == m - 14 + i) i--;
            if (i < 0) break;
            idx[i]++;
            for (int j = i + 1; j < 14; j++) idx[j] = idx[j - 1] + 1;
        }
        /* compare with the listed edges at vi */
        int listed = 0;
        for (int e = 0; e < ne; e++) {
            if (Efrom[e] != vi) continue;
            listed++;
            ll v[NC];
            tovec(EX[e], v);
            ll g = 0;
            for (int j = 0; j < NC; j++) g = gcdll(g, v[j]);
            for (int j = 0; j < NC; j++) v[j] /= g;
            int found = 0;
            for (int r = 0; r < nr && !found; r++)
                if (!memcmp(rays[r], v, sizeof v)) found = 1;
            if (!found) fail("listed edge direction is not an extreme ray", vi, e);
        }
        if (listed != nr) fail("number of extreme rays differs from the list", vi, nr);
        total_rays += nr;
        printf("vertex %d: minimum 4 on %d pairs, rank 15, %d extreme rays (all listed)\n", vi, m, nr);
    }
    if (total_rays != ne) fail("edges not accounted for", (int)total_rays, ne);

    int nb = 0, nu = 0;
    for (int e = 0; e < ne; e++) {
        int vi = Efrom[e];
        if (Eto[e] >= 0) {
            ll J2[N][N], J3[N][N];
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    J2[i][j] = VJ[vi][i][j] + Et[e] * EX[e][i][j];
            int ks[MAXK][N];
            ll mv;
            int m = small_vectors(J2, ks, &mv);
            if (mv != 4) fail("end point not in R(4) with minimum 4", e, (int)mv);
            if (rank_of(ks, m) != NC) fail("end point is not a vertex", e, m);
            /* T in Gamma: bottom row (0,0,0,0,+-1), det U = +-1 */
            for (int j = 0; j < D; j++)
                if (ET[e][D][j] != 0) fail("T not in Gamma", e, j);
            if (ET[e][D][D] != 1 && ET[e][D][D] != -1) fail("T not in Gamma", e, D);
            i128 U[D * D];
            for (int i = 0; i < D; i++)
                for (int j = 0; j < D; j++) U[i * D + j] = ET[e][i][j];
            i128 du = det_int(U, D);
            if (du != 1 && du != -1) fail("det U is not +-1", e, (int)du);
            const ll(*Jj)[N] = VJ[Eto[e]];
            for (int a = 0; a < N; a++)
                for (int b = 0; b < N; b++) {
                    ll s = 0;
                    for (int i = 0; i < N; i++)
                        for (int j = 0; j < N; j++)
                            s += ET[e][i][a] * Jj[i][j] * ET[e][j][b];
                    J3[a][b] = s;
                }
            if (memcmp(J2, J3, sizeof J2)) fail("T^T J_j T differs from the end point", e, Eto[e]);
            nb++;
        } else {
            const int *u = Eu[e];
            int a = Ea[e];
            ll W[N][N];
            int w[N] = {u[0], u[1], u[2], u[3], 0};
            /* X[(n,l)] = 2 (w.k - a l)(w.k - (a+1) l): X = (p q^T + q p^T) with p = w - a f, q = w - (a+1) f */
            int p[N], q[N];
            for (int i = 0; i < N; i++) {
                p[i] = w[i] - (i == D ? a : 0);
                q[i] = w[i] - (i == D ? a + 1 : 0);
            }
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    W[i][j] = (ll)p[i] * q[j] + (ll)q[i] * p[j];
            if (memcmp(W, EX[e], sizeof W)) fail("unbounded edge certificate", e, 0);
            nu++;
        }
    }
    printf("%d bounded edges: end points are vertices in the listed classes; %d unbounded edges certified\n", nb, nu);

    /* (E3) */
    {
        char path[1024];
        snprintf(path, sizeof path, "%s/unbounded_orbits.txt", dir);
        FILE *f = fopen(path, "r");
        if (!f) fail("cannot open unbounded_orbits.txt", 0, 0);
        int e, vi, r, norb = 0, reps[64], nreps = 0;
        while (fscanf(f, "%d %d %d", &e, &vi, &r) == 3) {
            ll T[N][N];
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    if (fscanf(f, "%lld", &T[i][j]) != 1) fail("read orbit", e, 0);
            if (Eto[e] >= 0 || Eto[r] >= 0 || Efrom[e] != vi || Efrom[r] != vi) fail("orbit record", e, r);
            for (int j = 0; j < D; j++)
                if (T[D][j] != 0) fail("orbit T not in Gamma", e, j);
            if (T[D][D] != 1 && T[D][D] != -1) fail("orbit T not in Gamma", e, D);
            i128 U[D * D];
            for (int i = 0; i < D; i++)
                for (int j = 0; j < D; j++) U[i * D + j] = T[i][j];
            i128 du = det_int(U, D);
            if (du != 1 && du != -1) fail("orbit det U", e, (int)du);
            for (int a = 0; a < N; a++)
                for (int b = 0; b < N; b++) {
                    ll s1 = 0, s2 = 0;
                    for (int i = 0; i < N; i++)
                        for (int j = 0; j < N; j++) {
                            s1 += T[i][a] * VJ[vi][i][j] * T[j][b];
                            s2 += T[i][a] * EX[r][i][j] * T[j][b];
                        }
                    if (s1 != VJ[vi][a][b]) fail("T does not fix J_i", e, vi);
                    if (s2 != EX[e][a][b]) fail("T does not map the representative ray", e, r);
                }
            int seen = 0;
            for (int k = 0; k < nreps; k++)
                if (reps[k] == r) seen = 1;
            if (!seen) reps[nreps++] = r;
            norb++;
        }
        fclose(f);
        if (norb != nu) fail("orbit records do not cover the unbounded edges", norb, nu);
        printf("%d unbounded edges in %d orbits under the stabilisers; representatives:", norb, nreps);
        for (int k = 0; k < nreps; k++) printf(" %d", reps[k]);
        printf("\n");
    }
    printf("ALL CHECKS PASSED\n");
    return 0;
}
