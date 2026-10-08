/*
 * psker.c -- the integral of step 3 of the LLM24 verification, without the
 * 128 GB.
 *
 * For a signature lambda of O(4) and admissible indices k1, k2 the quantity
 * wanted is
 *
 *     P(S)_{k1,k2} = int_{O(4)} rho_{0,k1}(omega gamma eps)
 *                               rho_{0,k2}(omega gamma S) dgamma,
 *
 * a polynomial in the seven free entries of S.  Reading the authors' code the
 * two factors are products of powers of single entries,
 *
 *     rho_A = det(A)^l2 A11^(m-k1) A12^k1,
 *     rho_B = B11^(l2+m-k2) B22^l2 B12^k2,     m = l1 - l2,
 *
 * with A = omega gamma eps and B = omega gamma S, and the second splits as
 * U * V with U = B11^(l2+m-k2) B12^k2 carrying rows 0 and 2 of gamma and
 * V = B22^l2 carrying rows 1 and 3.  The authors expand rho_A * rho_B in full
 * and store it keyed by the exponent matrix of gamma before integrating; that
 * stored expansion is what needs the memory.
 *
 * Integration is term by term, so the expansion never has to exist.  This walks
 * the triples (a, u, v) and adds each one straight into the coefficient of P(S)
 * it belongs to, of which there are at most a few tens of thousands.  Only
 * triples whose exponent matrix has every row sum and every column sum even
 * survive, and the loop is organised by parity class so the rest are skipped.
 *
 * The monomial integral is the Gorin-Lopez recursion in exact rational
 * arithmetic, peeling one column at a time, memoised on the canonical form of
 * the exponent matrix.  Values for matrices of at most three columns are kept
 * for the whole run; the four column ones are dropped between entries.
 *
 * Build: gcc -O2 -o psker psker.c -lgmp -I/usr/include/x86_64-linux-gnu
 * Use:   psker spec.bin out.txt
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <gmp.h>

#define NG 16
#define NS 7
#define DIM 4
#define MAXDEG 40

/* ----------------------------------------------------------------- ordering */
static int less_key(const uint8_t *v, int n, const uint8_t *w)
{
    int sv = 0, sw = 0, mv = 0, mw = 0, i;
    for (i = 0; i < n; i++) { sv += v[i]; if (v[i] > mv) mv = v[i]; }
    for (i = 0; i < n; i++) { sw += w[i]; if (w[i] > mw) mw = w[i]; }
    if (sv != sw) return sv < sw;
    if (mv != mw) return mv < mw;
    for (i = 0; i < n; i++) if (v[i] != w[i]) return v[i] < w[i];
    return 0;
}

typedef struct { uint8_t nr, nc, m[NG]; } Mat;

static void canon(const uint8_t *in, int nr0, int nc0, Mat *out)
{
    uint8_t buf[NG], t[NG], col[4], row[4];
    int nr = 0, nc = 0, i, j, k, keepr[4], keepc[4];
    for (i = 0; i < nr0; i++) {
        int s = 0;
        for (j = 0; j < nc0; j++) s += in[i * nc0 + j];
        if (s) keepr[nr++] = i;
    }
    for (j = 0; j < nc0; j++) {
        int s = 0;
        for (i = 0; i < nr0; i++) s += in[i * nc0 + j];
        if (s) keepc[nc++] = j;
    }
    memset(out->m, 0, NG);
    if (nr == 0 || nc == 0) { out->nr = 1; out->nc = 1; return; }
    for (i = 0; i < nr; i++)
        for (j = 0; j < nc; j++)
            buf[i * nc + j] = in[keepr[i] * nc0 + keepc[j]];
    if (nr < nc) {
        for (i = 0; i < nr; i++)
            for (j = 0; j < nc; j++) t[j * nr + i] = buf[i * nc + j];
        memcpy(buf, t, nr * nc);
        i = nr; nr = nc; nc = i;
    }
    for (i = 1; i < nc; i++) {                 /* columns decreasing */
        for (k = 0; k < nr; k++) col[k] = buf[k * nc + i];
        j = i - 1;
        while (j >= 0) {
            uint8_t other[4];
            for (k = 0; k < nr; k++) other[k] = buf[k * nc + j];
            if (!less_key(other, nr, col)) break;
            for (k = 0; k < nr; k++) buf[k * nc + j + 1] = other[k];
            j--;
        }
        for (k = 0; k < nr; k++) buf[k * nc + j + 1] = col[k];
    }
    for (i = 1; i < nr; i++) {                 /* rows increasing */
        memcpy(row, buf + i * nc, nc);
        j = i - 1;
        while (j >= 0 && less_key(row, nc, buf + j * nc)) {
            memcpy(buf + (j + 1) * nc, buf + j * nc, nc);
            j--;
        }
        memcpy(buf + (j + 1) * nc, row, nc);
    }
    out->nr = (uint8_t)nr; out->nc = (uint8_t)nc;
    memcpy(out->m, buf, nr * nc);
}

/* -------------------------------------------------------------- memo tables */
typedef struct {
    uint64_t *k0, *k1;
    uint8_t *kn;               /* nr * 8 + nc, 0 = empty slot */
    int32_t *val;
    size_t mask, used;
    mpq_t *vals;               /* each table owns its values */
    size_t nvals, capvals;
} Table;

static int32_t newval(Table *t)
{
    if (t->nvals == t->capvals) {
        t->capvals = t->capvals ? t->capvals * 2 : 4096;
        t->vals = realloc(t->vals, t->capvals * sizeof(mpq_t));
        if (!t->vals) { fprintf(stderr, "out of memory (values)\n"); exit(3); }
    }
    mpq_init(t->vals[t->nvals]);
    return (int32_t)t->nvals++;
}

static void table_alloc(Table *t, size_t cap)
{
    size_t n = 1;
    while (n < cap) n <<= 1;
    t->k0 = calloc(n, 8); t->k1 = calloc(n, 8);
    t->kn = calloc(n, 1); t->val = malloc(n * 4);
    if (!t->k0 || !t->k1 || !t->kn || !t->val) {
        fprintf(stderr, "out of memory (table)\n"); exit(3);
    }
    t->mask = n - 1; t->used = 0;
}

static void table_init(Table *t, size_t cap)
{
    memset(t, 0, sizeof(*t));
    table_alloc(t, cap);
}

static void table_free(Table *t)
{
    size_t i;
    free(t->k0); free(t->k1); free(t->kn); free(t->val);
    for (i = 0; i < t->nvals; i++) mpq_clear(t->vals[i]);
    free(t->vals);
    memset(t, 0, sizeof(*t));
}

static uint64_t mix(uint64_t a, uint64_t b, uint8_t n)
{
    uint64_t h = a * 0x9E3779B97F4A7C15ULL ^ (b + 0x165667B19E3779F9ULL)
                 ^ ((uint64_t)n << 47);
    h ^= h >> 30; h *= 0xBF58476D1CE4E5B9ULL;
    h ^= h >> 27; h *= 0x94D049BB133111EBULL;
    h ^= h >> 31;
    return h;
}

static void pack(const Mat *M, uint64_t *a, uint64_t *b, uint8_t *n)
{
    uint64_t x = 0, y = 0;
    int i;
    for (i = 0; i < 8; i++) x |= (uint64_t)M->m[i] << (8 * i);
    for (i = 8; i < 16; i++) y |= (uint64_t)M->m[i] << (8 * (i - 8));
    *a = x; *b = y; *n = (uint8_t)(M->nr * 8 + M->nc);
}

static size_t table_slot(Table *t, uint64_t a, uint64_t b, uint8_t n, int *found)
{
    size_t i = mix(a, b, n) & t->mask;
    while (t->kn[i]) {
        if (t->k0[i] == a && t->k1[i] == b && t->kn[i] == n) {
            *found = 1; return i;
        }
        i = (i + 1) & t->mask;
    }
    *found = 0; return i;
}

static void table_grow(Table *t)
{
    Table nt;
    size_t i;
    memset(&nt, 0, sizeof(nt));
    table_alloc(&nt, (t->mask + 1) * 2);
    for (i = 0; i <= t->mask; i++) {
        if (t->kn[i]) {
            int f;
            size_t j = table_slot(&nt, t->k0[i], t->k1[i], t->kn[i], &f);
            nt.k0[j] = t->k0[i]; nt.k1[j] = t->k1[i];
            nt.kn[j] = t->kn[i]; nt.val[j] = t->val[i];
            nt.used++;
        }
    }
    free(t->k0); free(t->k1); free(t->kn); free(t->val);
    t->k0 = nt.k0; t->k1 = nt.k1; t->kn = nt.kn; t->val = nt.val;
    t->mask = nt.mask; t->used = nt.used;
}

/* ------------------------------------------------------------ compositions */
/* all vectors of length n (1..3) summing to tot, as flat arrays */
static uint8_t *COMPS[4][MAXDEG + 1];
static int NCOMPS[4][MAXDEG + 1];

static void build_comps(int n, int tot)
{
    int cnt = 0, cap = 1024;
    uint8_t *out = malloc(cap * n);
    int a, b;
    if (n == 1) {
        out[0] = (uint8_t)tot; cnt = 1;
    } else if (n == 2) {
        for (a = 0; a <= tot; a++) {
            out[2 * cnt] = (uint8_t)a; out[2 * cnt + 1] = (uint8_t)(tot - a);
            cnt++;
        }
    } else {
        for (a = 0; a <= tot; a++)
            for (b = 0; b <= tot - a; b++) {
                if (cnt * 3 + 3 > cap * 3) { cap *= 2; out = realloc(out, cap * 3); }
                out[3 * cnt] = (uint8_t)a; out[3 * cnt + 1] = (uint8_t)b;
                out[3 * cnt + 2] = (uint8_t)(tot - a - b);
                cnt++;
            }
    }
    COMPS[n][tot] = out; NCOMPS[n][tot] = cnt;
}

static void comps(int n, int tot, uint8_t **p, int *cnt)
{
    if (!COMPS[n][tot]) build_comps(n, tot);
    *p = COMPS[n][tot]; *cnt = NCOMPS[n][tot];
}

/* ------------------------------------------------------------- the integral */
static Table SMALL, BIG;
static mpq_t POCH_HALF[MAXDEG + 1], POCH_D2[MAXDEG + 1];
static mpz_t FACT[MAXDEG + 1];

static void poch_q(mpq_t r, const mpq_t z, int n)
{
    mpq_t t, u;
    int k;
    mpq_init(t); mpq_init(u);
    mpq_set_ui(r, 1, 1);
    for (k = 0; k < n; k++) {
        mpq_set_si(u, k, 1);
        mpq_add(t, z, u);
        mpq_mul(r, r, t);
    }
    mpq_clear(t); mpq_clear(u);
}

static void int_row(mpq_t out, const uint8_t *m, int n)
{
    int i, s = 0;
    for (i = 0; i < n; i++) {
        if (m[i] & 1) { mpq_set_ui(out, 0, 1); return; }
        s += m[i];
    }
    mpq_set_ui(out, 1, 1);
    for (i = 0; i < n; i++)
        if (m[i]) mpq_mul(out, out, POCH_HALF[m[i] / 2]);
    mpq_div(out, out, POCH_D2[s / 2]);
}

static void int_canon(mpq_t out, const Mat *M);

static void int_mat(mpq_t out, const uint8_t *in, int nr, int nc)
{
    int i, j;
    Mat C;
    for (i = 0; i < nr; i++) {
        int s = 0;
        for (j = 0; j < nc; j++) s += in[i * nc + j];
        if (s & 1) { mpq_set_ui(out, 0, 1); return; }
    }
    for (j = 0; j < nc; j++) {
        int s = 0;
        for (i = 0; i < nr; i++) s += in[i * nc + j];
        if (s & 1) { mpq_set_ui(out, 0, 1); return; }
    }
    canon(in, nr, nc, &C);
    int_canon(out, &C);
}

static void int_canon(mpq_t out, const Mat *M)
{
    int nr = M->nr, nc = M->nc, rl = nc - 1, i, j;
    uint64_t ka, kb; uint8_t kname;
    int found, sumlast = 0;
    Table *T;
    size_t slot;
    uint8_t last[4], head[NG], k[4], K[16], colsum[4], sub[NG];
    uint8_t *clist[4]; int ccnt[4], cidx[4];
    mpq_t s, si, temp, t1, z1, z2, diff, p1, p2, p3;
    mpz_t mult, binom, z;

    pack(M, &ka, &kb, &kname);
    T = (nc >= 4) ? &BIG : &SMALL;
    slot = table_slot(T, ka, kb, kname, &found);
    if (found) { mpq_set(out, T->vals[T->val[slot]]); return; }

    for (i = 0; i < nr; i++) last[i] = M->m[i * nc + nc - 1];
    for (i = 0; i < nr; i++) sumlast += last[i];

    mpq_init(s);
    if (nc == 1) {
        int_row(s, last, nr);
    } else if (sumlast % 2 == 0) {
        for (i = 0; i < nr; i++)
            for (j = 0; j < rl; j++) head[i * rl + j] = M->m[i * nc + j];
        mpq_init(si); mpq_init(temp); mpq_init(t1);
        mpq_init(z1); mpq_init(z2); mpq_init(diff);
        mpq_init(p1); mpq_init(p2); mpq_init(p3);
        mpz_init(mult); mpz_init(binom); mpz_init(z);
        mpq_set_ui(z1, DIM, 2); mpq_canonicalize(z1);
        mpq_set_ui(z2, rl, 2); mpq_canonicalize(z2);
        mpq_sub(diff, z1, z2);

        for (i = 0; i < nr; i++) k[i] = 0;
        while (1) {
            int carry, sumk = 0;
            for (i = 0; i < nr; i++) sumk += k[i];
            mpq_set_ui(si, 0, 1);
            for (i = 0; i < nr; i++) {
                comps(rl, last[i] - k[i], &clist[i], &ccnt[i]);
                cidx[i] = 0;
            }
            while (1) {
                for (i = 0; i < nr; i++)
                    memcpy(K + i * rl, clist[i] + cidx[i] * rl, rl);
                for (j = 0; j < rl; j++) {
                    int c = 0;
                    for (i = 0; i < nr; i++) c += K[i * rl + j];
                    colsum[j] = (uint8_t)c;
                }
                int_row(temp, colsum, rl);
                if (mpq_sgn(temp)) {
                    for (i = 0; i < nr; i++)
                        for (j = 0; j < rl; j++)
                            sub[i * rl + j] = head[i * rl + j] + K[i * rl + j];
                    int_mat(t1, sub, nr, rl);
                    if (mpq_sgn(t1)) {
                        mpz_set_ui(mult, 1);
                        for (i = 0; i < nr; i++) {
                            mpz_mul(mult, mult, FACT[last[i] - k[i]]);
                            for (j = 0; j < rl; j++)
                                mpz_divexact(mult, mult, FACT[K[i * rl + j]]);
                        }
                        mpq_mul(temp, temp, t1);
                        mpq_set_z(t1, mult);
                        mpq_mul(temp, temp, t1);
                        mpq_add(si, si, temp);
                    }
                }
                for (i = nr - 1; i >= 0; i--) {
                    if (++cidx[i] < ccnt[i]) break;
                    cidx[i] = 0;
                }
                if (i < 0) break;
            }
            if (mpq_sgn(si)) {
                int a = sumlast / 2, b = sumk / 2;
                int_row(temp, k, nr);
                if (mpq_sgn(temp)) {
                    mpz_set_ui(binom, 1);
                    for (i = 0; i < nr; i++) {
                        mpz_bin_uiui(z, last[i], k[i]);
                        mpz_mul(binom, binom, z);
                    }
                    poch_q(p1, z1, b);
                    poch_q(p2, z1, a - b);
                    poch_q(p3, diff, a);
                    mpq_mul(t1, p1, p2);
                    mpq_div(t1, t1, p3);
                    if ((a - b) & 1) mpq_neg(t1, t1);
                    mpq_mul(temp, temp, t1);
                    mpq_set_z(t1, binom);
                    mpq_mul(temp, temp, t1);
                    mpq_mul(temp, temp, si);
                    mpq_add(s, s, temp);
                }
            }
            carry = 1;
            for (i = nr - 1; i >= 0 && carry; i--) {
                if (k[i] + 2 <= last[i]) { k[i] += 2; carry = 0; }
                else k[i] = 0;
            }
            if (carry) break;
        }
        mpq_clear(si); mpq_clear(temp); mpq_clear(t1);
        mpq_clear(z1); mpq_clear(z2); mpq_clear(diff);
        mpq_clear(p1); mpq_clear(p2); mpq_clear(p3);
        mpz_clear(mult); mpz_clear(binom); mpz_clear(z);
    }

    {
        int32_t vi = newval(T);
        mpq_set(T->vals[vi], s);
        slot = table_slot(T, ka, kb, kname, &found);
        T->k0[slot] = ka; T->k1[slot] = kb; T->kn[slot] = kname;
        T->val[slot] = vi;
        T->used++;
        if (T->used * 4 > (T->mask + 1) * 3) table_grow(T);
        mpq_set(out, s);
    }
    mpq_clear(s);
}

/* ------------------------------------------------------------------ driver */
typedef struct { uint8_t g[NG], sv[NS], cls; int64_t c; } Term;

static long BINOM[80][80];

static void binom_init(void)
{
    int i, j;
    for (i = 0; i < 80; i++) {
        BINOM[i][0] = 1;
        for (j = 1; j <= i; j++)
            BINOM[i][j] = BINOM[i - 1][j - 1] + (j <= i - 1 ? BINOM[i - 1][j] : 0);
    }
}

static int rank_comp(const uint8_t *b, int n, int tot)
{
    int r = 0, i, v, rest = tot;
    for (i = 0; i < n - 1; i++) {
        for (v = 0; v < b[i]; v++)
            r += (int)BINOM[rest - v + n - i - 2][n - i - 2];
        rest -= b[i];
    }
    return r;
}

static void unrank_comp(int r, int n, int tot, uint8_t *b)
{
    int i, v, rest = tot;
    for (i = 0; i < n - 1; i++) {
        v = 0;
        while (1) {
            int c = (int)BINOM[rest - v + n - i - 2][n - i - 2];
            if (r < c) break;
            r -= c; v++;
        }
        b[i] = (uint8_t)v; rest -= v;
    }
    b[n - 1] = (uint8_t)rest;
}

static uint8_t classof(const uint8_t *g)
{
    int i, j, c = 0;
    for (i = 0; i < 4; i++) {
        int s = 0;
        for (j = 0; j < 4; j++) s += g[4 * i + j];
        if (s & 1) c |= 1 << i;
    }
    for (j = 0; j < 4; j++) {
        int s = 0;
        for (i = 0; i < 4; i++) s += g[4 * i + j];
        if (s & 1) c |= 1 << (4 + j);
    }
    return (uint8_t)c;
}

int main(int argc, char **argv)
{
    FILE *fi, *fo;
    int i, j, nent, e;
    mpq_t half, d2;

    if (argc < 3) { fprintf(stderr, "usage: psker spec.bin out.txt\n"); return 2; }
    binom_init();
    mpq_init(half); mpq_set_ui(half, 1, 2);
    mpq_init(d2); mpq_set_ui(d2, DIM, 2); mpq_canonicalize(d2);
    for (i = 0; i <= MAXDEG; i++) {
        mpq_init(POCH_HALF[i]); poch_q(POCH_HALF[i], half, i);
        mpq_init(POCH_D2[i]); poch_q(POCH_D2[i], d2, i);
        mpz_init(FACT[i]); mpz_fac_ui(FACT[i], i);
    }
    mpq_clear(half); mpq_clear(d2);
    table_init(&SMALL, 1 << 18);

    fi = fopen(argv[1], "rb");
    fo = fopen(argv[2], "w");
    if (!fi || !fo) { fprintf(stderr, "cannot open files\n"); return 2; }
    if (fread(&nent, 4, 1, fi) != 1) return 2;

    for (e = 0; e < nent; e++) {
        int hdr[7], na, nu, nv, deg, nbeta, a_i, u_i, v_i, q;
        Term *A, *U, *V;
        mpz_t *num, den, gq, tz, cav;
        mpq_t val;
        long long visited = 0, hits = 0;
        uint8_t beta[NS], base[NG], ee[NG];

        if (fread(hdr, 4, 7, fi) != 7) { fprintf(stderr, "short read\n"); return 2; }
        na = hdr[4]; nu = hdr[5]; nv = hdr[6];
        deg = hdr[0] + hdr[1];
        nbeta = (int)BINOM[deg + NS - 1][NS - 1];
        A = malloc(sizeof(Term) * (na ? na : 1));
        U = malloc(sizeof(Term) * (nu ? nu : 1));
        V = malloc(sizeof(Term) * (nv ? nv : 1));
        for (j = 0; j < na; j++) {
            if (fread(A[j].g, 1, NG, fi) != NG) return 2;
            if (fread(&A[j].c, 8, 1, fi) != 1) return 2;
            A[j].cls = classof(A[j].g);
        }
        for (j = 0; j < nu; j++) {
            if (fread(U[j].g, 1, NG, fi) != NG) return 2;
            if (fread(U[j].sv, 1, NS, fi) != NS) return 2;
            if (fread(&U[j].c, 8, 1, fi) != 1) return 2;
            U[j].cls = classof(U[j].g);
        }
        for (j = 0; j < nv; j++) {
            if (fread(V[j].g, 1, NG, fi) != NG) return 2;
            if (fread(V[j].sv, 1, NS, fi) != NS) return 2;
            if (fread(&V[j].c, 8, 1, fi) != 1) return 2;
            V[j].cls = classof(V[j].g);
        }

        table_init(&BIG, 1 << 20);
        num = malloc(sizeof(mpz_t) * nbeta);
        for (j = 0; j < nbeta; j++) mpz_init(num[j]);
        mpz_init(den); mpz_set_ui(den, 1);
        mpz_init(gq); mpz_init(tz); mpz_init(cav);
        mpq_init(val);

        for (a_i = 0; a_i < na; a_i++) {
            for (v_i = 0; v_i < nv; v_i++) {
                uint8_t want = A[a_i].cls ^ V[v_i].cls;
                mpz_set_si(cav, (long)A[a_i].c);
                mpz_mul_si(cav, cav, (long)V[v_i].c);
                for (j = 0; j < NG; j++) base[j] = A[a_i].g[j] + V[v_i].g[j];
                for (u_i = 0; u_i < nu; u_i++) {
                    int idx;
                    if (U[u_i].cls != want) continue;
                    for (j = 0; j < NG; j++) ee[j] = base[j] + U[u_i].g[j];
                    int_mat(val, ee, 4, 4);
                    visited++;
                    if (!mpq_sgn(val)) continue;
                    hits++;
                    if (mpz_cmp_ui(mpq_denref(val), 1) != 0) {
                        mpz_gcd(gq, den, mpq_denref(val));
                        mpz_divexact(tz, mpq_denref(val), gq);
                        if (mpz_cmp_ui(tz, 1) != 0) {
                            for (q = 0; q < nbeta; q++) mpz_mul(num[q], num[q], tz);
                            mpz_mul(den, den, tz);
                        }
                    }
                    for (j = 0; j < NS; j++)
                        beta[j] = U[u_i].sv[j] + V[v_i].sv[j];
                    idx = rank_comp(beta, NS, deg);
                    mpz_divexact(tz, den, mpq_denref(val));
                    mpz_mul(tz, tz, mpq_numref(val));
                    mpz_mul(tz, tz, cav);
                    mpz_mul_si(tz, tz, (long)U[u_i].c);
                    mpz_add(num[idx], num[idx], tz);
                }
            }
        }

        fprintf(fo, "E %d %d %d %d %d\n", hdr[0], hdr[1], hdr[2], hdr[3], deg);
        gmp_fprintf(fo, "D %Zd\n", den);
        for (q = 0; q < nbeta; q++) {
            if (!mpz_sgn(num[q])) continue;
            unrank_comp(q, NS, deg, beta);
            for (j = 0; j < NS; j++) fprintf(fo, "%d ", beta[j]);
            gmp_fprintf(fo, "%Zd\n", num[q]);
        }
        fprintf(fo, ".\n");
        fflush(fo);
        fprintf(stderr, "lambda [%d %d] k1=%d k2=%d  triples=%lld nonzero=%lld "
                        "cache=%zu vals=%zu\n",
                hdr[0], hdr[1], hdr[2], hdr[3], visited, hits, BIG.used,
                SMALL.used);

        for (j = 0; j < nbeta; j++) mpz_clear(num[j]);
        free(num); free(A); free(U); free(V);
        mpz_clear(den); mpz_clear(gq); mpz_clear(tz); mpz_clear(cav);
        mpq_clear(val);
        table_free(&BIG);   /* the four column values go; the small ones stay */
    }
    fclose(fi); fclose(fo);
    return 0;
}
