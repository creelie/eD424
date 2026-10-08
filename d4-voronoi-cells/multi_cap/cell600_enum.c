/* cell600_enum.c -- exhaustive enumeration of the independent sets of size 23
 * in the edge graph of the 600-cell (two vertices adjacent when 36 degrees apart).
 *
 * Reads the graph from cell600_graph.txt (written by cell600_exact.py):
 *   first line: n m ; then m lines "i j" ; then a line "24cells 25" followed by
 *   25 lines of 24 vertex indices each.
 * Counts every independent set of size 23 that contains vertex 0 (the graph is
 * vertex transitive, so this loses nothing), and reports how many of them lie in
 * no listed 24-cell.  Bitsets of 120 bits, backtracking with a clique-cover bound.
 *
 *   cc -O2 -o cell600_enum cell600_enum.c && ./cell600_enum
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define N 120
typedef struct { uint64_t w[2]; } bs;
static bs adj[N];
static bs cells[32]; static int ncells = 0;
static long long count23 = 0, outside = 0;

static inline int popc(bs a){ return __builtin_popcountll(a.w[0]) + __builtin_popcountll(a.w[1]); }
static inline int empty(bs a){ return !(a.w[0] | a.w[1]); }
static inline int lowbit(bs a){ return a.w[0] ? __builtin_ctzll(a.w[0]) : 64 + __builtin_ctzll(a.w[1]); }
static inline void clr(bs *a, int v){ a->w[v>>6] &= ~(1ULL << (v & 63)); }
static inline void set(bs *a, int v){ a->w[v>>6] |=  (1ULL << (v & 63)); }
static inline int has(bs a, int v){ return (a.w[v>>6] >> (v & 63)) & 1; }
static inline bs and_(bs a, bs b){ bs r; r.w[0]=a.w[0]&b.w[0]; r.w[1]=a.w[1]&b.w[1]; return r; }
static inline bs andnot(bs a, bs b){ bs r; r.w[0]=a.w[0]&~b.w[0]; r.w[1]=a.w[1]&~b.w[1]; return r; }

/* clique cover of cand: an independent set meets each clique at most once */
static int cover_bound(bs cand){
    int k = 0;
    while(!empty(cand)){
        k++;
        int v = lowbit(cand); clr(&cand, v);
        bs cc = and_(cand, adj[v]);
        while(!empty(cc)){
            int u = lowbit(cc); clr(&cand, u); cc = and_(cc, adj[u]);
        }
    }
    return k;
}

static void rec(bs chosen, bs cand, int need){
    if(need == 0){
        count23++;
        int inside = 0;
        for(int c = 0; c < ncells; c++) if(empty(andnot(chosen, cells[c]))){ inside = 1; break; }
        if(!inside) outside++;
        return;
    }
    if(popc(cand) < need) return;
    if(cover_bound(cand) < need) return;
    while(!empty(cand)){
        if(popc(cand) < need) return;
        int v = lowbit(cand); clr(&cand, v);
        bs ch = chosen; set(&ch, v);
        rec(ch, andnot(cand, adj[v]), need - 1);
    }
}

int main(void){
    FILE *f = fopen("cell600_graph.txt", "r");
    if(!f){ fprintf(stderr, "cell600_graph.txt missing: run cell600_exact.py first\n"); return 2; }
    int n, m; if(fscanf(f, "%d %d", &n, &m) != 2 || n != N){ fprintf(stderr, "bad header\n"); return 2; }
    memset(adj, 0, sizeof adj);
    for(int e = 0; e < m; e++){ int i, j; if(fscanf(f, "%d %d", &i, &j) != 2) return 2; set(&adj[i], j); set(&adj[j], i); }
    char tag[16]; if(fscanf(f, "%15s %d", tag, &ncells) != 2 || ncells > 32) return 2;
    for(int c = 0; c < ncells; c++){ memset(&cells[c], 0, sizeof(bs)); for(int k = 0; k < 24; k++){ int v; if(fscanf(f, "%d", &v) != 1) return 2; set(&cells[c], v); } }
    fclose(f);
    for(int v = 0; v < N; v++) if(popc(adj[v]) != 12){ fprintf(stderr, "vertex %d has degree %d\n", v, popc(adj[v])); return 2; }
    /* sets containing vertex 0 */
    bs chosen; memset(&chosen, 0, sizeof chosen); set(&chosen, 0);
    bs cand; cand.w[0] = ~0ULL; cand.w[1] = (1ULL << 56) - 1;   /* 120 bits */
    clr(&cand, 0); cand = andnot(cand, adj[0]);
    rec(chosen, cand, 22);
    printf("independent 23-sets containing vertex 0: %lld\n", count23);
    printf("  => total over all vertices: %lld * 120 / 23 = %lld\n", count23, count23 * 120 / 23);
    printf("  of those containing vertex 0, lying in no inscribed 24-cell: %lld\n", outside);
    if(count23 * 120 % 23 != 0){ printf("FAIL: count not divisible as vertex transitivity requires\n"); return 1; }
    if(outside != 0){ printf("FAIL: a 23-set outside every 24-cell exists\n"); return 1; }
    printf("PASS\n");
    return 0;
}
