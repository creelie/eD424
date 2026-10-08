import numpy as np
import itertools

def build_roots():
    roots = []
    for i in range(4):
        for j in range(i + 1, 4):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(4)
                    v[i] = si; v[j] = sj
                    roots.append(v / np.sqrt(2))
    return np.array(roots)

roots = build_roots()
G = roots @ roots.T
adj = np.abs(G - 0.5) < 1e-9
np.fill_diagonal(adj, False)
adjlist = {i: set(np.where(adj[i])[0].tolist()) for i in range(24)}
print("degree of adjacency graph (Gram=+1/2):", set(len(v) for v in adjlist.values()))

# longest INDUCED path (chain: consecutive adjacent, no other adjacencies) via DFS with pruning
best = [0]
best_path = [None]

def dfs(path, pathset):
    if len(path) > best[0]:
        best[0] = len(path)
        best_path[0] = list(path)
    last = path[-1]
    for nxt in adjlist[last]:
        if nxt in pathset:
            continue
        # induced path condition: nxt must be adjacent to 'last' only among path (no other adjacency to earlier elts)
        if any(nxt in adjlist[p] for p in path[:-1]):
            continue
        path.append(nxt); pathset.add(nxt)
        dfs(path, pathset)
        path.pop(); pathset.discard(nxt)

import sys
sys.setrecursionlimit(10000)
for start in range(24):
    dfs([start], {start})

print("Longest induced (self-avoiding, no chords) path length found:", best[0])
print("example path:", best_path[0])
