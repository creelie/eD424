"""Match the vertices of data/vertices.json with the matrices J_0, ..., J_9 of Table 5.1
(verification/data/vertices.txt) under Gamma, and print the matching.

Usage: python3 table.py vertices.json vertices.txt"""
import sys, json
from fractions import Fraction as F
from equiv import isometries


def main(vpath, tpath):
    ours = [[[F(x) for x in row] for row in V['J']] for V in json.load(open(vpath))]
    table = []
    for line in open(tpath):
        a = line.split()
        table.append([[F(a[1 + 5 * i + j]) for j in range(5)] for i in range(5)])
    assert len(ours) == len(table) == 10
    used = set()
    for i, J in enumerate(ours):
        match = [j for j in range(10) if isometries(table[j], J)]
        assert len(match) == 1 and match[0] not in used, f"vertex {i}: {match}"
        j = match[0]
        used.add(j)
        T = isometries(table[j], J)[0]
        same = "the same matrix" if J == table[j] else f"T^T J_{j} T with T = {T}"
        print(f"vertex {i} = J_{j}: {same}")
    print("the ten vertices are the ten classes of Table 5.1")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
