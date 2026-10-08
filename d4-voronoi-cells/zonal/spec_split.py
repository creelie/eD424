"""
spec_split.py -- take a subset of the entries of a psker spec file.

Used to share the remaining entries between processes, and to skip entries that
have already been computed.
"""
import struct
import sys

NG, NS = 16, 7
REC_A = NG + 8
REC_UV = NG + NS + 8


def read_entries(path):
    with open(path, 'rb') as f:
        blob = f.read()
    n = struct.unpack_from('<i', blob, 0)[0]
    off = 4
    out = []
    for _ in range(n):
        start = off
        l1, l2, k1, k2, na, nu, nv = struct.unpack_from('<7i', blob, off)
        off += 28 + na * REC_A + (nu + nv) * REC_UV
        out.append(((l1, l2, k1, k2), blob[start:off]))
    return out


def main():
    src, done_file = sys.argv[1], sys.argv[2]
    outs = sys.argv[3:]
    done = set()
    if done_file != '-':
        for line in open(done_file):
            if line.startswith('E '):
                f = line.split()
                done.add((int(f[1]), int(f[2]), int(f[3]), int(f[4])))
    entries = read_entries(src)
    todo = [e for e in entries if e[0] not in done]
    # the cost of an entry is about |A| * |U| * |V|; deal them out greedily so
    # the processes finish together
    def cost(e):
        _, _, _, _, na, nu, nv = struct.unpack_from('<7i', e[1], 0)
        return na * nu * nv
    todo.sort(key=cost, reverse=True)
    buckets = [[] for _ in outs]
    load = [0] * len(outs)
    for e in todo:
        i = load.index(min(load))
        buckets[i].append(e)
        load[i] += cost(e)
    for path, b, l in zip(outs, buckets, load):
        with open(path, 'wb') as f:
            f.write(struct.pack('<i', len(b)))
            for _, blob in b:
                f.write(blob)
        print('%s: %d entries, relative load %.3g' % (path, len(b), l))
    print('%d of %d entries already done' % (len(entries) - len(todo),
                                             len(entries)))


if __name__ == '__main__':
    main()
