"""
merge4.py -- add one saved part of a constraint into a running total.

The four-point constraint is assembled a piece at a time, because no process
here can hold two of them at once: the coefficients are rationals with thousands
of digits and there are tens of thousands of them.  The running total lives in
one file and each new part is folded into it and then deleted.

    python3 merge4.py total.pkl part.pkl
"""
import os
import pickle
import sys

from flint import fmpq

import partio

sys.set_int_max_str_digits(0)


def load(path):
    if not os.path.exists(path):
        return {}
    return {e: fmpq(n, m) for e, (n, m) in partio.iter_parts(path)}


def save(path, acc):
    tmp = path + '.tmp'
    partio.save_parts(tmp, acc)
    os.replace(tmp, path)


def main(total_path, part_path):
    acc = load(total_path)
    before = len(acc)
    nin = 0
    for e, (n, m) in partio.iter_parts(part_path):
        nin += 1
        c = fmpq(n, m)
        if e in acc:
            v = acc[e] + c
            if v == 0:
                del acc[e]
            else:
                acc[e] = v
        elif c != 0:
            acc[e] = c
    save(total_path, acc)
    print('merged %-22s %7d terms in, total %7d -> %7d nonzero'
          % (part_path.split('/')[-1], nin, before, len(acc)))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
