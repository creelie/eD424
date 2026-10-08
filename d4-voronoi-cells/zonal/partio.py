"""
partio.py -- reading and writing a partial constraint polynomial.

The pieces of the four-point constraint have tens of thousands of coefficients
of several thousand digits, so a file is close to a gigabyte and building the
whole of it in memory before writing doubles the peak.  These write and read it
in chunks.  A file written in one piece by an earlier version is still read.
"""
import pickle

CHUNK = 4000


def save_parts(path, acc):
    """acc maps an exponent tuple to a flint fmpq (or anything with
    .numer()/.denom())"""
    items = [(e, c) for e, c in acc.items() if c != 0]
    with open(path, 'wb') as fh:
        pickle.dump(('chunks', len(items)), fh, 2)
        for i in range(0, len(items), CHUNK):
            pickle.dump({e: (int(c.numer()), int(c.denom()))
                         for e, c in items[i:i + CHUNK]}, fh, 2)


def iter_parts(path):
    """yield (exponent, (numerator, denominator)) for every stored coefficient"""
    with open(path, 'rb') as fh:
        head = pickle.load(fh)
        if isinstance(head, tuple) and head and head[0] == 'chunks':
            while True:
                try:
                    d = pickle.load(fh)
                except EOFError:
                    return
                for e, nd in d.items():
                    yield e, nd
        else:                       # an old single-dictionary file
            for e, nd in head.items():
                yield e, nd
