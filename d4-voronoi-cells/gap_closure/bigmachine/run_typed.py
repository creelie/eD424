#!/usr/bin/env python3
"""
run_typed.py -- the three-point exclusion "twenty-four centres within 2.0161
leave no room for a twenty-fifth within r" at high degree, for a machine with
more memory than the 15 GB of the cloud container.

For 24 points of S^3 with pairwise inner products at most t1 = 0.508 (centres
within 2.0161, thm:kissing-stable) and one further point with inner product at
most t2 with each of them (a centre within r), the typed programme of
multi_cap/typed_cardinality_sdp.py looks for a certificate Z < 0 that no such
code exists.  t2 = 0.6141 is r = sqrt 6, which would settle the case of
twenty-four close centres in statement (C); t2 = 0.51135 is r = 2.03, one leaf
of the count split at 26 centres in gap_closure/README.md.

Floating point with sampled constraints refined in rounds (exploration); a
corrected Z < 0 is what a rigorous check would then have to confirm.

    python run_typed.py [degrees ...] [--t2 a,b,...]
                     (default degrees 16 18 20, t2 0.5101,0.5114; the run
                     on record is "16 18 --t2 0.6141")

Writes run_typed.log next to itself, one line per round, flushed.
"""
import os
import sys
import time
import platform
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'multi_cap'))
LOG = open(os.path.join(HERE, 'run_typed.log'), 'a', buffering=1)


def say(msg):
    LOG.write(msg + '\n')
    print(msg, flush=True)


def memory_gb():
    try:
        if os.name == 'nt':
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                            ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                            ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                            ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                            ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
            m = MS(); m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullTotalPhys / 2 ** 30, m.ullAvailPhys / 2 ** 30
        pages = os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
        avail = os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
        return pages / 2 ** 30, avail / 2 ** 30
    except Exception:
        return float('nan'), float('nan')


if __name__ == '__main__':
    tot, av = memory_gb()
    say('machine: %s, %s, %d logical cores, %.1f GB memory (%.1f GB free), Python %s'
        % (platform.node(), platform.platform(), os.cpu_count() or 0, tot, av, platform.python_version()))
    try:
        import cvxpy
        import numpy
        say('numpy %s, cvxpy %s, solvers %s' % (numpy.__version__, cvxpy.__version__, cvxpy.installed_solvers()))
    except Exception as e:
        say('missing package: %r' % (e,))
        sys.exit(1)
    import typed_cardinality_sdp as T
    T.print = lambda *a, **k: say(' '.join(str(x) for x in a))     # route the per-round lines to the log
    args = sys.argv[1:]
    t2s = (0.5101, 0.5114)
    if '--t2' in args:
        i = args.index('--t2')
        t2s = tuple(float(x) for x in args[i + 1].split(','))
        del args[i:i + 2]
    degrees = [int(x) for x in args] or [16, 18, 20]
    t1 = 0.508
    for d in degrees:
        for t2 in t2s:
            t0 = time.time()
            say('degree %d, t1 %.4f, t2 %.4f: start' % (d, t1, t2))
            try:
                res = T.run(d, t1, t2, rounds=4, n=24)
            except MemoryError:
                say('degree %d: out of memory' % d)
                break
            except Exception as e:
                say('degree %d, t2 %.4f: failed: %r' % (d, t2, e))
                continue
            if res is None:
                say('degree %d, t2 %.4f: no solution' % (d, t2))
                continue
            say('RESULT degree %d t1 %.4f t2 %.4f: sampled Z %.5f, corrected Z %.5f [%.0f s] (below 0 excludes the code)'
                % (d, t1, t2, res[0], res[1], time.time() - t0))
    say('done')
