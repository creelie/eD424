#!/usr/bin/env python3
"""
run_combo.py -- the combined kernel of statement (C) (gap_closure/CM/combo_direct.py:
the two-point kernel labelled by distance plus the three-point kernel on directions
typed by distance range) for several counts at once, on a machine with more cores
than the cloud container.

    python run_combo.py CASE:DEGREE:ROUNDS[:METHOD[:THREADS]] ...
    e.g.  python run_combo.py case29_all.json:10:4 case27_all.json:8:5

Each job runs gap_closure/CM/combo_direct.py on the case file of that folder, in its
own process, with Clarabel's factorisation METHOD (auto, qdldl or faer, the last
multithreaded on THREADS threads).
Its output goes to <tag>_d<degree>.log next to this file, its certificates (one .npz
per round) to gap_closure/CM.  run_combo.log here records the machine and, when each
job ends, its last round.

Floating point on sampled constraints: a corrected bound below 9 pi^2/8 - 8 = 3.10330
is what the exact checker multi_cap/combo_case_check.py then has to confirm, and no
result here is used in a proof until it has.
"""
import json
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CM = os.path.join(HERE, '..', 'CM')
LOG = open(os.path.join(HERE, 'run_combo.log'), 'a', buffering=1)
sys.path.insert(0, HERE)


def say(msg):
    LOG.write(time.strftime('%Y-%m-%d %H:%M:%S ') + msg + '\n')
    print(msg, flush=True)


def main():
    from run_typed import memory_gb
    tot, av = memory_gb()
    say('machine: %s, %s, %d logical cores, %.1f GB memory (%.1f GB free), Python %s'
        % (platform.node(), platform.platform(), os.cpu_count() or 0, tot, av, platform.python_version()))
    try:
        import numpy, scipy, clarabel
        say('numpy %s, scipy %s, clarabel %s' % (numpy.__version__, scipy.__version__, clarabel.__version__))
    except Exception as e:
        say('missing package: %r' % (e,))
        sys.exit(1)
    jobs = []
    for spec in sys.argv[1:]:
        parts = spec.split(':')
        case, d3, rounds = parts[0], parts[1], parts[2]
        method = parts[3] if len(parts) > 3 else 'auto'
        threads = parts[4] if len(parts) > 4 else '4'
        tag = json.load(open(os.path.join(CM, case)))['tag']
        logpath = os.path.join(HERE, '%s_d%s.log' % (tag, d3))
        env = dict(os.environ, DSM=method, THREADS=threads, TAG=tag + 'pc')
        f = open(logpath, 'a', buffering=1)
        p = subprocess.Popen([sys.executable, '-u', os.path.join(CM, 'combo_direct.py'), d3, rounds, case],
                             cwd=CM, env=env, stdout=f, stderr=subprocess.STDOUT)
        say('started %s at degree %s, %s rounds, factorisation %s: log %s' % (case, d3, rounds, method, os.path.basename(logpath)))
        jobs.append((case, d3, p, logpath))
    while jobs:
        time.sleep(60)
        for job in list(jobs):
            case, d3, p, logpath = job
            if p.poll() is not None:
                last = [l for l in open(logpath) if 'round' in l]
                say('ended %s at degree %s, exit %s: %s' % (case, d3, p.returncode, last[-1].strip() if last else 'no round'))
                jobs.remove(job)
    say('done')


if __name__ == '__main__':
    main()
