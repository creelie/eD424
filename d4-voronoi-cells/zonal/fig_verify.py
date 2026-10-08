#!/usr/bin/env python3
"""
fig_verify.py -- the two pictures of the re-verification.

Panel (a) is why step 3 does not need the memory it is usually given. The
integral is taken over triples of monomials, one from each of the three
factors of the integrand; the parity of the exponent matrix kills all but a
thirtieth of them, the invariance of the integral under permuting rows and
columns collapses what is left onto its canonical forms, and the answer, being
homogeneous of degree at most fourteen in seven variables, has at most
C(20,6) coefficients however large the signature. Each bar is measured, not
estimated: the first two are the counters in psker (runs/triple_counts.log),
the third the size of its memo table on the largest entry, the last the
binomial bound.

Panel (b) is step 5 for the four-point constraint, the one that matters. The
constraint is a difference of two polynomials in the six inner products, one
assembled from fifty sum-of-squares blocks and one from the zonal matrices.
The curve is the zonal side filling up, signature by signature, over the three
ways of splitting the four points into two pairs; the numbers are the ones the
run printed (runs/step5_constraint_4_zonal.log). It ends on 53572 terms, which
is exactly where the sum-of-squares side ends too, and the difference is zero
at every one of them.

Writes fig_verify.png next to this file.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'fig_verify.png')

plt.rcParams.update({'font.family': 'serif', 'font.size': 9,
                     'figure.dpi': 200, 'savefig.dpi': 400,
                     'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02})

# runs/triple_counts.log, runs/step3_part*.log
CASCADE = [
    ('monomial triples,\nbefore the parity rule', 2.392e11, r'$2.39\times10^{11}$'),
    ('triples the parity rule\nleaves', 7.854e9, r'$7.85\times10^{9}$'),
    ('distinct canonical forms,\nlargest entry', 2505430, r'$2\,505\,430$'),
    ('coefficients of $P(S)$,\nany signature', 38760, r'$38\,760$'),
]

# runs/step5_constraint_4_zonal.log: (seconds, terms accumulated)
TRACE = [(17, 81), (26, 4104), (302, 20536), (1106, 34552), (1396, 36263),
         (1397, 36267), (1408, 36399), (1705, 38711), (2519, 46378),
         (2810, 47847), (2811, 47851), (2822, 47923), (3122, 48623),
         (3942, 52351), (4236, 53572)]
SPLITS = [(1396, '(01)(23)'), (2810, '(02)(13)'), (4236, '(03)(12)')]
FINAL = 53572


def panel_a(ax):
    names = [n for n, v, lab in CASCADE][::-1]
    vals = np.array([v for n, v, lab in CASCADE][::-1])
    labs = [lab for n, v, lab in CASCADE][::-1]
    y = np.arange(len(vals))
    ax.barh(y, vals, height=0.6, color='#3b6ea5', edgecolor='black', linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xscale('log')
    ax.set_xlim(1e4, 2e13)
    ax.set_xlabel('count')
    for i, (v, lab) in enumerate(zip(vals, labs)):
        ax.text(v * 2.2, i, lab, va='center', fontsize=8)
    ax.set_title('(a)  what step 3 visits', fontsize=9, loc='left')
    ax.grid(axis='x', linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(True)


def panel_b(ax):
    t = np.array([a for a, b in TRACE]) / 60.0
    n = np.array([b for a, b in TRACE])
    ax.plot(t, n, '-o', color='#3b6ea5', markersize=3, linewidth=1.2,
            label='zonal side, as it fills')
    ax.axhline(FINAL, color='#c1440e', linestyle='--', linewidth=1.1,
               label='sum-of-squares side, %d terms' % FINAL)
    for sec, name in SPLITS:
        ax.axvline(sec / 60.0, color='0.65', linewidth=0.6, linestyle=':')
        ax.text(sec / 60.0 - 0.9, 1500, name, rotation=90, fontsize=6.5,
                ha='right', va='bottom', color='0.35')
    ax.set_xlabel('minutes of the run')
    ax.set_ylabel('coefficients carried')
    ax.set_xlim(0, 76)
    ax.set_ylim(0, 86000)
    ax.set_yticks([0, 20000, 40000, 60000])
    ax.set_title('(b)  step 5, the four-point constraint', fontsize=9, loc='left')
    ax.legend(fontsize=7, frameon=False, loc='center right',
              bbox_to_anchor=(1.0, 0.40))
    ax.grid(linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(True)
    ax.text(1.5, 84000,
            'the two sides meet on the same %d monomials, each\n'
            'coefficient of one the exact negative of the other, about\n'
            '$15\\,700$ digits apiece; the difference is $0$ at every one'
            % FINAL, fontsize=7, va='top')


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    panel_a(ax1)
    panel_b(ax2)
    fig.tight_layout()
    fig.savefig(OUT)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
