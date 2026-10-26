"""How large is the noise floor of our metrics?

Bootstraps the test_like subset and the official test set so we can state the
minimum difference that is actually detectable. Quantifies why the last two
rounds of test_like gains do not transfer.

Read-only.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import load_split, parse_answer, test_like_mask  # noqa: E402

ROOT = HERE.parents[1]
CHOICES = ROOT / 'solutions' / '_analysis' / 'choices'
RUNS = ROOT / 'code' / 'runs'
RNG = np.random.default_rng(0)


def boot_ci(vec, mask, n_boot=4000):
    idx = np.nonzero(mask)[0]
    vals = vec[idx].astype(float)
    n = len(vals)
    draws = np.array([RNG.choice(vals, size=n, replace=True).mean() for _ in range(n_boot)])
    return 100 * vals.mean(), 100 * np.percentile(draws, 2.5), 100 * np.percentile(draws, 97.5)


def paired_diff(a, b, mask, n_boot=4000):
    idx = np.nonzero(mask)[0]
    d = (a[idx] - b[idx]).astype(float)
    draws = np.array([RNG.choice(d, size=len(d), replace=True).mean() for _ in range(n_boot)])
    return 100 * d.mean(), 100 * np.percentile(draws, 2.5), 100 * np.percentile(draws, 97.5)


def main():
    bank, arrays, gold, refs = load_split('validation')
    mask = test_like_mask(bank, gold)

    correct = {}
    for p in sorted(CHOICES.glob('*.json')):
        ch = json.loads(p.read_text())['choices']
        correct[p.stem] = np.array([
            int(parse_answer(bank[i]['candidates'][ch[i]]['text']) == gold[i]) for i in range(len(bank))
        ])

    print('=== validation test_like: accuracy with 95% bootstrap CI (n=355) ===')
    rows = []
    for name, vec in correct.items():
        m, lo, hi = boot_ci(vec, mask)
        rows.append((m, lo, hi, name))
    for m, lo, hi, name in sorted(rows, reverse=True):
        print(f'  {name:26s} {m:6.2f}   [{lo:5.2f}, {hi:5.2f}]  width={hi-lo:4.2f}')

    print()
    print('=== paired difference on test_like (95% CI) ===')
    for a, b in [('16_final_fusion', '11_meta_stacker'),
                 ('11_meta_stacker', '09_pairwise_aligned'),
                 ('11_meta_stacker', '07_pairwise_energy'),
                 ('16_final_fusion', '05_logreg_embed')]:
        if a in correct and b in correct:
            d, lo, hi = paired_diff(correct[a], correct[b], mask)
            verdict = 'significant' if lo > 0 or hi < 0 else 'NOT significant'
            print(f'  {a} - {b}: {d:+.2f}  [{lo:+.2f}, {hi:+.2f}]  -> {verdict}')

    # official test set, from the saved prediction files
    print()
    print('=== official test (n=467): accuracy with 95% bootstrap CI ===')
    cbank, _, cgold, _ = load_split('test')
    tcorrect = {}
    for p in sorted(RUNS.glob('chk_*.json')):
        name = p.stem.replace('chk_', '')
        ch = json.loads(p.read_text())['selected']
        tcorrect[name] = np.array([
            int(parse_answer(cbank[i]['candidates'][ch[i]]['text']) == cgold[i]) for i in range(len(cbank))
        ])
    for p in sorted(RUNS.glob('test_*.json')):
        name = p.stem.replace('test_', '')
        ch = json.loads(p.read_text())['selected']
        tcorrect.setdefault(name, np.array([
            int(parse_answer(cbank[i]['candidates'][ch[i]]['text']) == cgold[i]) for i in range(len(cbank))
        ]))
    allm = np.ones(len(cbank), dtype=bool)
    trows = []
    for name, vec in tcorrect.items():
        m, lo, hi = boot_ci(vec, allm)
        trows.append((m, lo, hi, name))
    for m, lo, hi, name in sorted(trows, reverse=True):
        print(f'  {name:26s} {m:6.2f}   [{lo:5.2f}, {hi:5.2f}]  width={hi-lo:4.2f}')

    if '16_final_fusion' in tcorrect and '11_meta_stacker' in tcorrect:
        d, lo, hi = paired_diff(tcorrect['16_final_fusion'], tcorrect['11_meta_stacker'], allm)
        print(f'\n  paired 16 - 11 on test: {d:+.2f}  [{lo:+.2f}, {hi:+.2f}]'
              f'  -> {"significant" if lo > 0 or hi < 0 else "NOT significant (exactly tied)"}')
        diff_choices = sum(1 for i in range(len(cbank))
                           if json.loads((RUNS / 'chk_16_final_fusion.json').read_text())['selected'][i] !=
                           json.loads((RUNS / 'chk_11_meta_stacker.json').read_text())['selected'][i])
        print(f'  but they disagree on {diff_choices}/{len(cbank)} questions')


if __name__ == '__main__':
    main()
