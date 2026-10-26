"""Round-3 analysis: are the errors of the top schemes complementary?

Reads the per-scheme choice dumps produced by dump_choices.py, so no method
module is imported here (avoids cross-scheme import collisions).

Writes solutions/_analysis/round3_complementarity.json (plus a k=1 error dump).
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import load_split, parse_answer, test_like_mask  # noqa: E402

ROOT = HERE.parents[1]
CHOICES_DIR = ROOT / 'solutions' / '_analysis' / 'choices'
# Auto-discover every dumped scheme so the oracle tracks the current best set.
SCHEMES = sorted(p.stem for p in CHOICES_DIR.glob('*.json'))
PREFERRED = ['09_pairwise_aligned', '11_meta_stacker']


def main():
    bank, arrays, gold, refs = load_split('validation')
    n = len(bank)
    mask = test_like_mask(bank, gold)

    choices = {}
    for s in SCHEMES:
        p = CHOICES_DIR / f'{s}.json'
        if p.exists():
            choices[s] = json.loads(p.read_text())['choices']
    names = list(choices)

    correct = {
        s: np.array([int(parse_answer(bank[i]['candidates'][choices[s][i]]['text']) == gold[i])
                     for i in range(n)], dtype=int)
        for s in names
    }
    k_true = np.array([sum(1 for c in bank[i]['candidates'] if parse_answer(c['text']) == gold[i])
                       for i in range(n)])

    def acc(vec, m=mask):
        return round(100.0 * vec[m].sum() / m.sum(), 2)

    rep = {'n_test_like': int(mask.sum()), 'per_scheme': {}, 'error_overlap': {},
           'fusion': {}, 'per_k': {}}

    for s in names:
        rep['per_scheme'][s] = {'test_like': acc(correct[s]),
                                'full': round(100.0 * correct[s].sum() / n, 2),
                                'errors_test_like': int((~correct[s].astype(bool))[mask].sum())}

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ea, eb = (~correct[a].astype(bool)) & mask, (~correct[b].astype(bool)) & mask
            union = int((ea | eb).sum())
            inter = int((ea & eb).sum())
            rep['error_overlap'][f'{a} | {b}'] = {
                'shared': inter, 'union': union,
                'jaccard': round(inter / union, 3) if union else 0.0,
                'a_only': int((ea & ~eb).sum()), 'b_only': int((eb & ~ea).sum()),
            }

    # --- fusion experiments that need only the chosen indices ---
    any_ok = np.zeros(n, dtype=int)
    for s in names:
        any_ok |= correct[s]
    rep['fusion']['oracle_any_scheme_test_like'] = acc(any_ok)
    rep['fusion']['oracle_any_scheme_full'] = round(100.0 * any_ok.sum() / n, 2)

    # consensus: pick the candidate index chosen by the most schemes
    cons = np.zeros(n, dtype=int)
    for i in range(n):
        cnt = Counter(choices[s][i] for s in names)
        top = max(cnt.values())
        winners = [c for c, v in cnt.items() if v == top]
        cons[i] = min(winners)
    cons_ok = np.array([int(parse_answer(bank[i]['candidates'][cons[i]]['text']) == gold[i])
                        for i in range(n)])
    rep['fusion']['consensus_majority_of_schemes'] = {
        'test_like': acc(cons_ok), 'full': round(100.0 * cons_ok.sum() / n, 2)}
    rep['fusion']['consensus_final_tie_breaks'] = int(sum(
        1 for i in np.nonzero(mask)[0]
        if max(Counter(choices[s][i] for s in names).values()) * 2 <= len(names)))

    # agreement between the two strongest schemes
    a, b = PREFERRED
    if a in names and b in names:
        agree = sum(1 for i in range(n) if choices[a][i] == choices[b][i])
        rep['fusion'][f'agreement_{a}_vs_{b}'] = {'agree': agree, 'n': n,
                                                  'rate': round(agree / n, 3)}

    # per-k stratification
    best = max(names, key=lambda s: acc(correct[s]))
    rep['best_by_test_like'] = best
    for kv in (1, 2, 3):
        m = mask & (k_true == kv)
        if m.sum():
            row = {'n': int(m.sum()), 'oracle': round(100.0 * any_ok[m].sum() / m.sum(), 2)}
            for s in names:
                row[s] = round(100.0 * correct[s][m].sum() / m.sum(), 2)
            rep['per_k'][str(kv)] = row

    # how many questions does NO scheme get right (irreducible floor)
    rep['fusion']['no_scheme_correct_test_like'] = int(((~any_ok.astype(bool)) & mask).sum())

    # qualitative: k=1 errors of the best scheme
    dump = []
    for i in np.nonzero(mask & (k_true == 1) & (correct[best] == 0))[0][:10]:
        row = bank[i]
        dump.append({
            'question': row['question'][:220],
            'parsed_values': [parse_answer(c['text']) for c in row['candidates']],
            'gold': gold[i],
            'picked_by': {s: choices[s][i] for s in names},
            'tail_snippets': [c['text'][-140:].replace('\n', ' ') for c in row['candidates']],
        })
    rep['k1_error_samples'] = dump

    out = ROOT / 'solutions' / '_analysis'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'round3_complementarity.json').write_text(json.dumps(rep, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in rep.items() if k != 'k1_error_samples'}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
