"""Data reconnaissance for DASE7506 MP2.

Read-only. Reports, per split:
  - distribution of correct-candidate counts
  - parse-failure rates
  - baseline accuracies (position-0, plurality vote, oracle)
  - position of first correct candidate

Runs on the supplied bank + references only. Never writes to data/.
"""
import json
import re
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'code' / 'data'


def parse_answer(text):
    """Reimplementation of common.parse_answer (kept in sync)."""
    text = str(text)
    tail = text.rsplit('####', 1)[1] if '####' in text else None
    if tail is None:
        boxed = re.findall(r'\\boxed\{([^{}]+)\}', text)
        tail = boxed[-1] if boxed else ''
    match = re.search(r'[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?', tail)
    if not match:
        return None
    try:
        return str(Fraction(match.group().replace(',', '')))
    except (ValueError, ZeroDivisionError):
        return None


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def recon(split):
    bank = read_jsonl(DATA / 'candidates' / f'gsm8k_{split}.jsonl')
    refs = read_jsonl(DATA / f'gsm8k_{split}.jsonl')
    assert len(bank) == len(refs)

    n = len(bank)
    n_correct_dist = Counter()
    first_correct_pos = Counter()
    cand_parse_fail = 0
    gold_parse_fail = 0
    pos0_correct = 0
    plurality_correct = 0
    plurality_tie = 0
    max_agreement_dist = Counter()
    any_parseable = 0

    for row, ref in zip(bank, refs):
        gold = parse_answer(ref['answer'])
        if gold is None:
            gold_parse_fail += 1
        parsed = []
        for c in row['candidates']:
            p = parse_answer(c['text'])
            parsed.append(p)
            if p is None:
                cand_parse_fail += 1
        if any(p is not None for p in parsed):
            any_parseable += 1

        correct = [int(p is not None and p == gold) for p in parsed]
        k = sum(correct)
        n_correct_dist[k] += 1
        if k > 0:
            first_correct_pos[correct.index(1)] += 1
        pos0_correct += correct[0]

        counts = Counter(p for p in parsed if p is not None)
        if counts:
            top = max(counts.values())
            winners = [v for v, c in counts.items() if c == top]
            max_agreement_dist[top] += 1
            if len(winners) > 1:
                plurality_tie += 1
            # plurality pick: highest count; ties broken by earliest candidate index
            best_value = None
            for p in parsed:
                if p in winners:
                    best_value = p
                    break
            plurality_correct += int(best_value == gold)
        else:
            max_agreement_dist[0] += 1

    return {
        'split': split,
        'n_questions': n,
        'correct_per_question': {str(k): n_correct_dist[k] for k in range(5)},
        'mean_correct_per_question': round(sum(k * v for k, v in n_correct_dist.items()) / n, 4),
        'questions_with_zero_correct': n_correct_dist[0],
        'candidate_parse_fail_rate': round(cand_parse_fail / (4 * n), 4),
        'gold_parse_fail': gold_parse_fail,
        'questions_with_no_parseable_candidate': n - any_parseable,
        'first_correct_position_hist': {str(k): first_correct_pos[k] for k in range(4)},
        'baseline_position0_accuracy': round(100 * pos0_correct / n, 3),
        'baseline_plurality_vote_accuracy': round(100 * plurality_correct / n, 3),
        'plurality_tie_questions': plurality_tie,
        'max_agreement_count_hist': {str(k): max_agreement_dist[k] for k in range(5)},
    }


if __name__ == '__main__':
    splits = sys.argv[1:] or ['train', 'validation', 'test']
    out = {s: recon(s) for s in splits}
    print(json.dumps(out, indent=2, ensure_ascii=False))
