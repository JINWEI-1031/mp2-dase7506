"""Shared evaluation harness for DASE7506 MP2 solution schemes.

Every scheme is evaluated through this harness so numbers are comparable.
It never reads test references for method development; it only uses the
reference file to *score* a frozen set of choices.

Usage
-----
    python harness.py --method <path/to/method.py> [--checkpoint DIR]
                      [--split validation] [--output result.json]

Reports
-------
  accuracy                selection accuracy on the full split
  accuracy_test_like      accuracy on the test-like subset (validation only)
  prediction_seconds      in-process time for load + import + predict
  assets_bytes            total size of the checkpoint directory (limit ~16 MiB)
"""
import argparse
import importlib.util
import json
import re
import sys
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'code' / 'data'
SPLITS = ('train', 'validation', 'test')


def parse_answer(text):
    """Mirror of code/common.py:parse_answer. Keep in sync with the official parser."""
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


def load_split(split):
    """Return (bank, arrays, gold_parsed, refs). arrays['embeddings'] is [N, 4, 1024]."""
    bank = read_jsonl(DATA / 'candidates' / f'gsm8k_{split}.jsonl')
    refs = read_jsonl(DATA / f'gsm8k_{split}.jsonl')
    with np.load(DATA / 'candidates' / f'gsm8k_{split}.npz', allow_pickle=False) as f:
        emb = f['embeddings'].copy()
    emb = emb.reshape(len(bank), 4, emb.shape[-1])
    gold = [parse_answer(r['answer']) for r in refs]
    return bank, {'embeddings': emb}, gold, refs


def test_like_mask(bank, gold):
    """Boolean mask for validation questions that resemble the test distribution.

    Test groups have 1..3 correct candidates and at least two distinct valid
    parsed answers, and never four correct candidates. This subset is the
    honest proxy for leaderboard behaviour; full validation is optimistic.
    """
    mask = np.zeros(len(bank), dtype=bool)
    for i, (row, g) in enumerate(zip(bank, gold)):
        vals = [parse_answer(c['text']) for c in row['candidates']]
        k = sum(1 for v in vals if v is not None and v == g)
        distinct = len({v for v in vals if v is not None})
        mask[i] = (1 <= k <= 3) and distinct >= 2
    return mask


def accuracy(choices, bank, gold, mask=None):
    idx = range(len(bank)) if mask is None else np.nonzero(mask)[0]
    idx = list(idx)
    if not idx:
        return {'n': 0, 'correct': 0, 'accuracy': 0.0}
    correct = 0
    for i in idx:
        ans = parse_answer(bank[i]['candidates'][choices[i]]['text'])
        correct += int(ans is not None and ans == gold[i])
    return {'n': len(idx), 'correct': correct, 'accuracy': 100.0 * correct / len(idx)}


def dir_size_bytes(path):
    path = Path(path)
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())


def load_method(method_path):
    method_path = Path(method_path).resolve()
    sys.path.insert(0, str(method_path.parent))
    spec = importlib.util.spec_from_file_location('scheme_method', method_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path)
    ap.add_argument('--split', choices=SPLITS, default='validation')
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()

    tick = time.perf_counter()
    bank, arrays, gold, refs = load_split(args.split)
    method = load_method(args.method)
    import_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    choices = method.predict(bank, arrays, args.checkpoint)
    predict_seconds = time.perf_counter() - tick

    if len(choices) != len(bank) or any(type(c) is not int or not 0 <= c < 4 for c in choices):
        raise ValueError('predict() must return one integer 0..3 per question')

    result = {
        'split': args.split,
        'full': accuracy(choices, bank, gold),
        'prediction_seconds': round(predict_seconds, 4),
        'load_and_import_seconds': round(import_seconds, 4),
        'assets_bytes': dir_size_bytes(args.checkpoint) if args.checkpoint else 0,
        'choice_histogram': {str(k): v for k, v in sorted(Counter(choices).items())},
    }
    if args.split == 'validation':
        mask = test_like_mask(bank, gold)
        result['test_like'] = accuracy(choices, bank, gold, mask)
        result['test_like_size'] = int(mask.sum())

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
