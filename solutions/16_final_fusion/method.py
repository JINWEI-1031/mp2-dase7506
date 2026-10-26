"""Scheme 16 -- inference for the ultimate-fusion stacking meta-learner.

Pure numpy.  `predict(bank, arrays, checkpoint)` reproduces exactly the frozen
pipeline:

    features (text/group/vote/embed/shape/ngsc)
        -> 8 base scorers
        -> meta features (within-question z / average rank / argmax of every base
           + n_bases_top1 + 8 group consensus statistics)
        -> Bradley-Terry meta learner -> argmax inside each question

No position, candidate-index or question-id prior is used anywhere: every input
is a within-question statistic, so the choice is invariant to a permutation of
the four candidates.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import f16_features as MF  # noqa: E402


def _load(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def base_scores(config, blocks):
    """Return (S [N,4,B], base_names) exactly as frozen in artifacts/."""
    n = blocks['text'].shape[0]
    names = config['base_names']
    S = np.zeros((n, 4, len(names)), dtype=np.float64)
    for b, spec in enumerate(config['bases']):
        X = MF.assemble(blocks, spec['blocks'])
        if spec['kind'] == 'raw':
            S[:, :, b] = X[:, :, 0]
            continue
        w = _load(Path(config['_artifacts']) / spec['file'])
        Z = (X - w['mu'].astype(np.float64)) / w['sd'].astype(np.float64)
        s = Z @ w['coef'].astype(np.float64)
        if spec['kind'] == 'pointwise':
            s = s + float(w['intercept'])
        S[:, :, b] = s
    return S, names


def predict(bank, arrays, checkpoint):
    ckpt = Path(checkpoint) if checkpoint is not None else Path(__file__).resolve().parent / 'artifacts'
    config = json.loads((ckpt / 'config.json').read_text())
    config['_artifacts'] = str(ckpt)
    meta = config['meta']

    blocks = MF.build_blocks(bank, arrays['embeddings'])
    S, base_names = base_scores(config, blocks)
    Fm, _ = MF.meta_design(S, {'vote': blocks['vote'], 'group': blocks['group']},
                           base_names, include_consensus=bool(meta['include_consensus']))
    w = _load(ckpt / 'meta.npz')

    cols = meta.get('cols')
    Fc = Fm if cols is None else Fm[:, :, cols]
    Z = (Fc - w['mu'].astype(np.float64)) / w['sd'].astype(np.float64)
    scores = Z @ w['w'].astype(np.float64)
    if meta['kind'] == 'point':
        scores = scores + float(w['intercept'])
    return [int(i) for i in scores.argmax(axis=1)]
