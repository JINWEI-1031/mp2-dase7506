"""Scheme 16 -- ultimate fusion: meta-stacker + answer-shape + number-grounding.

Pipeline (identical spine to scheme 11, re-implemented from scratch)
--------------------------------------------------------------------
  1. build 6 feature families per candidate
       text / group / vote / embed  (scheme 11's evidence families)
       + shape  (answer-form plausibility, scheme 14)   <- new base
       + ngsc   (arithmetic re-execution,  scheme 15)   <- new base
  2. define 8 base scorers.  The 6 of scheme 11 are reproduced verbatim, the two
     new ones are pointwise LR on `text+group+shape` / `text+group+ngsc`, i.e.
     exactly "the text base + one new evidence family".
  3. K-fold **out-of-fold** (OOF) construction of the base scores (K=10):
        for each fold -> fit every base on the other folds, score the held-out
        fold.  The concatenation forms the meta-feature matrix.
  4. meta features: within-question z / average rank / argmax of every base,
     + n_bases_top1 + 8 group consensus statistics.
  5. meta learner: **Bradley-Terry pairwise loss** on the OOF matrix.
  6. refit every base on 100 % of train, freeze into artifacts/.

Honesty controls
----------------
  * OOF only for meta training; an explicit in-fold control quantifies leakage.
  * validation is only *scored* (and used to break a CV tie, as the shared
    protocol sanctions); test is never consulted.
  * no candidate index / question id enters any feature.

Run:  python train.py [--no-cache]
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / '00_shared'))

import f16_features as MF  # noqa: E402
import harness  # noqa: E402

SEED = 20260916
K_FOLDS = 10
N_CV_FOLDS = 5
BLOCK_CACHE_VER = 'v1'
CACHE_VER = 'v2'

# (name, feature blocks, scorer kind, C grid)
BASE_SPECS = [
    ('text',      ('text', 'group'),                 'pointwise', [0.1, 0.3, 1.0, 3.0]),
    ('embed',     ('embed',),                        'pointwise', [0.0003, 0.001, 0.003, 0.01]),
    ('pair',      ('text', 'group', 'embed'),        'pairwise',  [0.003, 0.01, 0.03, 0.1]),
    ('pairab',    ('embed',),                        'pairwise',  [0.003, 0.01, 0.03, 0.1]),
    ('vote',      ('vote',),                         'raw',       []),
    ('voteaware', ('text', 'group', 'vote', 'embed'), 'pairwise', [0.01, 0.03, 0.1]),
    # --- new evidence families ------------------------------------------- #
    # shape is fused with the embedding (scheme 14's proven `emb+shape` view);
    # ngsc is a standalone question-grounded arithmetic-consistency view.
    ('shape',     ('embed', 'shape'),                'pointwise', [0.0001, 0.0003, 0.001, 0.003]),
    ('ngsc',      ('ngsc',),                         'pointwise', [0.1, 0.3, 1.0, 3.0]),
]
BASE_NAMES = [s[0] for s in BASE_SPECS]
ORIG6 = [BASE_NAMES.index(n) for n in ('text', 'embed', 'pair', 'pairab', 'vote', 'voteaware')]
SHAPE_I = BASE_NAMES.index('shape')
NGSC_I = BASE_NAMES.index('ngsc')

META_C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
CV_TIE_BAND = 0.2


# --------------------------------------------------------------------------- #
# labels / weights / folds
# --------------------------------------------------------------------------- #
def question_labels(bank, gold):
    n = len(bank)
    y = np.zeros((n, 4), dtype=np.int64)
    parsed = []
    for i, (row, g) in enumerate(zip(bank, gold)):
        ps = [MF.parse_answer(c['text']) for c in row['candidates']]
        parsed.append(ps)
        for j, p in enumerate(ps):
            y[i, j] = int(g is not None and p == g)
    return y, parsed


def aligned_index(y, parsed):
    out = []
    for i, ps in enumerate(parsed):
        k = int(y[i].sum())
        distinct = len({p for p in ps if p is not None})
        if 1 <= k <= 3 and distinct >= 2:
            out.append(i)
    return np.asarray(out, dtype=int)


def inv_k_weights(y):
    k = y.sum(1)
    w = np.zeros(len(y), dtype=np.float64)
    for i, ki in enumerate(k):
        if 1 <= ki <= 3:
            w[i] = 1.0 / ki
    return w


def make_folds(idx, n_splits, seed):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(idx[tr], idx[te]) for tr, te in kf.split(idx)]


def question_acc(scores, y):
    pick = scores.argmax(1)
    return 100.0 * float(y[np.arange(len(y)), pick].mean())


# --------------------------------------------------------------------------- #
# base scorers
# --------------------------------------------------------------------------- #
def make_pairs(Z, y, idx_q, w):
    """Bradley-Terry pairs d = z(correct) - z(incorrect); mirrored for sklearn."""
    Ds, Ls, Ws = [], [], []
    for q in idx_q:
        yq = y[q]
        pos = np.nonzero(yq == 1)[0]
        neg = np.nonzero(yq == 0)[0]
        if pos.size == 0 or neg.size == 0:
            continue
        Zq = Z[q]
        d = (Zq[pos][:, None, :] - Zq[neg][None, :, :]).reshape(-1, Zq.shape[-1])
        wq = np.full(d.shape[0], w[q] / pos.size)
        Ds.append(np.concatenate([d, -d], axis=0))
        Ls.append(np.concatenate([np.ones(d.shape[0]), np.zeros(d.shape[0])]))
        Ws.append(np.concatenate([wq, wq]) / 2.0)
    if not Ds:
        return np.zeros((0, Z.shape[-1])), np.zeros(0), np.zeros(0)
    return np.concatenate(Ds), np.concatenate(Ls), np.concatenate(Ws)


def fit_base(X, y, w, idx_fit, kind, C):
    flat = X[idx_fit].reshape(-1, X.shape[-1])
    mu = flat.mean(0)
    sd = flat.std(0) + 1e-8
    Z = (X - mu) / sd
    if kind == 'pointwise':
        F = Z[idx_fit].reshape(-1, Z.shape[-1])
        yy = y[idx_fit].reshape(-1)
        ww = np.repeat(w[idx_fit], 4)
        lr = LogisticRegression(C=C, max_iter=2000, solver='lbfgs')
        lr.fit(F, yy, sample_weight=ww)
        return mu, sd, lr.coef_[0].copy(), float(lr.intercept_[0])
    if kind == 'pairwise':
        D, L, W = make_pairs(Z, y, idx_fit, w)
        lr = LogisticRegression(C=C, max_iter=2000, solver='lbfgs', fit_intercept=False)
        lr.fit(D, L, sample_weight=W)
        return mu, sd, lr.coef_[0].copy(), 0.0
    raise ValueError(kind)


def score_base(X, mu, sd, coef, b):
    return ((X - mu) / sd) @ coef + b


def select_base_C(X, y, w, idx_aligned, kind, grid):
    if not grid:
        return None, {}
    folds = make_folds(idx_aligned, N_CV_FOLDS, SEED)
    curve = {}
    for C in grid:
        accs = []
        for tr_q, te_q in folds:
            mu, sd, coef, bi = fit_base(X, y, w, tr_q, kind, C)
            accs.append(question_acc(score_base(X[te_q], mu, sd, coef, bi), y[te_q]))
        curve[str(C)] = round(float(np.mean(accs)), 3)
    best = max(grid, key=lambda c: curve[str(c)])
    return best, curve


# --------------------------------------------------------------------------- #
# OOF / full base scores
# --------------------------------------------------------------------------- #
def build_oof(blocks, y, w, idx_aligned, folds, Cs):
    order = {q: i for i, q in enumerate(idx_aligned)}
    S = np.zeros((len(idx_aligned), 4, len(BASE_SPECS)), dtype=np.float64)
    for b, (name, keys, kind, _) in enumerate(BASE_SPECS):
        X = MF.assemble(blocks, keys)
        if kind == 'raw':
            S[:, :, b] = X[idx_aligned][:, :, 0]
            continue
        for tr_q, te_q in folds:
            mu, sd, coef, bi = fit_base(X, y, w, tr_q, kind, Cs[name])
            S[[order[q] for q in te_q], :, b] = score_base(X[te_q], mu, sd, coef, bi)
    return S


def build_full(fit_blocks, score_blocks, y, w, idx_fit, Cs):
    n_score = score_blocks['text'].shape[0]
    S = np.zeros((n_score, 4, len(BASE_SPECS)), dtype=np.float64)
    params = {}
    for b, (name, keys, kind, _) in enumerate(BASE_SPECS):
        Xf = MF.assemble(fit_blocks, keys)
        Xs = Xf if score_blocks is fit_blocks else MF.assemble(score_blocks, keys)
        if kind == 'raw':
            S[:, :, b] = Xs[:, :, 0]
            params[name] = None
            continue
        mu, sd, coef, bi = fit_base(Xf, y, w, idx_fit, kind, Cs[name])
        params[name] = (mu, sd, coef, bi)
        S[:, :, b] = score_base(Xs, mu, sd, coef, bi)
    return S, params


def build_blocks_cached(bank, emb, tag, use_cache=True):
    if use_cache:
        d = HERE / '_cache'
        d.mkdir(exist_ok=True)
        f = d / f'blocks_{tag}_{BLOCK_CACHE_VER}.npz'
        if f.exists():
            with np.load(f) as z:
                return {k: z[k] for k in z.files}
        b = MF.build_blocks(bank, emb)
        np.savez(f, **b)
        return b
    return MF.build_blocks(bank, emb)


def ensure_base_scores(blocks_tr, blocks_va, y_tr, w_tr, idx_aligned, use_cache):
    d = HERE / '_cache'
    d.mkdir(exist_ok=True)
    f = d / f'base_scores_{CACHE_VER}.npz'
    if use_cache and f.exists():
        with np.load(f, allow_pickle=True) as z:
            out = {k: z[k] for k in z.files}
        out['Cs'] = json.loads(str(out['Cs_json']))
        out['curves'] = json.loads(str(out['curves_json']))
        params = {}
        for name in BASE_NAMES:
            if f'p_{name}_1' in out:
                params[name] = tuple(out[f'p_{name}_{i}'] for i in range(4))
            else:
                params[name] = None
        out['params'] = params
        print('base scores loaded from cache')
        return out

    t = time.perf_counter()
    Cs, curves = {}, {}
    for name, keys, kind, grid in BASE_SPECS:
        if kind == 'raw':
            Cs[name] = None
            continue
        X = MF.assemble(blocks_tr, keys)
        t0 = time.perf_counter()
        best, curve = select_base_C(X, y_tr, w_tr, idx_aligned, kind, grid)
        Cs[name] = best
        curves[name] = curve
        print(f'  base {name:9s} C={best}  train-CV acc={curve[str(best)]:.2f}  '
              f'({time.perf_counter()-t0:.1f}s)')

    folds = make_folds(idx_aligned, K_FOLDS, SEED)
    S_oof = build_oof(blocks_tr, y_tr, w_tr, idx_aligned, folds, Cs)
    blocks_tr_al = {k: v[idx_aligned] for k, v in blocks_tr.items()}
    S_val, params = build_full(blocks_tr, blocks_va, y_tr, w_tr, idx_aligned, Cs)
    S_in, _ = build_full(blocks_tr, blocks_tr_al, y_tr, w_tr, idx_aligned, Cs)
    print(f'base scores built ({time.perf_counter()-t:.1f}s)')

    out = {'S_oof': S_oof, 'S_val': S_val, 'S_in': S_in, 'Cs': Cs, 'curves': curves,
           'params': params, 'Cs_json': json.dumps(Cs), 'curves_json': json.dumps(curves)}
    if use_cache:
        save = {'S_oof': S_oof, 'S_val': S_val, 'S_in': S_in,
                'Cs_json': np.array(json.dumps(Cs)), 'curves_json': np.array(json.dumps(curves))}
        for name, p in params.items():
            if p is not None:
                for i, key in enumerate(('mu', 'sd', 'coef', 'b')):
                    save[f'p_{name}_{i}'] = p[i]
            else:
                save[f'p_{name}_0'] = np.array([])
        np.savez(f, **save)
    return out


# --------------------------------------------------------------------------- #
# meta learners
# --------------------------------------------------------------------------- #
def _std_fit(Ftr):
    mu = Ftr.reshape(-1, Ftr.shape[-1]).mean(0)
    sd = Ftr.reshape(-1, Ftr.shape[-1]).std(0) + 1e-8
    return mu, sd


def make_meta_pairs(F, y):
    Ds, Ls, Ws = [], [], []
    for q in range(len(y)):
        yq = y[q]
        pos = np.nonzero(yq == 1)[0]
        neg = np.nonzero(yq == 0)[0]
        if pos.size == 0 or neg.size == 0:
            continue
        Fq = F[q]
        d = (Fq[pos][:, None, :] - Fq[neg][None, :, :]).reshape(-1, Fq.shape[-1])
        wq = np.full(d.shape[0], 1.0 / pos.size)
        Ds.append(np.concatenate([d, -d], axis=0))
        Ls.append(np.concatenate([np.ones(d.shape[0]), np.zeros(d.shape[0])]))
        Ws.append(np.concatenate([wq, wq]) / 2.0)
    return np.concatenate(Ds), np.concatenate(Ls), np.concatenate(Ws)


def fit_meta(kind, C, F, y, cols=None):
    Fc = F if cols is None else F[:, :, cols]
    if kind == 'point':
        mu, sd = _std_fit(Fc)
        lr = LogisticRegression(C=C, max_iter=3000, solver='lbfgs')
        lr.fit(((Fc - mu) / sd).reshape(-1, Fc.shape[-1]), y.reshape(-1))
        return mu, sd, lr.coef_[0].copy(), float(lr.intercept_[0])
    if kind == 'pair':
        mu, sd = _std_fit(Fc)
        D, L, W = make_meta_pairs((Fc - mu) / sd, y)
        lr = LogisticRegression(C=C, max_iter=3000, solver='lbfgs', fit_intercept=False)
        lr.fit(D, L, sample_weight=W)
        return mu, sd, lr.coef_[0].copy(), 0.0
    raise ValueError(kind)


def apply_meta(kind, mu, sd, w, b, F, cols=None):
    Fc = F if cols is None else F[:, :, cols]
    Z = (Fc - mu) / sd
    return Z @ w + b


def meta_cv(kind, C, F, y, cols=None, n_splits=5, seed=None):
    seed = SEED + 1 if seed is None else seed
    n = len(y)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    for tr, te in kf.split(np.arange(n)):
        mu, sd, w, b = fit_meta(kind, C, F[tr], y[tr], cols)
        accs.append(question_acc(apply_meta(kind, mu, sd, w, b, F[te], cols), y[te]))
    return float(np.mean(accs))


# --------------------------------------------------------------------------- #
# subset evaluation helper (ablation workhorse)
# --------------------------------------------------------------------------- #
def eval_subset(S_oof, S_val, base_idx, y_meta, oof_blocks, val_blocks,
                kind='pair', C_grid=META_C_GRID):
    """Fit the meta learner on a base subset (OOF), score it on validation.

    The meta hyper-parameter C is chosen on the **train OOF matrix only**
    (question-level 5-fold CV); a 0.2-point CV tie band is then broken on
    validation (the protocol-sanctioned selection split).
    """
    names = [BASE_NAMES[i] for i in base_idx]
    So = S_oof[:, :, base_idx]
    Sv = S_val[:, :, base_idx]
    Fo, nm = MF.meta_design(So, {'vote': oof_blocks[0], 'group': oof_blocks[1]}, names, True)
    Fv, _ = MF.meta_design(Sv, {'vote': val_blocks[0], 'group': val_blocks[1]}, names, True)
    cols = MF.z_ntop_cons_cols(nm, len(names))
    rows, best_cv = {}, -1.0
    for C in C_grid:
        cv = meta_cv(kind, C, Fo, y_meta, cols)
        mu, sd, w, b = fit_meta(kind, C, Fo, y_meta, cols)
        ch = apply_meta(kind, mu, sd, w, b, Fv, cols).argmax(1).tolist()
        perk = per_k_accuracy(ch, bank_va_ref, gold_va_ref, y_va, tl_idx)
        rows[str(C)] = {
            'meta_cv': round(cv, 3),
            'val_full': round(harness.accuracy(ch, bank_va_ref, gold_va_ref)['accuracy'], 3),
            'val_test_like': round(harness.accuracy(ch, bank_va_ref, gold_va_ref, mask_tl)['accuracy'], 3),
            'k1': perk['1']['accuracy'], 'k2': perk['2']['accuracy'], 'k3': perk['3']['accuracy']}
        best_cv = max(best_cv, cv)
    band = [c for c in C_grid if rows[str(c)]['meta_cv'] >= best_cv - CV_TIE_BAND]
    best_C = max(band, key=lambda c: rows[str(c)]['val_test_like'])
    r = {'kind': kind, 'bases': names, 'C': best_C, 'meta_cv': rows[str(best_C)]['meta_cv'],
         'grid': rows, 'val_full': rows[str(best_C)]['val_full'],
         'val_test_like': rows[str(best_C)]['val_test_like'],
         'per_k_test_like': {k: {'accuracy': rows[str(best_C)][f'k{k}']} for k in ('1', '2', '3')}}
    # expose the full per-k dict (with n / correct) for the chosen C
    mu, sd, w, b = fit_meta(kind, best_C, Fo, y_meta, cols)
    ch = apply_meta(kind, mu, sd, w, b, Fv, cols).argmax(1).tolist()
    r['per_k_test_like'] = per_k_accuracy(ch, bank_va_ref, gold_va_ref, y_va, tl_idx)
    r['choices'] = ch
    return r


def per_k_accuracy(choices, bank, gold, y, idx):
    out = {}
    for k in (1, 2, 3):
        sel = [i for i in idx if int(y[i].sum()) == k]
        if not sel:
            out[str(k)] = {'n': 0, 'accuracy': 0.0}
            continue
        c = sum(int(MF.parse_answer(bank[i]['candidates'][choices[i]]['text']) == gold[i]) for i in sel)
        out[str(k)] = {'n': len(sel), 'correct': c, 'accuracy': round(100.0 * c / len(sel), 3)}
    return out


# globals used by eval_subset (set in main) -- keeps signatures short
bank_va_ref = gold_va_ref = mask_tl = y_va = tl_idx = None


# --------------------------------------------------------------------------- #
def main():
    global bank_va_ref, gold_va_ref, mask_tl, y_va, tl_idx
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-cache', action='store_true')
    args = ap.parse_args()
    use_cache = not args.no_cache

    t0 = time.perf_counter()
    print('loading splits ...')
    bank_tr, arr_tr, gold_tr, _ = harness.load_split('train')
    bank_va, arr_va, gold_va, _ = harness.load_split('validation')
    blocks_tr = build_blocks_cached(bank_tr, arr_tr['embeddings'], 'train', use_cache)
    blocks_va = build_blocks_cached(bank_va, arr_va['embeddings'], 'val', use_cache)
    y_tr, parsed_tr = question_labels(bank_tr, gold_tr)
    y_va, parsed_va = question_labels(bank_va, gold_va)
    bank_va_ref, gold_va_ref = bank_va, gold_va
    print(f'features ready ({time.perf_counter()-t0:.1f}s)  train {y_tr.shape}  val {y_va.shape}')

    w_tr = inv_k_weights(y_tr)
    idx_aligned = aligned_index(y_tr, parsed_tr)
    mask_tl = harness.test_like_mask(bank_va, gold_va)
    tl_idx = np.nonzero(mask_tl)[0]
    print(f'aligned train {len(idx_aligned)}/{len(y_tr)}   val test_like {len(tl_idx)}/{len(bank_va)}')

    B = ensure_base_scores(blocks_tr, blocks_va, y_tr, w_tr, idx_aligned, use_cache)
    S_oof, S_val, S_in, Cs, curves = B['S_oof'], B['S_val'], B['S_in'], B['Cs'], B['curves']

    y_meta = y_tr[idx_aligned]
    vote_tr = blocks_tr['vote'][idx_aligned]
    grp_tr = blocks_tr['group'][idx_aligned]
    vblocks_va, gblocks_va = blocks_va['vote'], blocks_va['group']
    oof_cons, val_cons = (vote_tr, grp_tr), (vblocks_va, gblocks_va)

    ablations = {}

    # ---------------- A1: single base models -------------------------------- #
    single = {}
    for b, name in enumerate(BASE_NAMES):
        ch = S_val[:, :, b].argmax(1).tolist()
        single[name] = {
            'val_full': round(harness.accuracy(ch, bank_va, gold_va)['accuracy'], 3),
            'val_test_like': round(harness.accuracy(ch, bank_va, gold_va, mask_tl)['accuracy'], 3),
            'per_k_test_like': per_k_accuracy(ch, bank_va, gold_va, y_va, tl_idx),
        }
    ablations['A1_single_base'] = single
    print('A1 single bases:', json.dumps({k: v['val_test_like'] for k, v in single.items()}))

    # oracle over bases
    def oracle(base_subset):
        cnt = 0
        for i in tl_idx:
            ok = False
            for b in base_subset:
                c = int(S_val[i, :, b].argmax())
                a = MF.parse_answer(bank_va[i]['candidates'][c]['text'])
                ok |= (a is not None and a == gold_va[i])
            cnt += int(ok)
        return round(100.0 * cnt / len(tl_idx), 3)

    oracles = {'orig6': oracle(ORIG6), 'all8': oracle(list(range(8)))}
    ablations['oracle_over_bases_test_like'] = oracles
    print('oracle (val test_like):', oracles)

    # ---------------- candidate base-sets (pre-registered) ------------------ #
    # Space: the four *content* bases of scheme 11 are retained as the backbone;
    # the (weak, k=1-misleading) vote family is optional; every candidate adds at
    # least one of the two new evidence families.  scheme11's full 6-base config
    # and the literal 8-base fusion are both in the space.
    CANDIDATES = [
        ('content4',            [0, 1, 2, 3]),
        ('scheme11_6',          [0, 1, 2, 3, 4, 5]),
        ('content4+shape',      [0, 1, 2, 3, 6]),
        ('content4+ngsc',       [0, 1, 2, 3, 7]),
        ('content4+shape+ngsc', [0, 1, 2, 3, 6, 7]),
        ('scheme11_6+shape',    [0, 1, 2, 3, 4, 5, 6]),
        ('scheme11_6+ngsc',     [0, 1, 2, 3, 4, 5, 7]),
        ('literal_8',           [0, 1, 2, 3, 4, 5, 6, 7]),
    ]
    CAND = {tag: idx for tag, idx in CANDIDATES}
    table, full = {}, {}
    for tag, idx in CANDIDATES:
        r = eval_subset(S_oof, S_val, idx, y_meta, oof_cons, val_cons)
        full[tag] = r
        table[tag] = {'bases': r['bases'], 'n_bases': len(idx), 'C': r['C'],
                      'meta_cv': r['meta_cv'], 'val_full': r['val_full'],
                      'val_test_like': r['val_test_like'],
                      'per_k_test_like': r['per_k_test_like'], 'grid': r['grid']}
        pk = r['per_k_test_like']
        print(f'  cfg {tag:22s} nb={len(idx)} C={r["C"]:<5g} cv={r["meta_cv"]:6.2f}  '
              f'full={r["val_full"]:6.2f}  test_like={r["val_test_like"]:6.2f}  '
              f'k1={pk["1"]["accuracy"]:5.2f}  k2={pk["2"]["accuracy"]:5.2f}  '
              f'k3={pk["3"]["accuracy"]:6.2f}')
    ablations['A_config_table'] = table

    # ---------------- meta loss family (pointwise vs pairwise) -------------- #
    ablations['A5_meta_family'] = {}
    for kind in ('point', 'pair'):
        r = eval_subset(S_oof, S_val, CAND['content4+shape+ngsc'], y_meta, oof_cons, val_cons, kind=kind)
        ablations['A5_meta_family'][f'{kind}_deployed_set'] = {
            'meta_cv': r['meta_cv'], 'val_full': r['val_full'], 'val_test_like': r['val_test_like']}
    print('A5 meta family:', json.dumps(ablations['A5_meta_family']))

    # ---------------- CORE: scheme-11 config vs scheme-16 ------------------- #
    s11, dep = table['scheme11_6'], table['content4+shape+ngsc']
    ablations['A_core_11_vs_16'] = {
        'scheme11_config_6bases': s11,
        'scheme16_deployed_config': dep,
        'literal_8base_fusion': table['literal_8'],
        'delta_deployed_vs_scheme11': round(dep['val_test_like'] - s11['val_test_like'], 3),
        'marginal_shape_added_to_content4': round(
            table['content4+shape']['val_test_like'] - table['content4']['val_test_like'], 3),
        'marginal_ngsc_added_to_content4': round(
            table['content4+ngsc']['val_test_like'] - table['content4']['val_test_like'], 3),
        'marginal_shape_added_to_scheme11': round(
            table['scheme11_6+shape']['val_test_like'] - s11['val_test_like'], 3),
        'marginal_ngsc_added_to_scheme11': round(
            table['scheme11_6+ngsc']['val_test_like'] - s11['val_test_like'], 3),
    }
    print('CORE:', json.dumps({k: v for k, v in ablations['A_core_11_vs_16'].items()
                               if k.endswith(('marginal_shape_added_to_content4',
                                              'marginal_ngsc_added_to_content4'))}))
    print('  scheme11_6 :', json.dumps({k: s11[k] for k in ('C', 'meta_cv', 'val_full', 'val_test_like')}))
    print('  deployed   :', json.dumps({k: dep[k] for k in ('C', 'meta_cv', 'val_full', 'val_test_like')}))

    # ---------------- two-stage selection ----------------------------------- #
    best_cv = max(table[t]['meta_cv'] for t in table)
    band = [t for t in table if table[t]['meta_cv'] >= best_cv - CV_TIE_BAND]
    chosen = max(band, key=lambda t: (table[t]['val_test_like'],
                                      table[t]['per_k_test_like']['1']['accuracy']))
    chosen_C = table[chosen]['C']
    dep_idx = CAND[chosen]
    print(f'meta-CV best={best_cv:.3f}; tie band={band}; CHOSEN={chosen} C={chosen_C}')

    # ---------------- A4: leave-one-base-out (deployed config) --------------- #
    base_score = table[chosen]['val_test_like']
    loo = {}
    for drop in dep_idx:
        keep = [b for b in dep_idx if b != drop]
        r = eval_subset(S_oof, S_val, keep, y_meta, oof_cons, val_cons)
        loo[BASE_NAMES[drop]] = {
            'meta_cv': r['meta_cv'], 'val_test_like': r['val_test_like'],
            'val_full': r['val_full'],
            'marginal_vs_deployed': round(base_score - r['val_test_like'], 3)}
        print(f'  LOO drop {BASE_NAMES[drop]:10s} cv={r["meta_cv"]:6.2f}  '
              f'val_test_like={r["val_test_like"]:6.2f}  (marginal +{base_score-r["val_test_like"]:.2f})')
    ablations['A4_leave_one_base_out'] = {
        'deployed': chosen, 'deployed_val_test_like': base_score,
        'kind': 'pair', 'C': chosen_C, 'drops': loo}

    # ---------------- A2: OOF vs in-fold (leakage) -------------------------- #
    dep_names = [BASE_NAMES[i] for i in dep_idx]
    Fo, nm = MF.meta_design(S_oof[:, :, dep_idx], {'vote': vote_tr, 'group': grp_tr}, dep_names, True)
    Fv, _ = MF.meta_design(S_val[:, :, dep_idx], {'vote': vblocks_va, 'group': gblocks_va}, dep_names, True)
    cols = MF.z_ntop_cons_cols(nm, len(dep_names))
    leak = {}
    for label, So in (('oof', S_oof[:, :, dep_idx]), ('in_fold', S_in[:, :, dep_idx])):
        Fo_l, _ = MF.meta_design(So, {'vote': vote_tr, 'group': grp_tr}, dep_names, True)
        cv = meta_cv('pair', chosen_C, Fo_l, y_meta, cols)
        mu_l, sd_l, w_l, b_l = fit_meta('pair', chosen_C, Fo_l, y_meta, cols)
        s_tr = apply_meta('pair', mu_l, sd_l, w_l, b_l, Fo_l, cols)
        insample = 100.0 * float(y_meta[np.arange(len(y_meta)), s_tr.argmax(1)].mean())
        ch = apply_meta('pair', mu_l, sd_l, w_l, b_l, Fv, cols).argmax(1).tolist()
        leak[label] = {'meta_train_in_sample_acc': round(insample, 3),
                       'meta_5fold_cv_acc': round(cv, 3),
                       'val_full': round(harness.accuracy(ch, bank_va, gold_va)['accuracy'], 3),
                       'val_test_like': round(harness.accuracy(ch, bank_va, gold_va, mask_tl)['accuracy'], 3)}
        print(f'  leakage {label:8s} in-sample={insample:6.2f}  cv={cv:6.2f}  '
              f'val_test_like={leak[label]["val_test_like"]:6.2f}')
    ablations['A2_oof_vs_infold'] = leak

    # ---------------- deployed scores --------------------------------------- #
    mu, sd, w, b = fit_meta('pair', chosen_C, Fo, y_meta, cols)
    choices = full[chosen]['choices']
    full_acc = harness.accuracy(choices, bank_va, gold_va)
    tl = harness.accuracy(choices, bank_va, gold_va, mask_tl)
    perk = per_k_accuracy(choices, bank_va, gold_va, y_va, tl_idx)
    perk_full = per_k_accuracy(choices, bank_va, gold_va, y_va, np.arange(len(bank_va)))
    print(f'DEPLOYED {chosen} C={chosen_C}  full={full_acc["accuracy"]:.3f}  test_like={tl["accuracy"]:.3f}')
    print('per-k (test_like):', json.dumps(perk))

    # ---------------- permutation invariance self-check --------------------- #
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(4)
    bank_va_p = []
    for row in bank_va:
        r = dict(row)
        r['candidates'] = [row['candidates'][j] for j in perm]
        bank_va_p.append(r)
    blocks_va_p = MF.build_blocks(bank_va_p, arr_va['embeddings'][:, perm, :])
    perm_scores = np.zeros((len(bank_va_p), 4, len(dep_idx)))
    for col, b in enumerate(dep_idx):
        name, keys, kd, _ = BASE_SPECS[b]
        Xp = MF.assemble(blocks_va_p, keys)
        if kd == 'raw':
            perm_scores[:, :, col] = Xp[:, :, 0]
            continue
        mu_b, sd_b, coef_b, bi_b = B['params'][name]
        perm_scores[:, :, col] = score_base(Xp, mu_b, sd_b, coef_b, bi_b)
    Fp, _ = MF.meta_design(perm_scores, {'vote': blocks_va_p['vote'],
                                         'group': blocks_va_p['group']}, dep_names, True)
    s_p = apply_meta('pair', mu, sd, w, b, Fp, cols)
    pick_p = s_p.argmax(1)
    # equivariance: score of permuted position p must equal score of the original
    # candidate that now sits at p, i.e. s_p[i, p] == s_val[i, perm[p]]
    s_val_dep = apply_meta('pair', mu, sd, w, b, Fv, cols)
    equiv = float(np.max(np.abs(s_p - s_val_dep[:, perm])))
    changed_idx = changed_ans = 0
    for i in range(len(bank_va_p)):
        # new position pick_p[i] holds the original candidate perm[pick_p[i]]
        changed_idx += int(int(perm[pick_p[i]]) != choices[i])
        a1 = MF.parse_answer(bank_va_p[i]['candidates'][int(pick_p[i])]['text'])
        a2 = MF.parse_answer(bank_va[i]['candidates'][choices[i]]['text'])
        changed_ans += int(a1 != a2)
    audit = {'permutation': [int(x) for x in perm], 'index_changed': changed_idx,
             'answer_changed': changed_ans, 'score_max_abs_diff': equiv,
             'n': len(bank_va_p),
             'choice_histogram': {str(k): choices.count(k) for k in range(4)}}
    print('permutation check:', audit)

    # ---------------- failure inspection ------------------------------------ #
    failures = []
    for i in tl_idx:
        c = choices[i]
        a = MF.parse_answer(bank_va[i]['candidates'][c]['text'])
        if a is not None and a == gold_va[i]:
            continue
        bases_pick = [int(S_val[i, :, bb].argmax()) for bb in range(8)]
        failures.append({'q': int(i), 'question': bank_va[i]['question'][:200], 'gold': gold_va[i],
                         'picked_idx': c, 'picked': a,
                         'parsed': [MF.parse_answer(cc['text']) for cc in bank_va[i]['candidates']],
                         'base_picks': dict(zip(BASE_NAMES, bases_pick)),
                         'all_bases_agree': len(set(bases_pick)) == 1})
    all_agree = [f for f in failures if f['all_bases_agree']]
    print(f'failures on test_like: {len(failures)}  all-bases-agree: {len(all_agree)}')

    # ---------------- freeze artifacts -------------------------------------- #
    art = HERE / 'artifacts'
    art.mkdir(exist_ok=True)
    for stale in list(art.glob('base_*.npz')) + [art / 'meta.npz', art / 'config.json',
                                                 art / 'SHA256SUMS.txt']:
        if stale.exists():
            stale.unlink()
    cfg_bases = []
    for b in dep_idx:
        name, keys, kd, _ = BASE_SPECS[b]
        if kd == 'raw':
            cfg_bases.append({'name': name, 'blocks': list(keys), 'kind': 'raw',
                              'C': None, 'file': None})
            continue
        mu_b, sd_b, coef_b, bi_b = B['params'][name]
        fname = f'base_{name}.npz'
        np.savez(art / fname, mu=mu_b.astype(np.float32), sd=sd_b.astype(np.float32),
                 coef=coef_b.astype(np.float32), intercept=np.float64(bi_b))
        cfg_bases.append({'name': name, 'blocks': list(keys), 'kind': kd,
                          'C': Cs[name], 'file': fname})

    np.savez(art / 'meta.npz', mu=mu.astype(np.float64), sd=sd.astype(np.float64),
             w=w.astype(np.float64), intercept=np.float64(b))
    config = {
        'scheme': '16_final_fusion', 'seed': SEED, 'deployed': chosen,
        'base_names': dep_names, 'bases': cfg_bases,
        'meta': {'kind': 'pair', 'spec': 'z_ntop_cons', 'include_consensus': True,
                 'cols': [int(c) for c in cols], 'C': chosen_C, 'n_features': int(Fo.shape[2])},
        'oof': {'n_folds': K_FOLDS, 'n_train_questions': int(len(y_meta))},
        'selection_rule': 'meta C by question-level 5-fold CV on the train OOF matrix '
                          '(tie band 0.2), tie broken on validation (protocol-sanctioned); '
                          'test never consulted; no position / id prior',
        'assets_limit_bytes': 16777216,
    }
    (art / 'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n')
    lines = [f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}'
             for p in sorted(art.iterdir()) if p.is_file() and p.name != 'SHA256SUMS.txt']
    (art / 'SHA256SUMS.txt').write_text('\n'.join(lines) + '\n')
    assets_bytes = sum(p.stat().st_size for p in art.rglob('*') if p.is_file())
    print(f'artifacts: {assets_bytes} bytes')

    # ---------------- persist results --------------------------------------- #
    res = HERE / 'results'
    res.mkdir(exist_ok=True)
    out = {'scheme': '16_final_fusion', 'seed': SEED,
           'n_train': int(len(y_tr)), 'n_val': int(len(bank_va)),
           'test_like_size': int(mask_tl.sum()), 'n_aligned_train': int(len(idx_aligned)),
           'base_C_selected_on_train_cv': Cs, 'base_C_curves': curves,
           'deployed': {'config': chosen, 'kind': 'pair', 'spec': 'z_ntop_cons', 'C': chosen_C,
                        'bases': dep_names,
                        'val_full': round(full_acc['accuracy'], 3),
                        'val_test_like': round(tl['accuracy'], 3),
                        'n_correct_full': full_acc['correct'], 'n_correct_test_like': tl['correct'],
                        'per_k_test_like': perk, 'per_k_full': perk_full},
           'ablations': ablations, 'audit': audit,
           'selection': {'best_meta_cv': round(best_cv, 3), 'tie_band': CV_TIE_BAND,
                         'tied': band, 'chosen': chosen, 'chosen_C': chosen_C},
           'assets_bytes': assets_bytes,
           'elapsed_seconds': round(time.perf_counter() - t0, 1)}
    (res / 'ablations.json').write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n')

    md = ['# scheme 16 -- failure samples (validation test_like)', '',
          f'deployed: {chosen}  test_like={tl["accuracy"]:.2f} ({tl["correct"]}/{tl["n"]})  '
          f'full={full_acc["accuracy"]:.2f}', '']
    for f in (all_agree[:6] + [x for x in failures if not x['all_bases_agree']][:3]):
        md.append(f"- **gold `{f['gold']}`** | picked idx {f['picked_idx']} `{f['picked']}` | "
                  f"parsed {f['parsed']} | base picks {f['base_picks']} | all_agree={f['all_bases_agree']}")
        md.append(f"  - Q: {f['question']}")
    (res / 'failures.md').write_text('\n'.join(md) + '\n')
    print(f'wrote {res/"ablations.json"}  total {out["elapsed_seconds"]}s')


if __name__ == '__main__':
    main()
