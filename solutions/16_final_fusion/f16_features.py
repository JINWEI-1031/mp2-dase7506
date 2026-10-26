"""Self-contained feature builder for scheme 16 (final fusion).

This module is written from scratch for scheme 16.  It imports no other
solution.  It mirrors the evidence families that the earlier rounds found to be
useful and *diverse*:

  TEXT   (31)  surface / self-consistency statistics read off one candidate
  GROUP  ( 9)  within-question relative text (length rank, mode membership, ...)
  VOTE   ( 8)  within-question answer-consensus / plurality statistics
  EMBED  (1024) frozen trajectory embedding, demeaned inside the question
  SHAPE  ( 7)  **new** answer-form plausibility (integer? fraction? negative?
               magnitude relative to the question's numbers; is the answer one
               of the question's own numbers)                       [scheme 14]
  NGSC   ( 9)  **new** question-grounded numerical self-consistency: is the
               candidate's final answer reachable from the question numbers by a
               bounded number of + - * / operations, and are the `LHS = RHS`
               fragments in its text symbolically re-executable?   [scheme 15]

Every statistic depends only on the candidate's own text, the question text and
*order-free aggregate* statistics of the group.  The candidate index 0..3 is
never used, so every feature -- and therefore every base score and the final
choice -- is invariant to a permutation of the four candidates.
"""
from __future__ import annotations

import ast
import operator
import re
from collections import Counter
from fractions import Fraction

import numpy as np

# ---------------------------------------------------------------------------
# official answer parser (byte-compatible with code/common.py:parse_answer)
# ---------------------------------------------------------------------------
_NUM = re.compile(r'[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?')
_PLAIN = re.compile(r'[-+]?\d[\d,]*(?:\.\d+)?')
_BOXED = re.compile(r'\\boxed\{([^{}]+)\}')


def parse_answer(text):
    text = str(text)
    tail = text.rsplit('####', 1)[1] if '####' in text else None
    if tail is None:
        boxed = _BOXED.findall(text)
        tail = boxed[-1] if boxed else ''
    m = _NUM.search(tail)
    if not m:
        return None
    try:
        return str(Fraction(m.group().replace(',', '')))
    except (ValueError, ZeroDivisionError):
        return None


def _frac(x):
    try:
        return Fraction(x)
    except (ValueError, ZeroDivisionError, TypeError):
        return None


def _numbers(text):
    out = []
    for m in _PLAIN.finditer(str(text)):
        v = _frac(m.group().replace(',', ''))
        if v is not None:
            out.append(v)
    return out


_SELF_CORRECT = ('wait', 'actually', 'however', 'hmm', 'let me', 'reconsider',
                 'check', 'but ', 'alternatively', 'instead')


# ---------------------------------------------------------------------------
# question numeric leaves -- used by the SHAPE block
# ---------------------------------------------------------------------------
def question_leaves(question, cap=8):
    """Numbers appearing in the question (int / decimal / thousands / fraction /
    `p%` -> p and p/100), de-duplicated in order of appearance, capped."""
    q = str(question)
    out, seen = [], set()
    for m in _PLAIN.finditer(q):
        raw = m.group().replace(',', '')
        v = _frac(raw)
        if v is None:
            continue
        cands = [v]
        if q[m.end():m.end() + 1] == '%':
            cands.append(v / 100)
        for c in cands:
            if c not in seen:
                seen.add(c)
                out.append(c)
        if len(out) >= cap:
            break
    return out


# ---------------------------------------------------------------------------
# TEXT block -- per candidate, group independent
# ---------------------------------------------------------------------------
TEXT_NAMES = [
    'parse_ok', 'ans_neg', 'ans_frac', 'ans_zero', 'ans_int',
    'log_len', 'n_lines', 'n_eq', 'log_nnum', 'n_boxed',
    'has_hash', 'n_selfcorrect', 'has_final', 'ans_occurs', 'last_num_is_ans',
    'frac_nums_in_q', 'ans_in_q', 'log_tail_len', 'num_dedup_ratio',
    'ends_punct', 'log_abs_ans', 'n_bullets', 'has_money', 'n_plusminus',
    'rhs_last_is_ans', 'n_rhs_is_ans', 'n_distinct_rhs', 'digit_density',
    'tail_has_boxed', 'n_eq_chain', 'frac_rhs_is_ans',
]


def _text_features(question, text, parsed, q_nums):
    t = str(text)
    low = t.lower()
    nums = _numbers(t)
    n_num = len(nums)
    af = None
    if parsed is not None:
        pf = _frac(parsed)
        af = float(pf) if pf is not None else None

    f = []
    f.append(1.0 if parsed is not None else 0.0)
    f.append(1.0 if (af is not None and af < 0) else 0.0)
    f.append(1.0 if (parsed is not None and '/' in str(parsed)) else 0.0)
    f.append(1.0 if (af is not None and af == 0) else 0.0)
    f.append(1.0 if (af is not None and float(af).is_integer()) else 0.0)

    f.append(float(np.log1p(len(t))))
    f.append(float(t.count('\n')))
    f.append(float(t.count('=')))
    f.append(float(np.log1p(n_num)))
    f.append(float(low.count('\\boxed{')))
    f.append(1.0 if '####' in t else 0.0)
    f.append(float(sum(low.count(w) for w in _SELF_CORRECT)))
    f.append(1.0 if ('final answer' in low or 'answer is' in low) else 0.0)

    if af is not None and n_num:
        f.append(float(sum(1 for v in nums if v == Fraction(parsed))))
    else:
        f.append(0.0)
    f.append(1.0 if (af is not None and nums and nums[-1] == Fraction(parsed)) else 0.0)

    if nums:
        uniq = set(nums)
        f.append(sum(1 for v in uniq if v in q_nums) / len(uniq))
    else:
        f.append(0.0)
    f.append(1.0 if (af is not None and Fraction(parsed) in q_nums) else 0.0)

    tail = t.rsplit('####', 1)[1] if '####' in t else t[-64:]
    f.append(float(np.log1p(len(tail))))
    f.append(len(set(nums)) / n_num if n_num else 0.0)
    f.append(1.0 if t.rstrip().endswith(('.', '!', '?')) else 0.0)
    f.append(float(np.log1p(abs(af))) if af is not None else 0.0)
    f.append(float(sum(1 for ln in t.split('\n')
                       if ln.lstrip()[:2] in ('- ', '* ')
                       or (ln.lstrip()[:2].rstrip('.').isdigit() and len(ln.lstrip()) > 2))))
    f.append(1.0 if ('$' in t or 'dollar' in low or 'cent' in low) else 0.0)
    f.append(float(t.count('+') + t.count('-')))

    rhs = []
    for part in t.split('=')[1:]:
        m = _PLAIN.match(part.strip())
        if m:
            v = _frac(m.group().replace(',', ''))
            if v is not None:
                rhs.append(v)
    tgt = Fraction(parsed) if af is not None else None
    f.append(1.0 if (tgt is not None and rhs and rhs[-1] == tgt) else 0.0)
    f.append(float(sum(1 for v in rhs if tgt is not None and v == tgt)))
    f.append(float(len(set(rhs))))
    f.append(sum(ch.isdigit() for ch in t) / max(1, len(t)))
    f.append(1.0 if ('\\boxed' in t[-80:] or '####' in t[-20:]) else 0.0)
    f.append(float(max(0, t.count('=') - 1)))
    f.append((sum(1 for v in rhs if tgt is not None and v == tgt) / len(rhs)) if rhs else 0.0)
    return f


# ---------------------------------------------------------------------------
# GROUP block -- within-question relative text
# ---------------------------------------------------------------------------
GROUP_NAMES = [
    'ans_is_mode', 'mode_share', 'n_distinct_ans', 'len_rank', 'len_dev',
    'ans_share', 'ans_singleton', 'ans_mag_rank', 'ans_logmag_dev',
]


def _avg_rank(x):
    x = np.asarray(x, dtype=np.float64)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    start = cum - counts
    avg = (start + cum - 1) / 2.0
    return avg[inv]


def _group_block(texts, parsed):
    n = len(texts)
    valid = [p for p in parsed if p is not None]
    cnt = Counter(valid)
    n_distinct = len(cnt)
    if cnt:
        top = max(cnt.values())
        winners = [v for v, c in cnt.items() if c == top]
        mode_val = winners[0] if len(winners) == 1 else None
    else:
        top, mode_val = 0, None

    lengths = [len(str(t)) for t in texts]
    len_rank = _avg_rank(lengths)
    mean_len = float(np.mean(lengths)) if lengths else 0.0
    std_len = float(np.std(lengths)) + 1e-6

    mags = [float(_frac(p)) if p is not None else None for p in parsed]
    logmags = [np.log1p(abs(v)) if v is not None else -1e9 for v in mags]
    finite = [v for v in logmags if v > -1e8]
    m_mu = float(np.mean(finite)) if finite else 0.0
    m_sd = float(np.std(finite)) + 1e-6
    mag_rank = _avg_rank(logmags)

    rows = []
    for j in range(n):
        p = parsed[j]
        c = cnt.get(p, 0) if p is not None else 0
        lm = logmags[j]
        rows.append([
            1.0 if (p is not None and p == mode_val and c == top) else 0.0,
            float(top) / n,
            float(n_distinct),
            float(len_rank[j]) / max(1, n - 1),
            (lengths[j] - mean_len) / std_len,
            float(c) / n,
            1.0 if (p is not None and c == 1) else 0.0,
            float(mag_rank[j]) / max(1, n - 1),
            ((lm - m_mu) / m_sd) if lm > -1e8 else 0.0,
        ])
    return rows


# ---------------------------------------------------------------------------
# VOTE block -- explicit answer-consensus statistics
# ---------------------------------------------------------------------------
VOTE_NAMES = [
    'vote_share', 'vote_count', 'is_mode', 'vote_margin', 'is_tied',
    'count_gap_mode', 'ans_outlier', 'parse_fail',
]


def _vote_block(texts, parsed):
    n = len(texts)
    valid = [p for p in parsed if p is not None]
    cnt = Counter(valid)
    n_valid = len(valid)
    n_distinct = len(cnt)
    top = max(cnt.values()) if cnt else 0
    rows = []
    for j in range(n):
        p = parsed[j]
        c = cnt.get(p, 0) if p is not None else 0
        others = [v for k, v in cnt.items() if k != p] if p is not None else list(cnt.values())
        other_max = max(others) if others else 0
        rows.append([
            (float(c) / n_valid) if n_valid else 0.0,
            float(c),
            1.0 if (p is not None and c == top and top > 0 and len(
                [v for v, k in cnt.items() if k == top]) == 1) else 0.0,
            float(c - other_max),
            1.0 if (p is not None and c >= 2 and sum(1 for k in cnt.values() if k == c) >= 2) else 0.0,
            float(c - top),
            1.0 if (p is not None and c == 1 and n_distinct >= 2) else 0.0,
            1.0 if p is None else 0.0,
        ])
    return rows


# ===========================================================================
# SHAPE block -- answer-form plausibility (scheme 14's "answer shape")
# ===========================================================================
SHAPE_NAMES = ['is_int', 'is_frac', 'is_neg', 'log_mag', 'mag_ratio',
               'ans_density', 'ans_leaf_cnt']


def _shape_block_one(text, parsed, leaves):
    v = _frac(parsed) if parsed is not None else None
    if v is None:
        return [0.0] * len(SHAPE_NAMES)
    mag = abs(float(v))
    maxleaf = max((abs(float(x)) for x in leaves), default=0.0)
    toks = _numbers(text)
    density = (sum(1 for t in toks if t == v) / (len(toks) + 1.0)) if toks else 0.0
    return [
        1.0 if v.denominator == 1 else 0.0,
        1.0 if v.denominator != 1 else 0.0,
        1.0 if v < 0 else 0.0,
        float(np.log1p(mag)),
        mag / (maxleaf + 1e-9),
        density,
        float(sum(1 for x in leaves if x == v)),
    ]


# ===========================================================================
# NGSC block -- question-grounded numerical self-consistency (scheme 15)
# ===========================================================================
NGSC_NAMES = [
    'ng_reach_self', 'ng_reach_1', 'ng_reach_2', 'ng_reach_depth',
    'ng_reach_logsize', 'ng_eq_n', 'ng_eq_valid_frac', 'ng_eq_all_valid',
    'ng_final_eq_last',
]

_CLOSURE_CAP = 20000
_LATEX = [(r'\\times', '*'), (r'\\cdot', '*'), (r'\\div', '/'),
          (r'\\left', ''), (r'\\right', ''), (r'\\,', ''), (r'\\!', ''),
          (r'\\text\{[^{}]*\}', ''), (r'\\mathrm\{[^{}]*\}', ''),
          (r'\\%', '/100'), (r'%', '/100'), (r'\$', ''),
          (r'\\frac\{([^{}]*)\}\{([^{}]*)\}', r'((\1)/(\2))')]
_EQ = re.compile(r'([^=\n]{1,120}?)\s*=\s*([^=\n]{1,40})')
_ALLOWED = re.compile(r'^[\d\s\.\+\-\*/\(\)]+$')

_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub,
           ast.Mult: operator.mul, ast.Div: operator.truediv}


def _ev(node):
    """Safe arithmetic evaluation of a parsed AST node -> Fraction or None."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return None
        if isinstance(node.value, (int, float)):
            return Fraction(node.value).limit_denominator(10 ** 6)
        return None
    if isinstance(node, ast.UnaryOp):
        v = _ev(node.operand)
        if v is None:
            return None
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return v
        return None
    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            return None
        a = _ev(node.left)
        b = _ev(node.right)
        if a is None or b is None:
            return None
        if isinstance(node.op, ast.Div) and b == 0:
            return None
        try:
            return op(a, b)
        except (ZeroDivisionError, OverflowError):
            return None
    return None


def _eval_expr(expr):
    expr = expr.strip().strip(' +*')
    if not expr or not _ALLOWED.match(expr):
        return None
    try:
        tree = ast.parse(expr, mode='eval')
    except SyntaxError:
        return None
    v = _ev(tree.body)
    if v is None:
        return None
    return v.limit_denominator(10 ** 6)


def _clean(s):
    for pat, rep in _LATEX:
        s = re.sub(pat, rep, s)
    return s.replace(',', '').replace('{', '').replace('}', '')


def _equations(text):
    """List of (lhs, rhs, valid) for every 'LHS = RHS' fragment in the text."""
    out = []
    for m in _EQ.finditer(_clean(str(text))):
        lv = _eval_expr(m.group(1))
        rv = _eval_expr(m.group(2))
        if lv is None or rv is None:
            continue
        out.append((lv, rv, lv == rv))
    return out


def closure_levels(qnums, depth=2, cap=_CLOSURE_CAP):
    """R0 = question numbers; R_d = combine(R_{d-1}, R0).  Returns [R0, R1, ...]."""
    S0 = set(qnums)
    levels = [S0]
    cur = set(S0)
    for _ in range(depth):
        new = set()
        for u in cur:
            for b in S0:
                outs = [u + b, u - b, u * b]
                if b != 0:
                    outs.append(u / b)
                for r in outs:
                    if abs(r.numerator) > 10 ** 9 or abs(r.denominator) > 10 ** 9:
                        continue
                    new.add(r)
            if len(new) > cap:
                break
        levels.append(new)
        cur = new
        if len(new) > cap:
            break
    return levels


def _ngsc_block_one(text, parsed, levels):
    tgt = _frac(parsed) if parsed is not None else None
    R0 = levels[0]
    R1 = levels[1] if len(levels) > 1 else R0
    R2 = levels[2] if len(levels) > 2 else R1

    if tgt is None:
        reach_self = reach1 = reach2 = 0.0
        depth = 3.0
    else:
        reach_self = 1.0 if tgt in R0 else 0.0
        reach1 = 1.0 if tgt in R1 else 0.0
        reach2 = 1.0 if tgt in R2 else 0.0
        if tgt in R0:
            depth = 0.0
        elif tgt in R1:
            depth = 1.0
        elif tgt in R2:
            depth = 2.0
        else:
            depth = 3.0

    eqs = _equations(text)
    n_eq = len(eqs)
    n_valid = sum(1 for _, _, ok in eqs if ok)
    rhs_vals = [rv for _, rv, _ in eqs]
    final_eq_last = 1.0 if (tgt is not None and rhs_vals and rhs_vals[-1] == tgt) else 0.0
    return [
        reach_self,
        reach1,
        reach2,
        (2.0 - min(depth, 2.0)) / 2.0,
        float(np.log1p(len(R2))),
        float(n_eq),
        (n_valid / n_eq) if n_eq else 0.0,
        1.0 if (n_eq and n_valid == n_eq) else 0.0,
        final_eq_last,
    ]


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
BLOCK_DIMS = {'text': len(TEXT_NAMES), 'group': len(GROUP_NAMES),
              'vote': len(VOTE_NAMES), 'shape': len(SHAPE_NAMES),
              'ngsc': len(NGSC_NAMES), 'embed': None}


def build_blocks(bank, embeddings, demean_embed=True):
    """Return dict of float64 blocks; index 0..3 is never used (permutation
    invariant by construction).

    text [N,4,31]  group [N,4,9]  vote [N,4,8]  shape [N,4,7]  ngsc [N,4,9]
    embed [N,4,1024]
    """
    n = len(bank)
    text = np.zeros((n, 4, len(TEXT_NAMES)), dtype=np.float64)
    group = np.zeros((n, 4, len(GROUP_NAMES)), dtype=np.float64)
    vote = np.zeros((n, 4, len(VOTE_NAMES)), dtype=np.float64)
    shape = np.zeros((n, 4, len(SHAPE_NAMES)), dtype=np.float64)
    ngsc = np.zeros((n, 4, len(NGSC_NAMES)), dtype=np.float64)

    for i, row in enumerate(bank):
        q_nums_list = question_leaves(row['question'])
        q_nums = set(q_nums_list)
        levels = closure_levels(q_nums_list, depth=2)
        texts = [c['text'] for c in row['candidates']]
        parsed = [parse_answer(t) for t in texts]
        for j in range(4):
            text[i, j] = _text_features(row['question'], texts[j], parsed[j], q_nums)
            shape[i, j] = _shape_block_one(texts[j], parsed[j], q_nums_list)
            ngsc[i, j] = _ngsc_block_one(texts[j], parsed[j], levels)
        for j, gf in enumerate(_group_block(texts, parsed)):
            group[i, j] = gf
        for j, vf in enumerate(_vote_block(texts, parsed)):
            vote[i, j] = vf

    emb = np.asarray(embeddings, dtype=np.float64)
    if demean_embed:
        emb = emb - emb.mean(axis=1, keepdims=True)
    return {'text': text, 'group': group, 'vote': vote, 'shape': shape,
            'ngsc': ngsc, 'embed': emb}


def assemble(blocks, names):
    """names: iterable of block keys -> concatenated [N,4,D] float64."""
    return np.concatenate([blocks[k] for k in names], axis=2)


def candidate_view(X):
    return X.reshape(-1, X.shape[-1])


# ---------------------------------------------------------------------------
# META features -- built on top of the *scores* of the base scorers
# ---------------------------------------------------------------------------
def _avg_rank_groups(a):
    """a [N,K] -> average rank (0-based, ties share the mean rank)."""
    less = (a[:, None, :] < a[:, :, None]).sum(2).astype(np.float64)
    eq = (a[:, None, :] == a[:, :, None]).sum(2).astype(np.float64)
    return less + (eq - 1.0) / 2.0


CONSENSUS_NAMES = ['vote_share', 'is_mode', 'is_tied', 'vote_margin',
                   'ans_outlier', 'parse_fail', 'mode_share', 'n_distinct_ans']


def meta_features(S, blocks, base_names, include_consensus=True):
    """S [N,4,B] base scores -> meta feature tensor [N,4,M].

    Every column is a within-question statistic (z-score / average rank / argmax
    indicator of the base scores) plus a few group consensus statistics, so the
    matrix is invariant to a permutation of the four candidates.
    """
    n, k, nb = S.shape
    parts, names = [], []
    top_stack = []
    for b in range(nb):
        s = S[:, :, b]
        mu = s.mean(1, keepdims=True)
        sd = s.std(1, keepdims=True) + 1e-9
        z = ((s - mu) / sd)[:, :, None]
        r = (_avg_rank_groups(s) / max(1, k - 1))[:, :, None]
        t = (s >= s.max(1, keepdims=True)).astype(np.float64)[:, :, None]
        top_stack.append(t)
        parts += [z, r, t]
        names += [f'z_{base_names[b]}', f'rank_{base_names[b]}', f'top_{base_names[b]}']
    n_top = np.sum(top_stack, axis=0)
    parts.append(n_top)
    names.append('n_bases_top1')
    if include_consensus:
        vote = blocks['vote']
        grp = blocks['group']
        cons = [vote[:, :, 0:1], vote[:, :, 2:3], vote[:, :, 4:5], vote[:, :, 3:4],
                vote[:, :, 6:7], vote[:, :, 7:8],
                grp[:, :, 1:2], grp[:, :, 2:3] / 4.0]
        parts += cons
        names += list(CONSENSUS_NAMES)
    return np.concatenate(parts, axis=2), names


def meta_design(S, blocks, base_names, include_consensus=True):
    """Meta feature matrix + names."""
    return meta_features(S, blocks, base_names, include_consensus=include_consensus)


def z_ntop_cons_cols(names, n_bases, include_consensus=True):
    """The scheme-11 winning meta column subset: base z-scores + n_bases_top1 +
    the 8 group consensus statistics."""
    ncore = 3 * n_bases + 1
    cols = [i for i, nm in enumerate(names) if nm.startswith('z_')]
    cols += [names.index('n_bases_top1')]
    if include_consensus:
        cols += list(range(ncore, len(names)))
    return cols
