# DASE7506 MP2 — submission

**Frozen full-test selection accuracy: `83.2976%` (389/467)** — supplied plurality-vote baseline: `69.16%`.

| | measured | limit |
|---|---:|---:|
| Complete test evaluation, fresh prediction process + independent scorer, 4 threads | **3.81 s** (predict 3.375 s + score 0.437 s, wall clock incl. startup) | ≤ 30 s |
| Inference assets, uncompressed | **60,125 B** (9 files) | ≤ 16 MiB |
| Threads used during prediction and scoring | **4** | ≤ 4 |

The score above is the `accuracy` field of `results/official/test-score.json`, produced by the
supplied `code/score_predictions.py` on the frozen checkpoint
`solutions/16_final_fusion/artifacts/`. The same predictions were produced by a fresh process and
compared with the archived run used during development: **0 of 467 chosen indices differ**.

The timing figure is local (Intel Core i5-11300H, 4 cores / 8 threads, 4 threads pinned via
OMP/OpenBLAS/MKL/NumExpr). The reference Xeon Platinum 8457C was not available to us; the
in-process prediction time recorded by the official runner is 2.67 s and scoring 0.048 s, both far
below the 30 s gate. Eligibility remains determined by the reference machine.

**Submitted code version:** the `submitted` tag in this repository.
**Matching checkpoint:** `solutions/16_final_fusion/artifacts/` at that tag (9 files, 60,125 B,
SHA-256 hashes in `artifacts/SHA256SUMS.txt`).
**Checkpoint download (single file):**
<https://github.com/JINWEI-1031/mp2-dase7506/releases/download/submitted/s16_artifacts.zip> —
35,938 B zip, 60,125 B uncompressed, SHA-256
`2f612ff7fe73b3b82040e339b64ca377e2442c28c5dad60418f23d5e461a2c13`. The same files are also
browsable in `solutions/16_final_fusion/artifacts/`.

---

## What is here

| path | contents |
|---|---|
| `REPORT_en.pdf` / `REPORT_en.tex` | the 9-page report (NeurIPS-style short paper; 2 figures, 6 tables) |
| `GUIDE.md` | the assignment guide, as supplied |
| `code/` | the supplied release code and data, plus the implemented `submission.py` |
| `solutions/16_final_fusion/` | the submitted method: features, training, frozen checkpoint, results |
| `solutions/00_shared/harness.py` | development harness used for the validation hard-subset metric |
| `results/official/` | official-pipeline JSON from the final verification (validation and test) |
| `figures/` | the two report figures as PNG |
| `requirements-train.txt` | extra dependency needed only for retraining |

The data in `code/data/` are the supplied fixed benchmark (train 4,755 / validation 798 / test 467
questions, four candidates each, frozen `[N,4,1024]` embeddings). `code/check_data.py` verifies
their hashes.

## What this submission adds

* `code/submission.py` — was an empty interface in the release; it now delegates to the frozen
  selector in `solutions/16_final_fusion/method.py`. `PACKAGE_MANIFEST.json` still records the
  release hash; this is the one supplied file intentionally changed, because the release asks the
  student to implement it.
* `solutions/16_final_fusion/` — `f16_features.py` (feature extraction), `method.py` (inference),
  `train.py` (training + ablations), `artifacts/` (frozen checkpoint), `results/` (recorded
  JSON and failure cases), `REPORT.md` (development log).
* `REPORT_en.pdf` / `REPORT_en.tex` and `figures/` — the submitted report.
* `results/official/` — official evaluator output from the final verification.

Everything else in `code/` is the supplied release and is left unchanged: `common.py`,
`predict.py`, `score_predictions.py`, `check_data.py`, `data/`, `requirements.txt` and
`PACKAGE_MANIFEST.json`.

## Reproduce the score (no retraining)

Python 3.12 is recommended. The released `code/requirements.txt` pins `numpy==2.5.3`; our final
verification was run with Python 3.11.4 + NumPy 1.26.4 and produced identical choices, so the
result does not depend on the minor NumPy version.

```bash
cd code
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
python -m pip install -r requirements.txt
python check_data.py

# validation (full split and hard subset are scored by the official scorer / harness)
python predict.py --method submission.py \
  --checkpoint ../solutions/16_final_fusion/artifacts \
  --split validation --output ../results/validation-predictions.json
python score_predictions.py --split validation \
  --predictions ../results/validation-predictions.json \
  --output ../results/validation-score.json

# test (run once, after freezing)
python predict.py --method submission.py \
  --checkpoint ../solutions/16_final_fusion/artifacts \
  --split test --output ../results/test-predictions.json
python score_predictions.py --split test \
  --predictions ../results/test-predictions.json \
  --output ../results/test-score.json
```

Set `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4` to match the
four-thread evaluation rule. The equivalent explicit entry point is
`--method ../solutions/16_final_fusion/method.py`.

## Retrain (optional)

```bash
python -m pip install -r requirements-train.txt
cd solutions/16_final_fusion
python train.py
```

Training uses only the training split, seed `20260916`, 10-fold out-of-fold scores, and takes
about six minutes on CPU (63.8 s with the base-score cache warm). Validation is used only to break
ties between configurations whose training cross-validation is within 0.2 points; test references
are never used for training, tuning or selection.

## Report and resource accounting

`REPORT_en.pdf` documents the method, the validation/test comparison with the previous best stacker,
the ablations, the failure cases, the resource use and the limitations. Headline result: **83.30%
on test (389/467), exactly equal to the previous best stacker although the two systems disagree on
42 of 467 questions; validation hard-subset accuracy improves from 86.48% to 88.17%.** The report
treats the missing test gain as a finding, not as an omission.

## AI and reuse disclosure

AI coding assistants were used to draft implementation code, analysis scripts and parts of the
report. The headline validation and test numbers were re-run against the official evaluator before
submission; ablation and diagnostic numbers are taken from the recorded JSON shipped in
`solutions/16_final_fusion/results/`. No external dataset or additional pretrained weights were
used, and no model is called at inference time.

## Attribution

Questions and references derive from GSM8K (MIT license). Candidate solutions and the frozen
embeddings were prepared with Qwen3-0.6B; no model weights are redistributed. The supplied
`code/data/LICENSE.txt` and `code/data/NOTICE.txt` are kept unchanged.
