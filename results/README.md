# Official evaluation output

| file | contents |
|---|---|
| `test-score.json` | official scorer on the frozen submission, complete test bank: **389/467 = 83.2976%** |
| `test-predictions.json` | the 467 chosen indices that produced that score |
| `validation-score.json` | official scorer on validation: 756/798 = 94.7368% |
| `validation-predictions.json` | the 798 chosen indices for validation |

These files were produced by a fresh run of the supplied `code/predict.py` and
`code/score_predictions.py` on the frozen checkpoint. They match the archived development
predictions (0/467 index changes on test).

The validation *hard-subset* number used in the report (313/355 = 88.17%) is computed by
`solutions/00_shared/harness.py`; it is not a field of the official validation scorer.
