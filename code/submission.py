"""Submission entry point for DASE7506 MP2.

The supplied interface is implemented by delegating to the frozen scheme-16 selector in
``solutions/16_final_fusion``. Run from ``code/``:

    python predict.py --method submission.py \
      --checkpoint ../solutions/16_final_fusion/artifacts --split test \
      --output ../results/test-predictions.json

The explicit equivalent is
``--method ../solutions/16_final_fusion/method.py``.
"""
import sys
from pathlib import Path

_SOL = Path(__file__).resolve().parent.parent / "solutions" / "16_final_fusion"
if str(_SOL) not in sys.path:
    sys.path.insert(0, str(_SOL))


def predict(bank, arrays, checkpoint):
    """Return one candidate index in 0..3 per question, in bank order."""
    import method

    return method.predict(bank, arrays, checkpoint)
