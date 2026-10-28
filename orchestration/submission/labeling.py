"""Acquisition function: coordinate label purchases across subjects for the first few
labels, then fall back to the random stream.

Items are streamed one at a time. While fewer than `acq_coord_until` labels have been
bought, accept an item when a uniform hash of its content falls below
acq_c * labels_remaining / items_remaining: every pair prefers the same "early" items,
so different subjects' revealed labels overlap on the same items, which the item term
in model.py feeds on. After that, accept everything (random order), which spreads labels
over more items and helps the large budgets. The evaluator force-buys any shortfall.
"""

import hashlib
import json
from pathlib import Path

_P = json.loads((Path(__file__).resolve().parent / "params.json").read_text())
COORD_UNTIL = int(_P.get("acq_coord_until", 7))
ACQ_C = float(_P.get("acq_c", 1.5))


def _u(item) -> float:
    c = item.get("item_content")
    if c is None:
        return 0.5
    h = hashlib.sha1(str(c).encode("utf-8", "ignore")).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def acquisition_function(input, prediction=None, labeled=None, context=None):
    context = context or {}
    rem = context.get("labels_remaining")
    left = context.get("items_remaining")
    budget = context.get("budget")
    if not rem or not left:
        return True
    bought = (budget - rem) if budget is not None else 0
    if bought >= COORD_UNTIL:
        return True
    return _u(input[1]) < min(1.0, ACQ_C * rem / max(left, 1))
