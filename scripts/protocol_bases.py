"""
The bases of the protocol cost model which featurize_protocol_data extracts
and condense_protocol_data collects, by name. The features of each basis are
written to DIR/data under its name, so several bases can share DIR, and a new
basis needs no new timings.
"""

from finch.autoschedule.cost_models import Basis, GlobalAffine
from finch.autoschedule.formatter.smart_formatter import LEVEL_BASIS

BASES: dict[str, Basis] = {
    # The formatter's basis, and a constant for each program.
    "level": GlobalAffine(LEVEL_BASIS),
}


def features_kind(name: str) -> str:
    """The kind of the features file of the basis name, for path."""
    return f"{name}.features.json"
