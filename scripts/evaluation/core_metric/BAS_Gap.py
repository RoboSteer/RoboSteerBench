"""Reference-relative beat-alignment metric."""

from __future__ import annotations

from .BAS_Gen import BASGenScore


def bas_gap(generated: BASGenScore | float, reference: BASGenScore | float) -> float:
    """Return ``BAS-Gen - BAS-Reference`` for one matched sample."""
    generated_value = generated.value if isinstance(generated, BASGenScore) else float(generated)
    reference_value = reference.value if isinstance(reference, BASGenScore) else float(reference)
    return float(generated_value - reference_value)
