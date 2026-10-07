"""Pure, model-independent evaluation metric implementations.

Metric modules can require optional scientific/audio dependencies.  Keep the
package import lazy so level-specific CLIs only load the metric they use.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "bas_gap": ("BAS_Gap", "bas_gap"),
    "BASGenEvaluator": ("BAS_Gen", "BASGenEvaluator"),
    "BASGenScore": ("BAS_Gen", "BASGenScore"),
    "bas_gen_from_positions": ("BAS_Gen", "bas_gen_from_positions"),
    "full_pair_diversity_gpu": ("Diversity", "full_pair_diversity_gpu"),
    "motion_fid": ("FID", "motion_fid"),
    "paired_mm_distance": ("MM_Distance", "paired_mm_distance"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value
