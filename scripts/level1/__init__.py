"""Task-aware Level-1 evaluation and IR₁ scoring."""
from pathlib import Path
import sys

_EVALUATION_ROOT = Path(__file__).resolve().parents[2] / "evaluation"
if str(_EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(_EVALUATION_ROOT))


from .catalog import infer_task_type
from .ir import compute_ir1

__all__ = ["compute_ir1", "infer_task_type"]
