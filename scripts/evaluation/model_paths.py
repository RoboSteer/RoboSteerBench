"""Default model registry for a checkout with an optional weight bundle."""

from pathlib import Path


def default_registry(project_root: Path) -> Path:
    bundle = project_root / "models"
    downloaded = bundle / "registry.json"
    if downloaded.is_file():
        return downloaded
    components = [
        path for path in bundle.glob("*/registry.json") if path.is_file()
    ]
    if len(components) == 1:
        return components[0]
    return project_root / "evaluator_code" / "registry.json"
