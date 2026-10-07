from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
# The delivery package is intentionally flat: audio_motion_evaluator/ sits
# directly below Audio_Motion_Evaluator_Delivery/.
WORKSPACE_ROOT = PACKAGE_DIR.parent


@dataclass(frozen=True)
class VoiceSpec:
    name: str
    gender: str


@dataclass(frozen=True)
class ProjectConfig:
    """Single source of truth for reproducible preprocessing and training."""

    workspace_root: Path = WORKSPACE_ROOT
    batch_root: Path = WORKSPACE_ROOT / "Evaluator" / "motion_example_0723_00000"
    artifact_root: Path = WORKSPACE_ROOT
    omg_root: Path = WORKSPACE_ROOT / "OMG"
    omg_checkpoint: Path = WORKSPACE_ROOT / "OMG" / "models" / "evaluator" / "step_004000.pt"
    derived_fps: int = 30
    window_frames: int = 60
    window_stride: int = 30
    audio_sample_rate: int = 16_000
    whisper_model: str = "openai/whisper-small"
    whisper_revision: str = "973afd24965f72e36ca33b3055d56a652f456b4d"
    embedding_dim: int = 512
    retrieval_batch_size: int = 32
    voices: tuple[VoiceSpec, ...] = field(
        default_factory=lambda: (
            VoiceSpec("en-US-AriaNeural", "female"),
            VoiceSpec("en-US-JennyNeural", "female"),
            VoiceSpec("en-US-GuyNeural", "male"),
            VoiceSpec("en-GB-RyanNeural", "male"),
        )
    )
    rates: tuple[int, ...] = (-10, -5, 0, 5, 10)
    pitches: tuple[int, ...] = (-5, -2, 0, 2, 5)

    @property
    def raw_tts_dir(self) -> Path:
        return self.artifact_root / "raw_tts_mp3"

    @property
    def derived_motion_dir(self) -> Path:
        return self.artifact_root / "derived_motion_30fps"

    @property
    def feature_dir(self) -> Path:
        return self.artifact_root / "features"

    @property
    def metadata_dir(self) -> Path:
        return self.artifact_root / "metadata"

    @property
    def checkpoint_dir(self) -> Path:
        return self.artifact_root / "checkpoints"

    @property
    def model_cache_dir(self) -> Path:
        return self.artifact_root / "model_cache"
