# Audio-Motion Evaluator

## 文件结构

以下路径均相对于本目录：

```text
Audio_Motion_Evaluator_Delivery/
├── README.md
├── requirements.txt
├── audio_motion_evaluator/                  # Python API
├── evaluate_audio_motion_artifact.py        # R@1 / MM Distance 评测
├── prepare_mul_aud_omg_artifact.py          # MUL_AUD 数据匹配
└── checkpoints/
    └── audio_motion_evaluator.pt
```

运行时另外需要现有的完整 OMG 仓库，以及其中：OMG/models/evaluator/step_004000.pt

Whisper-small 权重不随本目录提供。首次运行需要能够访问 Hugging Face，程序会自动下载固定 revision 的 `openai/whisper-small`（约 0.9 GB），并默认缓存到：

```text
Audio_Motion_Evaluator_Delivery/model_cache/
```



## 训练方式

- 使用 10 个额外 Motion batch，共 10,240 条数据，不使用 benchmark 测试集训练。
- 将 caption JSON 中同一主体的非空 `action` 按时间排序，以 `" Then, "` 拼接后通过 Edge TTS 生成 speech；一条完整 Motion 对应一条 speech。
- 按 action text 的 `semantic_id` 划分数据：9,467 条 train，773 条 validation。
- Whisper-small 和 OMG Motion Encoder 均冻结，只训练 `Linear(768,512) + LayerNorm` Audio projection 与 CLIP logit scale。
- Loss 为双向 masked InfoNCE：对角线 pair 为 positive；semantic_id 不同的非配对样本为 negative；相同 semantic_id 的非配对样本从分母屏蔽。
- 训练参数：30 epochs、batch size 64、AdamW、learning rate `1e-4`、weight decay `0.01`。

Validation 使用 fixed-batch32、Motion→Audio、严格 diagonal matching，并保证同一 group 内同一个 semantic_id 最多出现一次。最佳 checkpoint 来自 epoch 29：R@1 `0.2500`、R@2 `0.4102`、R@3 `0.4818`、MM Distance `1.2839`。

## 网络结构

```text
Speech
  -> frozen Whisper-small
  -> full-audio chunk pooling
  -> 768-D
  -> Linear(768,512) + LayerNorm + L2 normalize

Motion qpos_36
  -> 30 FPS / 60-frame windows / stride 30
  -> frozen OMG Motion Encoder
  -> window mean + L2 normalize
  -> 512-D
```

- Whisper revision：`973afd24965f72e36ca33b3055d56a652f456b4d`
- OMG checkpoint SHA256：`e346790c5ce33c42840aa19f7fcc2891a4cab27e0d92e23bfaff7d48cb339c40`

## Python API

```python
import sys
from pathlib import Path

import torch

EVALUATOR_ROOT = Path("/path/to/Audio_Motion_Evaluator_Delivery").resolve()
OMG_ROOT = Path("/path/to/OMG").resolve()
sys.path.insert(0, str(EVALUATOR_ROOT))

from audio_motion_evaluator import AudioMotionEvaluator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
evaluator = AudioMotionEvaluator(
    EVALUATOR_ROOT / "checkpoints" / "audio_motion_evaluator.pt",
    omg_root=OMG_ROOT,
    omg_checkpoint=OMG_ROOT / "models" / "evaluator" / "step_004000.pt",
    device=device,
)

audio_embedding, audio_metadata = evaluator.encode_audio(Path("sample.wav"))
motion_embedding, motion_metadata = evaluator.encode_motion(qpos_36, source_fps=50.0)
```

- Audio：speech WAV/MP3，内部派生为 16 kHz mono waveform。
- Motion：NumPy `(T,36)`，顺序为 root xyz、root quaternion wxyz、29 个 joint positions，并传入真实 FPS。
- 输出：Audio 和 Motion 各一个 `(512,)` L2-normalized float32 embedding，以及预处理 metadata。

## 接入 MUL_AUD

```bash
EVALUATOR_ROOT=/path/to/Audio_Motion_Evaluator_Delivery
OMG_ROOT=/path/to/OMG
DATA_ROOT=/path/to/project_data_root
GENERATED_ROOT=/path/to/MUL_AUD/generated_motion
RESULT_ROOT=/path/to/MUL_AUD/evaluation
```

先生成评测 artifact：

```bash
python "$EVALUATOR_ROOT/prepare_mul_aud_omg_artifact.py" \
  --generated-root "$GENERATED_ROOT" \
  --workspace-root "$DATA_ROOT" \
  --task-root "$DATA_ROOT/Tasks/Level1/full_conditioning_reproduction/audio_to_motion_generation/MUL_AUD" \
  --output-dir "$RESULT_ROOT/artifact" \
  --seed 0
```

每个生成 Motion clip 目录需包含 `joint_pos.csv`、`body_pos.csv`、`body_quat.csv` 和 `info.txt`。

随后运行评测：

```bash
python "$EVALUATOR_ROOT/evaluate_audio_motion_artifact.py" \
  --generated-qpos "$RESULT_ROOT/artifact/generated_qpos.npz" \
  --reference-qpos "$RESULT_ROOT/artifact/reference_qpos.npz" \
  --samples-path "$RESULT_ROOT/artifact/samples.jsonl" \
  --checkpoint "$EVALUATOR_ROOT/checkpoints/audio_motion_evaluator.pt" \
  --omg-root "$OMG_ROOT" \
  --omg-checkpoint "$OMG_ROOT/models/evaluator/step_004000.pt" \
  --output-dir "$RESULT_ROOT/audio_semantic" \
  --seed 0
```
