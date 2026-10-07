# Rhythm-Motion Evaluator

## 文件结构

以下路径均相对于本目录：

```text
Rhythm_Motion_Evaluator_Delivery/
├── README.md
├── requirements.txt
├── evaluation_results.json
├── checkpoints/
│   └── rhythm_motion_evaluator.pt
└── rhythm_motion_evaluator/
    ├── __init__.py
    ├── inference.py
    ├── metrics.py
    ├── model.py
    ├── omg_encoder.py
    ├── preprocessing.py
    └── runtime.py
```

本目录不包含 OMG 源码、OMG Motion Encoder checkpoint 或 Hugging Face 模型缓存。运行时需要提供完整 OMG 仓库，并确认存在：

```text
OMG/models/evaluator/step_004000.pt
```

其 SHA256 应为：

```text
e346790c5ce33c42840aa19f7fcc2891a4cab27e0d92e23bfaff7d48cb339c40
```

首次运行需要访问 Hugging Face，自动下载固定 revision 的 `MIT/ast-finetuned-audioset-10-10-0.4593`：

```text
f826b80d28226b62986cc218e5cec390b1096902
```

## 训练方式

- 使用 19 个额外 batch：00120、00121、00128–00135、00141–00143、00146、00148–00152，不使用 benchmark 测试集训练。
- 按既有 MUL_RHY Motion Type 白名单和 `audio_valid=true` 筛出 2,836 对，使用 AST 排除 53 条非音乐，最终保留 2,783 对。
- 按 `kinematic_id` 确定性切分：2,535 对训练、248 对 validation，seed 为 `20250815`。
- 冻结 AudioSet AST。完整音乐按 10 秒窗口、5 秒步长编码，汇总为 768-D feature。
- Motion 派生为 30 FPS、60-frame window、stride 30，经冻结的 OMG Motion Encoder 得到 512-D embedding。
- 只训练 `Linear(768,512) + LayerNorm + L2 normalize` 和 CLIP logit scale。
- Loss 为 symmetric masked InfoNCE：严格配对 Audio/Motion 是 positive，不同 `kinematic_id` 是 negative；相同运动的非配对样本不作为 false negative。
- 训练参数为 30 epochs、batch size 64、learning rate `1e-4`、AdamW；best 规则为先最大化 R@1，R@1 相同时最小化 MM Distance，最终选择 epoch 9。

Validation 使用 OMG-compatible fixed-batch32、Motion→Audio、严格 diagonal matching：7 组、224 对、排除余数 24 对，semantic conflict 为 0。

```text
R@1 = 0.174107
R@2 = 0.250000
R@3 = 0.334821
MM Distance = 1.323545
```

完整记录见 `evaluation_results.json`。

## 网络输入输出

Audio 输入为完整音乐 WAV/MP3，内部转换为 16 kHz mono waveform；输出一个 `(512,)` L2-normalized float32 embedding。

Motion 支持 OMG converted-motion 目录：

```text
body_pos.csv + body_quat.csv + joint_pos.csv + info.txt
```

也可以直接输入 NumPy `(T,36)` qpos，排列为：

```text
root xyz + root quaternion wxyz + 29 joint angles
```

并提供真实 FPS。Motion 同样输出一个 `(512,)` L2-normalized float32 embedding。

## Python 使用

先安装与目标 CUDA 匹配的 PyTorch，再安装 `requirements.txt`：

```python
import sys
from pathlib import Path

import numpy as np
import torch

DELIVERY_ROOT = Path("/path/to/Rhythm_Motion_Evaluator_Delivery").resolve()
OMG_ROOT = Path("/path/to/OMG").resolve()
sys.path.insert(0, str(DELIVERY_ROOT))

from rhythm_motion_evaluator import RhythmMotionEvaluator

evaluator = RhythmMotionEvaluator(
    DELIVERY_ROOT / "checkpoints" / "rhythm_motion_evaluator.pt",
    omg_root=OMG_ROOT,
    omg_checkpoint=OMG_ROOT / "models" / "evaluator" / "step_004000.pt",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

audio_embedding, audio_metadata = evaluator.encode_audio(Path("music.mp3"))
motion_embedding, motion_metadata = evaluator.encode_motion_directory(
    Path("generated_motion_directory")
)

# 也可以编码生成模型直接输出的 qpos_36
motion_embedding, motion_metadata = evaluator.encode_qpos(qpos_36, source_fps=30.0)
```

## 接入评测

对同一组 pair 按相同顺序编码输入音乐和生成 Motion。每个 retrieval group 必须恰好包含 32 对，并保证同一组内同一个 semantic/kinematic ID 最多出现一次。

```python
audio_embeddings = np.stack(audio_embeddings)    # (32, 512)
motion_embeddings = np.stack(motion_embeddings)  # (32, 512)

# Motion → Audio，严格 diagonal matching
r1, r2, r3 = evaluator.retrieval_at_k(
    motion_embeddings,
    audio_embeddings,
    top_k=3,
)
mm_distance = evaluator.mm_distance(audio_embeddings, motion_embeddings)
```

如果需要从较大的 validation pool 确定性构造无冲突 fixed-batch32，可调用：

```python
from rhythm_motion_evaluator import evaluate_fixed_groups

result = evaluate_fixed_groups(
    motion_embeddings,
    audio_embeddings,
    group_size=32,
    semantic_ids=semantic_ids,
    dataset_names=dataset_names,
    seed=20250815,
)
```

无法构造无冲突分组时函数会直接报错。评测时应保存返回的最终顺序和被排除 pair。

推理 checkpoint SHA256：

```text
4be52f075a442ddad6808ef223c12ab52c8ddf50eaa62898b41e3052cf33b5b2
```
