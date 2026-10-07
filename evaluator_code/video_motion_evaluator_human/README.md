# Human Video-Motion Evaluator

## 文件结构

以下路径均相对于本目录：

```text
Human_Video_Motion_Evaluator_Delivery/
├── README.md
├── requirements.txt
├── evaluation_results.json
├── checkpoints/
│   └── human_video_motion_evaluator.pt
└── video_motion_evaluator/
    ├── __init__.py
    ├── inference.py
    ├── model.py
    ├── video.py
    ├── omg_encoder.py
    ├── preprocessing.py
    └── runtime.py
```

本目录不包含 OMG 源码、OMG motion evaluator checkpoint 和 XCLIP 权重。运行时需要提供：

```text
OMG/models/evaluator/step_004000.pt
```

其 SHA256 应为：

```text
e346790c5ce33c42840aa19f7fcc2891a4cab27e0d92e23bfaff7d48cb339c40
```

首次运行会从 Hugging Face 下载固定 revision 的 `microsoft/xclip-base-patch32`，默认缓存到本目录的 `model_cache/`。

## 训练方式与网络

- 使用 10 个额外数据 batch，共 10,240 对 Human video / converted motion，不使用 benchmark 测试集训练。
- 按 Motion 的 `kinematic_id` 确定性切分，seed 为 `20250815`：9,203 对训练、1,037 对 validation。
- Video 按 Motion 时间窗口采样：Motion 派生为 30 FPS、60-frame window、stride 30；每个窗口按真实时间从 Video 采样 8 帧。
- 冻结 `microsoft/xclip-base-patch32`，得到 512-D video feature；仅训练 `Linear(512,512) + LayerNorm + L2 normalize` 和 logit scale。
- Motion 使用冻结的 OMG Motion Encoder，输出 L2-normalized 512-D embedding。
- Loss 为 symmetric masked InfoNCE；对应 Video/Motion 是 positive，batch 内其他不同 `kinematic_id` 的 pair 是 negative，相同 Motion 的非配对项从分母屏蔽。
- 训练参数：30 epochs、batch size 64、AdamW、learning rate `1e-4`、weight decay `0.01`。

最佳 checkpoint 来自 epoch 28。Validation 使用 OMG-compatible fixed-batch32、Motion→HumanVideo、严格 diagonal matching：32 组、1,024 对、排除余数 13 对、semantic conflict 为 0。R@1/R@2/R@3 为 `0.5381/0.7070/0.7949`，MM Distance 为 `1.1859`。完整记录见 `evaluation_results.json`。

## Python 使用

先安装与目标 CUDA 匹配的 PyTorch，再安装 `requirements.txt`：

```python
import sys
from pathlib import Path

import numpy as np
import torch

DELIVERY_ROOT = Path("/path/to/Human_Video_Motion_Evaluator_Delivery").resolve()
OMG_ROOT = Path("/path/to/OMG").resolve()
sys.path.insert(0, str(DELIVERY_ROOT))

from video_motion_evaluator import VideoMotionEvaluator

evaluator = VideoMotionEvaluator(
    DELIVERY_ROOT / "checkpoints" / "human_video_motion_evaluator.pt",
    omg_root=OMG_ROOT,
    omg_checkpoint=OMG_ROOT / "models" / "evaluator" / "step_004000.pt",
    expected_video_kind="human",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

video_embedding, video_metadata = evaluator.encode_video(Path("condition_video.mp4"))
motion_embedding, motion_metadata = evaluator.encode_motion_directory(
    Path("generated_motion_directory")
)

# 也可以直接编码生成模型输出的 qpos_36
motion_embedding, motion_metadata = evaluator.encode_qpos(qpos_36, source_fps=30.0)
```

Motion qpos 排列为 `root xyz + root quaternion wxyz + 29 joint angles`。Video 和 Motion 均输出 `(512,)` L2-normalized float32 embedding，并返回可追溯的预处理 metadata。

## 接入评测

对同一组 pair 按相同顺序分别编码 Human video 和 generated motion。每个 retrieval group 必须恰好包含 32 对，执行 Motion→HumanVideo 严格 diagonal matching：

```python
video_embeddings = np.stack(video_embeddings)    # (32, 512)
motion_embeddings = np.stack(motion_embeddings)  # (32, 512)

r1, r2, r3 = evaluator.retrieval_at_k(
    motion_embeddings,
    video_embeddings,
    top_k=3,
)
mm_distance = evaluator.mm_distance(video_embeddings, motion_embeddings)
```

多组结果按 pair 数量汇总。Video 预处理 metadata 应与评测结果一起保存。

Checkpoint SHA256：

```text
2e629743ccca9f5ae18ff49126bdf746595ab8f0e8360a96edca726d5b8170e6
```
