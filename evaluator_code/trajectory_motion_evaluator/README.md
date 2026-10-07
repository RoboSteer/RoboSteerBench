# Rotation-Motion Evaluator

## 文件结构

以下路径均相对于本目录：

```text
Rotation_Motion_Evaluator_Delivery/
├── README.md
├── requirements.txt
├── evaluation_results.json
├── checkpoints/
│   └── rotation_motion_evaluator.pt
└── rotation_motion_evaluator/
    ├── __init__.py
    ├── inference.py
    ├── model.py
    ├── omg_encoder.py
    ├── preprocessing.py
    └── runtime.py
```

本目录不包含 OMG 源码及其 251 MB 的 motion evaluator checkpoint。运行时需要另外提供完整 OMG 仓库，并确认存在：

```text
OMG/models/evaluator/step_004000.pt
```

要求该文件的 SHA256 为：

```text
e346790c5ce33c42840aa19f7fcc2891a4cab27e0d92e23bfaff7d48cb339c40
```

## 训练方式

- 使用 `motion_example_0723_00000` 至 `00009`，共 10,240 对 Rotation condition / Motion。
- 按 `kinematic_id` 确定性切分，seed 为 `20250814`：9,223 对训练、1,017 对 validation。
- 两侧原始数据都转换为 `root xyz + root quaternion wxyz + 29-DoF`，再派生为 30 FPS、60-frame window、stride 30；原始数据不被修改。
- 两侧共用冻结的 OMG Motion Encoder，得到 512-D motion embedding。
- 同时提取每帧 38-D rotation feature：相对 root position 3-D、root rotation 6-D、29-DoF；通过共享的 2-layer temporal encoder 得到 256-D feature。
- 将 `512-D OMG embedding + 256-D rotation feature` 输入共享 residual adapter，输出 L2-normalized 512-D embedding。
- Loss 为 symmetric masked InfoNCE，加 `0.25 × corruption margin loss`；对应 condition/motion 是 positive，batch 内其他不同 `kinematic_id` 的 pair 是 negative，重复运动不作为 false negative。
- corruption 包含时间倒放、root trajectory 缩放、root rotation 扰动和 joint-angle 扰动。
- 共训练 30 epochs；best 规则为先最大化 R@1，R@1 相同时最小化 MM Distance，最终选择 epoch 2。

最终 validation 使用 OMG-compatible fixed-batch32、Motion→Condition、严格 diagonal matching：31 组、992 对、余数排除 25 对、组内 kinematic conflict 为 0。结果为 R@1/R@2/R@3 = `1.0/1.0/1.0`，MM Distance = `0.0166958`。完整结果见 `evaluation_results.json`。

## 网络输入输出

Condition 支持 MUL_POS 的 retargeted `.pkl`：

```text
fps + root_pos(T,3) + root_rot(T,4, xyzw) + dof_pos(T,29)
```

Motion 支持 OMG converted-motion 目录：

```text
body_pos.csv + body_quat.csv + joint_pos.csv + info.txt
```

也可以直接输入 NumPy `(T,36)` qpos，排列为 `root xyz + quaternion wxyz + 29 joint angles`，并提供真实 FPS。Condition 和 Motion 均输出一个 `(512,)` L2-normalized float32 embedding。

## Python 使用

先安装与目标 CUDA 匹配的 PyTorch，再安装 `requirements.txt`。将本目录和 OMG 的实际位置传入：

```python
import sys
from pathlib import Path

import numpy as np
import torch

DELIVERY_ROOT = Path("/path/to/Rotation_Motion_Evaluator_Delivery").resolve()
OMG_ROOT = Path("/path/to/OMG").resolve()
sys.path.insert(0, str(DELIVERY_ROOT))

from rotation_motion_evaluator import RotationMotionEvaluator

evaluator = RotationMotionEvaluator(
    DELIVERY_ROOT / "checkpoints" / "rotation_motion_evaluator.pt",
    omg_root=OMG_ROOT,
    omg_checkpoint=OMG_ROOT / "models" / "evaluator" / "step_004000.pt",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

condition_embedding, condition_metadata = evaluator.encode_condition(
    Path("sample_retargeted_motion.pkl")
)
motion_embedding, motion_metadata = evaluator.encode_motion_directory(
    Path("generated_motion_directory")
)

# 或直接编码生成模型输出的 qpos_36
motion_embedding, motion_metadata = evaluator.encode_qpos(qpos_36, source_fps=30.0)
```

## 接入评测

对同一组 pair 按相同顺序分别编码 condition 和 generated motion。每个 retrieval group 必须恰好 32 对，且同一组内 `kinematic_id` 最多出现一次；如果无法构造无冲突分组，应直接报错，不能用重复 pair 补齐。

```python
condition_embeddings = np.stack(condition_embeddings)  # (32, 512)
motion_embeddings = np.stack(motion_embeddings)        # (32, 512)

# Motion -> Condition，严格 diagonal matching
r1, r2, r3 = evaluator.retrieval_at_k(
    motion_embeddings,
    condition_embeddings,
    top_k=3,
)
mm_distance = evaluator.mm_distance(condition_embeddings, motion_embeddings)
```

多组 validation 分别计算后，按 pair 数量汇总 R@1/R@2/R@3 和 MM Distance。预处理 metadata 应随结果保存，以记录输入 FPS、派生帧数及 window 划分。

推理 checkpoint SHA256：

```text
40894e2487084b49b3de606ad9fa594c1881127016d2afe2cf3ae9e2206e6b4d
```
