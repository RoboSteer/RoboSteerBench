# Text-Motion Evaluator

## 文件结构

以下路径均相对于本目录：

```text
Text_Motion_Evaluator_Delivery/
├── README.md
├── requirements.txt
├── evaluation_results.json
├── checkpoints/
│   └── text_motion_evaluator.pt
└── text_motion_evaluator/
    ├── __init__.py
    ├── inference.py
    ├── model.py
    ├── omg_encoder.py
    └── runtime.py
```

本目录不包含 OMG 源码、OMG motion evaluator checkpoint 和 Hugging Face 模型缓存。使用时需要另外提供完整 OMG 仓库，并确认存在：

```text
OMG/models/evaluator/step_004000.pt
```

该 OMG checkpoint 的 SHA256 必须为：

```text
e346790c5ce33c42840aa19f7fcc2891a4cab27e0d92e23bfaff7d48cb339c40
```

文本 backbone 为固定 revision 的 `google-t5/t5-base`。如果本机能访问 Hugging Face，第一次运行会自动下载到默认缓存；也可以通过构造函数的 `model_cache` 指定缓存目录。

## 训练方式

- 使用 `motion_example_0723_00000` 至 `00009`，共 10,240 对 Text/Motion。
- 每条数据按 `subject_id` 选择 caption segment，按时间排序后用 ` Then, ` 拼接所有非空 `action`，与完整 Motion 配对；原始 caption 和 Motion 未修改。
- 按规范化完整 caption 的 `semantic_id` 确定性切分，seed 为 `20250822`：9,325 对训练、915 对 validation。
- Text 侧使用冻结的 `google-t5/t5-base`，对最后隐状态做 attention-mask mean pooling，得到 768-D feature；训练 `Linear(768,512) + LayerNorm + L2 normalization`。
- Motion 侧复用冻结的 OMG Motion Encoder；Motion 派生为 30 FPS、60-frame window、stride 30，窗口 embedding 平均后 L2 normalization，输出 512-D。
- Loss 为 symmetric masked InfoNCE。严格对应的 Text/Motion 为 positive；batch 内其他语义为 negative；相同 caption 的非对应 pair 从 negative denominator 中排除。
- 共训练 30 epochs，batch size 64，learning rate `1e-4`。best 规则为先最大化 R@1，R@1 相同时最小化 MM Distance，最终选择 epoch 27。

正式 validation 使用 fixed-batch32、Motion→Text、严格 diagonal matching。同一个 retrieval group 内同一 `semantic_id` 最多出现一次；最终为 27 组、864 对，确定性排除 51 对，semantic conflict 为 0：

```text
R@1 = 0.2708333
R@2 = 0.4166667
R@3 = 0.5266204
MM Distance = 1.2716582
```

完整顺序、排除 pair 和指标见 `evaluation_results.json`。

## Python 使用

先安装与目标 CUDA 匹配的 PyTorch，再安装 `requirements.txt`。将本目录和 OMG 的实际位置传入：

```python
import sys
from pathlib import Path

import numpy as np
import torch

DELIVERY_ROOT = Path("/path/to/Text_Motion_Evaluator_Delivery").resolve()
OMG_ROOT = Path("/path/to/OMG").resolve()
sys.path.insert(0, str(DELIVERY_ROOT))

from text_motion_evaluator import TextMotionEvaluator

evaluator = TextMotionEvaluator(
    DELIVERY_ROOT / "checkpoints" / "text_motion_evaluator.pt",
    omg_root=OMG_ROOT,
    omg_checkpoint=OMG_ROOT / "models" / "evaluator" / "step_004000.pt",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # model_cache=Path("/path/to/huggingface_cache"),  # 可选
)

text_embedding, text_metadata = evaluator.encode_text(
    "A person walks forward and then turns left."
)
motion_embedding, motion_metadata = evaluator.encode_motion_directory(
    Path("generated_motion_directory")
)

# 也可直接编码生成模型输出的 qpos (T,36)
# motion_embedding, motion_metadata = evaluator.encode_qpos(qpos_36, source_fps=30.0)
```

`encode_text` 和 Motion 编码接口均输出一个 `(512,)`、L2-normalized `float32` embedding。批量文本可调用 `encode_texts(texts)`。

## 接入评测

按相同顺序编码 32 对 caption 和 generated Motion。同一组内不能出现重复 `semantic_id`；无法构造无冲突组时应直接报错，不能用重复 pair 补齐。

```python
text_embeddings = np.stack(text_embeddings)      # (32, 512)
motion_embeddings = np.stack(motion_embeddings)  # (32, 512)

# Motion -> Text，严格 diagonal matching
r1, r2, r3 = evaluator.retrieval_at_k(
    motion_embeddings,
    text_embeddings,
    top_k=3,
)
mm_distance = evaluator.mm_distance(text_embeddings, motion_embeddings)
```

多组结果按 pair 数量汇总。建议同时保存 caption、pair 顺序、semantic_id、排除 pair，以及 Motion 编码返回的 FPS/window metadata。

推理 checkpoint SHA256：

```text
618f6eb88aa1a569cbd0ff515860073941e28c753f82e93f56ba60e00f02b91c
```
