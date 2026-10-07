# RoboSteer Evaluation Code

This repository evaluates Level 1, Level 2, and Level 3 motion predictions on the RoboSteer benchmark. The benchmark dataset, predictions from the model being evaluated, and evaluator weights are provided separately.

## Repository Structure

- `scripts/level1`, `scripts/level2`, and `scripts/level3`: evaluation entry points and scoring logic for each level.
- `scripts/evaluation`: shared metrics, data matching, preprocessing, and model adapters.
- `evaluator_code`: evaluator model architectures, inference code, and a local model registry template. Weights are not included.
- `configs`: default evaluation protocols and parameters.

The evaluation entry points read text instructions and source trajectory conditions from the benchmark task JSON files. The rhythm evaluator automatically downloads the pinned AST backbone on first use if it is not available locally. The audio29 features required for rhythm tasks are also cached automatically during evaluation. To generate the cache in advance, run:

```bash
python -m scripts.evaluation.data.prepare_audio29 --root /path/to/rhythm/audio --workers 8
```

## Environment Setup

Run the following commands from the repository root. Linux, Python 3.10, and an NVIDIA GPU with sufficient memory are recommended. First, create a dedicated environment:

```bash
conda create -n robosteer-eval python=3.10 -y
conda activate robosteer-eval
```

Install the Python dependencies required for evaluation:

```bash
python -m pip install -r requirements.txt
```

## Data

Download the [evaluator weight bundle](https://huggingface.co/Leontc/RoboSteer-evaluator) from the repository root:

```bash
hf download Leontc/RoboSteer-evaluator --local-dir models
```

After the download completes, `models/` should be next to `evaluator_code/` in the repository root. The evaluator automatically reads `models/registry.json` and uses the OMG files included in the weight bundle to compute kinematic metrics. You do not need to specify a model configuration or configure OMG separately.

The RoboSteer [benchmark dataset](https://huggingface.co/datasets/YanCORANV/RoboSteer) is downloaded separately from the evaluator weights. Run the following command from the repository root:

```bash
bash scripts/download_dataset.sh
```

The script downloads and extracts the data according to the [official checksum manifest](https://huggingface.co/datasets/YanCORANV/RoboSteer/resolve/main/SHA256SUMS). By default, it creates `SteerableMotionBenchmarkDataset/` in the repository root, with `Data/` and `Tasks/` directly inside it. The complete dataset is approximately 352 GiB. To use another disk, set `DATASET_ROOT=/path/to/SteerableMotionBenchmarkDataset` before running the script.

Generate motions with the model you want to evaluate. See the model-specific guides in [RoboSteer Model Resources](https://huggingface.co/datasets/PhoebeCC/RoboSteer-Model-Resources) for inference and postprocessing instructions. Output locations vary by model. For example, [TextOp](https://huggingface.co/datasets/PhoebeCC/RoboSteer-Model-Resources/blob/main/models/textop/README_PIPELINES.md) writes converted CSV files to `outputs/<run_name>_csv/<sample_name>/`, while [GEM](https://huggingface.co/datasets/PhoebeCC/RoboSteer-Model-Resources/blob/main/models/gem/POSTPROCESS.md) writes them to `postprocess_outputs/sonic/<task_path>/`. The evaluation entry points read postprocessed G1 CSV files. Intermediate SMPL, BVH, or PKL files must first be converted using the corresponding model workflow.

We recommend organizing evaluation-ready predictions by model under `model_outputs/` in the repository root, next to `SteerableMotionBenchmarkDataset/` and `models/`. Each predicted motion must be stored in its own directory and contain at least `body_pos.csv`, `body_quat.csv`, and `joint_pos.csv`. The recommended layout is shown below. Level 1, Level 2, and Level 3 outputs for the same model are siblings, and each additional model has its own directory at the same level.

```text
RoboSteer/                                      # repository root
├── evaluator_code/
├── models/                                     # evaluator weights
├── scripts/
├── SteerableMotionBenchmarkDataset/
│   ├── Data/
│   └── Tasks/
└── model_outputs/
    ├── <model_name>/
    │   ├── level1/<task_name>/<source_id>/
    │   │   ├── body_pos.csv
    │   │   ├── body_quat.csv
    │   │   └── joint_pos.csv
    │   ├── level2/<task_name>/<source_id>_<task_type>/  # contains the same three CSV files
    │   └── level3/
    │       ├── text/<task_id>__order_<order>/            # contains the same three CSV files
    │       ├── audio/<task_id>__order_<order>/           # contains the same three CSV files
    │       └── video/<task_id>__order_<order>/           # contains the same three CSV files
    └── <another_model>/
```

For example, based on the public [Level 1](https://huggingface.co/datasets/YanCORANV/RoboSteer/blob/main/examples/tasks/L1_text_to_motion_generation_text_--04DHOI32k_00009_38_121.json), [Level 2](https://huggingface.co/datasets/YanCORANV/RoboSteer/blob/main/examples/tasks/L2_speed_text_--WFaLyeUko_00009_0_92_fast.json), and [Level 3](https://huggingface.co/datasets/YanCORANV/RoboSteer/blob/main/examples/tasks/L3_interleave_--6s5bu1NRU_00008_0_463.json) task JSON files in the dataset, the three prediction directories can be named as follows:

```text
model_outputs/GEM-GMR/level1/text_to_motion_generation/--04DHOI32k_00009_38_121/
model_outputs/GEM-GMR/level2/Speed/text/--WFaLyeUko_00009_0_92_fast/
model_outputs/GEM-GMR/level3/text/L3_interleave_--6s5bu1NRU_00008_0_463__order_3/
```

The placeholders in this example have the following meanings:

- `<model_name>` is the model name. This example uses `GEM-GMR`, a model evaluated by RoboSteer. `<task_name>` distinguishes tasks, such as `text_to_motion_generation` or `Speed/text`.
- `<source_id>` is the final directory name in `ground_truth.motion_parameters` in the task JSON, such as `--04DHOI32k_00009_38_121`.
- `<task_type>` comes from `metadata.task_type` in the Level 2 task JSON. It is `fast` in the Speed example above.
- `<task_id>` comes from `metadata.task_id` in the Level 3 task JSON. `<order>` is the index of the corresponding generation segment in `input.interleave.sequence`; it is `3` for the text segment above.

## Running Evaluation

The following commands use `GEM-GMR` as an example. Run all commands from the repository root and organize model outputs according to the directory structure shown above.

### Level 1: Text Generation

```bash
python scripts/level1/evaluate.py \
  --prediction model_outputs/GEM-GMR/level1/text_to_motion_generation \
  --motion-groundtruth SteerableMotionBenchmarkDataset/Data/Shared/Motion \
  --condition-groundtruth SteerableMotionBenchmarkDataset/Tasks/Level1/full_conditioning_reproduction/text_to_motion_generation \
  --task-root SteerableMotionBenchmarkDataset/Tasks/Level1/full_conditioning_reproduction/text_to_motion_generation \
  --dataset-root SteerableMotionBenchmarkDataset \
  --modality text \
  --output results/level1/GEM-GMR_text \
  --metrics FID,MM-Distance,BG,BS_1 \
  --mm-encoder text_motion \
  --mm-gpus 0
```

### Level 2: Speed with Text Conditioning

```bash
python scripts/level2/evaluate.py \
  --prediction model_outputs/GEM-GMR/level2/Speed/text \
  --level2-task-root SteerableMotionBenchmarkDataset/Tasks/Level2/Speed/text \
  --base-prediction model_outputs/GEM-GMR/level1/text_to_motion_generation \
  --base-motion-groundtruth SteerableMotionBenchmarkDataset/Data/Shared/Motion \
  --base-condition-groundtruth SteerableMotionBenchmarkDataset/Tasks/Level1/full_conditioning_reproduction/text_to_motion_generation \
  --dataset-root SteerableMotionBenchmarkDataset \
  --modality text \
  --output results/level2/GEM-GMR_Speed_text \
  --metrics FID,MM-Distance,BG,BS_2 \
  --mm-encoder text_motion \
  --mm-gpus 0
```

Level 2 Order and Times use PKL-formatted reference motions. Run `scripts/level2/evaluate_order.py` and `scripts/level2/evaluate_times.py`, respectively, and configure GMR and the VLM using `--gmr-root`, `--gmr-python`, and the corresponding `--vlm-*` arguments. To view all available arguments, run:

```bash
python scripts/level2/evaluate_order.py --help
python scripts/level2/evaluate_times.py --help
```

### Level 3: Multimodal Interleaved Generation

```bash
python scripts/level3/evaluate.py \
  --text-prediction model_outputs/GEM-GMR/level3/text \
  --audio-prediction model_outputs/GEM-GMR/level3/audio \
  --video-prediction model_outputs/GEM-GMR/level3/video \
  --task-root SteerableMotionBenchmarkDataset/Tasks/Level3 \
  --dataset-root SteerableMotionBenchmarkDataset \
  --output results/level3/GEM-GMR \
  --metrics FID,MM-Distance,BG,BS_3 \
  --mm-gpus 0
```

See the [task path reference](https://huggingface.co/datasets/YanCORANV/RoboSteer/blob/main/docs/task-map.md) for dataset paths used by other tasks. Run any entry point with `--help` to see its complete argument list. Evaluation results are written to the directory specified by `--output`. The default evaluation protocol is stored in `configs/evaluation/default.json`, and the video cache location can be changed with `--video-cache-root`.

## Release Scope

This repository contains source code and small configuration files only. The dataset, evaluator and VLM weights, prediction CSV or video files, caches, and evaluation results must be distributed separately and used in accordance with their applicable licenses.
