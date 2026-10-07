# `default.json` 参数说明

`default.json` 是评测的默认协议配置。它必须保持为标准 JSON，因为
`scripts/evaluate.py` 通过 `json.loads()` 直接读取；标准 JSON 不支持
`//` 或 `#` 行内注释。本文件提供逐项中文说明，不参与运行。

| 参数 | 当前默认值 | 含义 |
| --- | ---: | --- |
| `prediction_fps` | `50` | prediction CSV 的声明帧率。用于时间重采样、物理指标的帧率校验，以及跨模态 evaluator 的 motion 输入。 |
| `motion_groundtruth_fps` | `50` | motion GT CSV 的声明帧率。用于与 prediction 的时间对齐及物理指标校验。 |
| `target_fps` | `30` | 通用 MotionEncoder / 旧窗口协议的目标帧率。跨模态 evaluator 可使用其自身固定帧率。 |
| `window_frames` | `60` | 通用 motion 窗口长度；在 `30 FPS` 下等于 `2` 秒。 |
| `num_windows` | `4` | 通用窗口预处理时，每条 source motion 均匀抽取的窗口数量。当前 FID/Diversity 正式 phase 协议不使用该参数。 |
| `fid_phase_frames` | `60` | FID 与 Diversity 的正式协议：按 `metadata.duration` 将每条完整 motion phase-normalize 为的固定帧数。 |
| `retrieval_batch_size` | `32` | R@1/R@5/R@10 每个固定检索候选组的大小；因此随机基线分别是 `1/32`、`5/32`、`10/32`。 |
| `mm_pair_batch_size` | `8192` | MM-Distance / Recall@K 管线一次交给 evaluator 的 source pair 上限。实际有效值不超过 `mm_cache_chunk_size`，当前为 `min(8192, 1024)=1024`。 |
| `mm_cache_chunk_size` | `1024` | 跨模态 embedding 的可恢复缓存块大小，也是多 GPU 动态调度和断点写盘的工作单元。 |
| `preprocess_workers` | `32` | CSV / condition 预读取的 CPU 线程数。过大可能造成共享存储的 I/O 竞争。 |
| `mm_preprocess_workers_per_gpu` | `8` | 多 GPU MM 编码时，每个 GPU worker 内用于读取 prediction CSV 的线程数。 |
| `mm_video_decode_workers_per_gpu` | `6` | 多 GPU 视频 MM 编码时，每个进程的视频本地缓存/FFmpeg 并发数；4 GPU 时合计最多 24。 |
| `video_local_cache_root` | `.cache/robosteer/video` | 相对于运行目录的视频字节缓存；可通过 `--video-cache-root` 指定其他位置。 |
| `max_frames_per_batch` | `32768` | 通用 MotionEncoder 推理时，每个 GPU batch 累计允许的最大帧数，用于限制显存占用。 |
| `contact_target_fps` | `50` | ContactSliding 计算足底速度与接触状态时使用的帧率。 |
| `diversity_block_size` | `8192` | Diversity 全 pair GPU 流式计算的 embedding block 大小。只影响速度和显存，不改变精确均值。 |
| `instruction_groundtruth_fps` | `50` | 对齐类 instruction 特征（例如 BAS audio29）对应的声明帧率。 |
| `bas_beat_threshold` | `0.5` | BAS 中 audio29 最后一维被判为 music beat 的阈值。 |
| `bas_sigma_frames` | `3.0` | BAS 的 beat 对齐核宽度，单位为帧。数值越大，对轻微时间偏移越宽容。 |
| `bas_min_motion_beat_distance_seconds` | `0.25` | BAS 提取 motion beat 时，相邻 motion beat 的最小时间间隔，单位秒。 |
| `mpjpe_fps` | `50` | OMG Video/SONIC MPJPE 的固定物理评测帧率；GT 先移除同目录 `info.txt` 中的 `start_hold_frames`，随后要求与 prediction 原始 `50 FPS` 严格等帧；不重采样、不截断。 |
| `g_mpjpe_fps` | `50` | OMG Video/SONIC g-MPJPE 的固定物理评测帧率；同样先移除 GT `start_hold_frames`，再要求原始 `50 FPS` 严格等帧。 |
| `e_vel_fps` | `50` | OMG Video/SONIC E_vel 的固定物理评测帧率；GT 先移除 `info.txt` 中的 `start_hold_frames`，随后要求与 prediction 原始 `50 FPS` 严格等帧；不重采样、不截断。 |
| `retrieval_seed` | `0` | R@K 构造确定性固定候选组时的随机种子。 |

## 修改建议
- 需要与已完成实验横向比较时，不应修改时间协议、`fid_phase_frames`、`retrieval_batch_size` 或 BAS 参数。Diversity 的全 pair 定义不依赖随机采样参数。
- 仅为加速跨模态评测时，可调整 `mm_cache_chunk_size`、`mm_pair_batch_size` 和 evaluator registry 中的内部 batch 参数；应使用新的输出目录，避免与旧缓存混用。
- `audio_motion` 的 `audio_chunk_batch_size` 和 `motion_window_batch_size` 不在本文件，而在选用的模型 registry 的 `cross_modal_evaluators.audio_motion` 条目中。
