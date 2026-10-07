# OMG MotionEncoder

`model.py` and `weights.pt` are the architecture and MotionEncoder-only weights
extracted from the OMG checkpoint `models/evaluator/step_004000.pt`.

The source evaluator checkpoint also stores optimizer and text-projection state;
they are intentionally excluded. Unlike the previous CPU-traced TorchScript
export, this state-dict version constructs the model on the requested device, so
it works with both CPU and CUDA evaluation.

Input: `body_pos_local`, `(B, T, 30, 3)` or flattened `(B, T, 90)`.
Output: L2-normalized `(B, 512)` embedding.
