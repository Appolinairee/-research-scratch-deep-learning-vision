# Parallel PyTorch GPU Benchmark

## Goal

Add a GPU-accelerated PyTorch benchmark beside the existing NumPy
from-scratch implementation, so the project preserves its pedagogical
reference while making the serious CIFAR-10 experiment practical in Colab.

## Scope and boundaries

The NumPy implementation in `src/` remains unchanged and continues to be the
reference implementation. The PyTorch path is notebook-specific benchmark
orchestration; it does not replace or rewrite the reusable NumPy modules.

The notebook will use the existing reproducible 20,000/5,000/10,000
train/validation/test split and the same preprocessing. The test set remains
held out until the final evaluation.

## PyTorch benchmark

The notebook will define an equivalent model:

```text
3072 -> 512 -> 256 -> 128 -> 10
```

with ReLU activations in hidden layers and a logits output. Cross-entropy will
consume logits directly, rather than adding a separate softmax layer during
training. The model, loss, optimizer, and batches will be moved to:

```python
torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

The benchmark will use mini-batch SGD, learning rate `5e-4`, batch size `64`,
200 epochs, explicit seeds, and weight decay equivalent to the existing L2
strength `1e-4`. The exact PyTorch and CUDA versions will be reported.

If CUDA is unavailable, the notebook must fail clearly for the GPU benchmark
cell rather than silently presenting a CPU run as a GPU result. The NumPy
benchmark remains independently runnable.

## Parity and verification

Before training, the notebook will verify:

- the selected device and CUDA availability;
- model parameters are on the selected CUDA device;
- input and target batches are on the same device;
- model output shape is `(batch_size, 10)`;
- data split sizes are 20,000/5,000/10,000;
- normalization uses training statistics only;
- train and validation indices are disjoint;
- explicit seeds are set for NumPy, Python, and PyTorch/CUDA;
- the architecture and hyperparameters match the reference configuration.

During and after training, it will verify:

- one training and validation metric is recorded per epoch;
- best epoch is selected by validation loss;
- test data is not used for selection;
- final test metrics are reported once after selection;
- elapsed training time and device are reported;
- the NumPy and PyTorch results are labeled separately because their
  optimization implementations may not be bit-for-bit identical.

## Dependency and notebook changes

Add the PyTorch dependency to the Colab requirements only if it is not already
provided by the Colab runtime. Keep the local project dependency footprint
unchanged unless a repository-level requirement is necessary.

Add dedicated notebook cells for device setup, the PyTorch model/training
function, metric curves, and final comparison. Do not mix PyTorch tensors into
the NumPy training modules.

## Acceptance criteria

The notebook can run the PyTorch path on a Colab GPU, proves that computation
is on CUDA, completes the configured benchmark, and reports validation-selected
and held-out test metrics. The existing NumPy tests continue to pass, and the
notebook syntax remains valid.
