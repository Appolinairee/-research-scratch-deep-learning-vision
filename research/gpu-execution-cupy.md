# GPU execution for the from-scratch MLP

## Recommendation

Use an explicit array backend abstraction and add CuPy as the CUDA backend.
Keep the existing `Dense`, `ReLU`, `Sequential`, losses, optimizer, and manual
training loop. Do not replace the implementation with PyTorch: that would
benchmark a different framework rather than accelerate this project.

```python
xp = numpy_backend  # CPU
xp = cupy_backend   # NVIDIA GPU
```

The backend should own array creation, math operations, random initialization,
index generation, host/device conversion, and optional synchronization.

## Why CuPy

CuPy provides GPU-resident arrays and NumPy-like operations, including matrix
multiplication and elementwise functions. Its official documentation describes
`cupy.ndarray` as a GPU array with NumPy-like syntax:

- [CuPy basics](https://docs.cupy.dev/en/stable/user_guide/basic.html)
- [CuPy installation](https://docs.cupy.dev/en/stable/install.html)

CuPy is appropriate for the current Colab/NVIDIA target, but it is not a
perfect NumPy replacement. Its documented differences include indexing,
duplicate-index assignment, reductions, casting, and scalar behavior:

- [CuPy differences from NumPy](https://docs.cupy.dev/en/stable/user_guide/difference.html)

## Required code changes

Replacing `import numpy as np` with `import cupy as np` is not sufficient.
The current code directly uses `np.asarray`, `np.ndarray` checks, NumPy random
generators, and Python scalar conversions. Those operations can force CPU
execution or reject CuPy arrays.

The correct migration is:

1. Add a backend interface with operations such as `asarray`, `zeros`,
   `zeros_like`, `sqrt`, `exp`, `maximum`, `clip`, `sum`, `mean`, `max`,
   `argmax`, `arange`, random permutation, and `asnumpy`.
2. Pass the backend explicitly to the model/training path or store it on the
   model.
3. Initialize weights, biases, caches, gradients, batches, and regularization
   arrays on the same backend/device.
4. Keep forward, loss, backward, regularization, and parameter updates on the
   GPU for the whole mini-batch.
5. Convert only epoch-level metrics and final predictions back to NumPy/Python.

Important current coupling points are [Dense](/home/appolinaire/Projects/research-scratch-deep-learning-vision/src/layers/dense.py),
[Sequential](/home/appolinaire/Projects/research-scratch-deep-learning-vision/src/models/sequential.py),
[the training loop](/home/appolinaire/Projects/research-scratch-deep-learning-vision/src/training/sequential.py),
and [SGD](/home/appolinaire/Projects/research-scratch-deep-learning-vision/src/optim/sgd.py).

## Host/device boundaries

CuPy documents `cp.asarray(numpy_array)` for host-to-device transfer and
`cp.asnumpy(cupy_array)` for device-to-host transfer:

- [`cupy.asarray`](https://docs.cupy.dev/en/stable/reference/generated/cupy.asarray.html)
- [`cupy.asnumpy`](https://docs.cupy.dev/en/stable/reference/generated/cupy.asnumpy.html)

CUDA operations are asynchronous. Repeatedly converting losses, predictions,
or diagnostics to Python values inside the batch loop can synchronize the GPU
and remove the speedup. Transfer data once before training where possible and
synchronize only at measurement boundaries:

- [CuPy streams and current device](https://docs.cupy.dev/en/stable/user_guide/basic.html#current-stream)

## Testing strategy

Keep the existing NumPy tests as the CPU reference. Add backend tests for:

- forward values and shapes;
- backward gradients;
- parameter updates;
- CPU/GPU numerical parity on small deterministic inputs;
- one or more deterministic training updates;
- transfer boundaries and device placement;
- clear failure when GPU/CUDA is requested but unavailable.

Skip GPU-specific tests when no compatible GPU exists; never silently fall back
to CPU for a requested GPU benchmark. Time only after correctness is verified,
warm up kernels first, and synchronize around timing boundaries.

## What not to do

- Do not add PyTorch layers if the goal is to accelerate this implementation.
- Do not replace every NumPy import blindly.
- Do not call `cp.asnumpy` on every batch or layer.
- Do not mix NumPy labels/weights with CuPy activations.
- Do not leave parameters on CPU while copying activations to GPU.
- Do not assume CuPy and NumPy have identical indexing, scalar, casting, or
  random semantics.

NumPy's interoperability protocols are useful background, but they do not
remove the need to change this repository's direct `np.asarray`, concrete type
checks, and scalar conversions:

- [NumPy array interoperability](https://numpy.org/doc/stable/user/basics.interoperability.html)
- [NumPy dispatch protocols](https://numpy.org/doc/stable/user/basics.dispatch.html)
- [NEP 18](https://numpy.org/neps/nep-0018-array-function-protocol.html)
