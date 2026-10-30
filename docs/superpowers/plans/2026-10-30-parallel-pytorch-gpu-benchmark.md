# Parallel PyTorch GPU Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a validation-driven PyTorch GPU benchmark beside the existing NumPy CIFAR-10 benchmark without changing the from-scratch implementation.

**Architecture:** Keep `src/` as the NumPy reference path. Add notebook-local PyTorch model and training helpers that consume the already prepared NumPy split, convert batches to tensors, and explicitly place the model and data on CUDA. Report the two implementations separately and use validation loss, never test accuracy, for model selection.

**Tech Stack:** Python, PyTorch, torchvision, NumPy, Jupyter Notebook, pytest.

## Global Constraints

- The NumPy implementation in `src/` remains unchanged.
- The PyTorch path uses the existing 20,000/5,000/10,000 split and train-only normalization.
- The architecture is `3072 -> 512 -> 256 -> 128 -> 10`.
- The PyTorch output is logits; `CrossEntropyLoss` performs the class normalization.
- Use SGD with learning rate `5e-4`, batch size `64`, 200 epochs, and weight decay `1e-4`.
- Set and report Python, NumPy, PyTorch, and CUDA seeds/versions.
- A missing CUDA device raises a clear error in the GPU benchmark cell.
- Test data is evaluated only once after validation-based selection.
- `requirements-colab.txt` already declares `torch>=2.2` and `torchvision>=0.17`; do not add duplicate dependencies.

---

### Task 1: Add explicit PyTorch device and reproducibility setup

**Files:**
- Modify: `notebooks/colab/apprentissage_automatique_from_scratch.ipynb` after the existing NumPy benchmark configuration

**Interfaces:**
- Consumes: `SEED`, `INPUT_DIM`, `HIDDEN_UNITS`, `NUM_CLASSES`, `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, and `L2_STRENGTH`.
- Produces: `torch`, `device`, and a verified GPU runtime for the later PyTorch cells.

- [ ] **Step 1: Add imports and deterministic seed setup**

Add a notebook cell:

```python
import random
import time
import torch
from torch import nn

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
```

- [ ] **Step 2: Require CUDA explicitly**

Add:

```python
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU unavailable. Select Runtime > Change runtime type > T4 GPU "
        "before running the PyTorch benchmark."
    )

device = torch.device("cuda")
print({
    "device": str(device),
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "gpu_name": torch.cuda.get_device_name(device),
})
```

This prevents a CPU execution from being misreported as a GPU benchmark.

- [ ] **Step 3: Validate notebook syntax**

Run:

```bash
python3 - <<'PY'
import ast, json
from pathlib import Path
path = Path("notebooks/colab/apprentissage_automatique_from_scratch.ipynb")
nb = json.loads(path.read_text())
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        ast.parse("".join(cell.get("source", [])), filename=f"{path}:cell-{i}")
print("all code cells parse")
PY
```

Expected: `all code cells parse`.

- [ ] **Step 4: Commit the runtime setup**

```bash
git add notebooks/colab/apprentissage_automatique_from_scratch.ipynb
git commit -m "feat: add explicit PyTorch CUDA setup"
```

### Task 2: Add the equivalent PyTorch model and data loaders

**Files:**
- Modify: `notebooks/colab/apprentissage_automatique_from_scratch.ipynb` after device setup

**Interfaces:**
- Consumes: normalized NumPy arrays and benchmark constants.
- Produces: `torch_model`, `train_loader`, `validation_loader`, and `test_loader`.

- [ ] **Step 1: Convert arrays to tensors without changing split semantics**

Add:

```python
X_train_tensor = torch.from_numpy(X_train).float()
X_validation_tensor = torch.from_numpy(X_validation).float()
X_test_tensor = torch.from_numpy(X_test).float()
y_train_tensor = torch.from_numpy(y_train_idx).long()
y_validation_tensor = torch.from_numpy(y_validation_idx).long()
y_test_tensor = torch.from_numpy(y_test_idx).long()

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor),
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=torch.Generator().manual_seed(SEED),
)
validation_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_validation_tensor, y_validation_tensor),
    batch_size=BATCH_SIZE,
    shuffle=False,
)
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor),
    batch_size=BATCH_SIZE,
    shuffle=False,
)
```

- [ ] **Step 2: Define and move the equivalent model**

Add:

```python
torch_model = nn.Sequential(
    nn.Linear(INPUT_DIM, HIDDEN_UNITS[0]),
    nn.ReLU(),
    nn.Linear(HIDDEN_UNITS[0], HIDDEN_UNITS[1]),
    nn.ReLU(),
    nn.Linear(HIDDEN_UNITS[1], HIDDEN_UNITS[2]),
    nn.ReLU(),
    nn.Linear(HIDDEN_UNITS[2], NUM_CLASSES),
).to(device)

assert next(torch_model.parameters()).device.type == "cuda"
with torch.no_grad():
    sample_logits = torch_model(X_train_tensor[:8].to(device))
assert sample_logits.shape == (8, NUM_CLASSES)
```

Do not add `Softmax`; `CrossEntropyLoss` expects logits.

- [ ] **Step 3: Verify loader and device invariants**

Add:

```python
sample_batch, sample_targets = next(iter(train_loader))
sample_batch = sample_batch.to(device)
sample_targets = sample_targets.to(device)
assert sample_batch.device == sample_targets.device == device
assert sample_batch.shape[1] == INPUT_DIM
assert sample_targets.ndim == 1
```

- [ ] **Step 4: Commit the model/data deliverable**

```bash
git add notebooks/colab/apprentissage_automatique_from_scratch.ipynb
git commit -m "feat: add equivalent PyTorch CIFAR-10 model"
```

### Task 3: Implement validation-driven PyTorch training

**Files:**
- Modify: `notebooks/colab/apprentissage_automatique_from_scratch.ipynb` after model/data cells

**Interfaces:**
- Consumes: `torch_model`, loaders, `device`, and benchmark hyperparameters.
- Produces: `torch_history` with one dictionary per epoch and `torch_best_epoch`.

- [ ] **Step 1: Define the evaluation helper**

Add:

```python
@torch.no_grad()
def evaluate_torch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        logits = model(X_batch)
        total_loss += criterion(logits, y_batch).item() * len(y_batch)
        total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total_samples += len(y_batch)
    return total_loss / total_samples, total_correct / total_samples
```

- [ ] **Step 2: Add the training loop with validation each epoch**

Add:

```python
criterion = nn.CrossEntropyLoss()
torch_optimizer = torch.optim.SGD(
    torch_model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=L2_STRENGTH,
)
torch_history = []
start_time = time.perf_counter()

for epoch in range(EPOCHS):
    torch_model.train()
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        torch_optimizer.zero_grad(set_to_none=True)
        logits = torch_model(X_batch)
        batch_loss = criterion(logits, y_batch)
        batch_loss.backward()
        torch_optimizer.step()

    train_loss, train_accuracy = evaluate_torch(
        torch_model, train_loader, criterion, device
    )
    validation_loss, validation_accuracy = evaluate_torch(
        torch_model, validation_loader, criterion, device
    )
    torch_history.append({
        "epoch": epoch,
        "loss": train_loss,
        "accuracy": train_accuracy,
        "validation_loss": validation_loss,
        "validation_accuracy": validation_accuracy,
    })

torch_elapsed_seconds = time.perf_counter() - start_time
torch_best_epoch = min(
    range(len(torch_history)),
    key=lambda index: torch_history[index]["validation_loss"],
)
assert len(torch_history) == EPOCHS
assert 0 <= torch_best_epoch < EPOCHS
print({
    "device": str(device),
    "elapsed_seconds": round(torch_elapsed_seconds, 2),
    "best_epoch": torch_best_epoch,
    "best_validation": torch_history[torch_best_epoch]["validation_accuracy"],
})
```

Do not evaluate `test_loader` in this loop.

- [ ] **Step 3: Commit the training deliverable**

```bash
git add notebooks/colab/apprentissage_automatique_from_scratch.ipynb
git commit -m "feat: train PyTorch benchmark with validation"
```

### Task 4: Add held-out reporting, plots, and comparison

**Files:**
- Modify: `notebooks/colab/apprentissage_automatique_from_scratch.ipynb` after PyTorch training

**Interfaces:**
- Consumes: `torch_history`, `torch_best_epoch`, `torch_model`, `test_loader`, and existing NumPy test metrics.
- Produces: final held-out PyTorch score, comparison table, and plots.

- [ ] **Step 1: Evaluate the test set once after selection**

Add:

```python
torch_test_loss, torch_test_accuracy = evaluate_torch(
    torch_model, test_loader, criterion, device
)
print({
    "implementation": "PyTorch GPU",
    "best_validation_epoch": torch_best_epoch,
    "test_loss": torch_test_loss,
    "test_accuracy": torch_test_accuracy,
    "device": str(device),
})
```

- [ ] **Step 2: Plot PyTorch train/validation curves**

Build lists from `torch_history` and add train/validation loss and accuracy
curves beside the existing NumPy plots. Label every curve with its
implementation and keep the plots deterministic.

- [ ] **Step 3: Print a labeled comparison**

Use a small list of dictionaries:

```python
comparison = [
    {
        "implementation": "NumPy from scratch",
        "test_accuracy": float(test_acc),
        "device": "CPU",
    },
    {
        "implementation": "PyTorch",
        "test_accuracy": float(torch_test_accuracy),
        "device": str(device),
    },
]
comparison
```

Explain in notebook markdown that the implementations are compared by
protocol, but need not be bit-for-bit identical because their optimization
details and initialization differ.

- [ ] **Step 4: Run local syntax and focused tests**

Run:

```bash
python3 - <<'PY'
import ast, json
from pathlib import Path
path = Path("notebooks/colab/apprentissage_automatique_from_scratch.ipynb")
nb = json.loads(path.read_text())
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        ast.parse("".join(cell.get("source", [])), filename=f"{path}:cell-{i}")
print("all code cells parse")
PY
PYTHONPATH=. pytest -q tests/test_multiclass_sequential_model.py tests/test_training_history.py tests/test_l2_regularization.py tests/test_training_diagnostics.py
```

Expected: syntax validation succeeds and the focused existing tests pass.

- [ ] **Step 5: Commit the reporting deliverable**

```bash
git add notebooks/colab/apprentissage_automatique_from_scratch.ipynb
git commit -m "feat: compare NumPy and PyTorch benchmark results"
```

### Task 5: Execute and verify in Colab

**Files:**
- Verify: `notebooks/colab/apprentissage_automatique_from_scratch.ipynb`
- Verify: `requirements-colab.txt`

**Interfaces:**
- Consumes: a Colab runtime configured with T4 GPU and synchronized `main`.
- Produces: verified GPU execution and recorded benchmark results.

- [ ] **Step 1: Select and verify the Colab GPU**

In Colab choose **Runtime > Change runtime type > T4 GPU**, then run the
device cell. Confirm it prints `device: cuda` and a T4 GPU name.

- [ ] **Step 2: Install only missing Colab dependencies**

Run the existing Colab requirements cell. Confirm `torch>=2.2` and
`torchvision>=0.17` are already declared; do not install duplicate packages
inside the notebook.

- [ ] **Step 3: Synchronize code and restart imports**

Run the existing Drive/GitHub synchronization cell, verify the checkout is on
`main` and its `HEAD` matches `origin/main`, then restart the runtime or reload
imports before executing benchmark cells.

- [ ] **Step 4: Run the PyTorch benchmark end to end**

Confirm all device, split, shape, history, validation-selection, and held-out
test assertions pass. Record elapsed time, best validation epoch, final test
accuracy, and GPU name.

- [ ] **Step 5: Run the final repository checks**

Run:

```bash
git --no-pager diff --check
git --no-pager status --short
```

Expected: no whitespace errors and no unintended source or dependency changes.
