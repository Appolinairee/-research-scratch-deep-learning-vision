# Serious MLP Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the CIFAR-10 notebook from the lightweight 8k/2k/2k experiment to a reproducible 20k/5k/10k MLP benchmark with explicit pre-run verification and a clean final report.

**Architecture:** Keep all reusable training and evaluation behavior in `src/`. Modify only the notebook orchestration: data sizes, invariant checks, benchmark configuration display, and final validation/test reporting. The notebook will select the best epoch from validation loss and will not use test data for selection.

**Tech Stack:** Python, Jupyter Notebook JSON, NumPy, torchvision datasets, pytest, existing NumPy MLP modules.

## Global Constraints

- Dataset split is exactly 20,000 training, 5,000 validation, and 10,000 test samples.
- Normalization statistics are fitted on training data only.
- Architecture is `3072 -> 512 -> 256 -> 128 -> 10`.
- Optimizer is mini-batch SGD with learning rate `5e-4`, batch size `64`, and `200` epochs.
- L2 strength is `1e-4`, applied to weights and not biases.
- Explicit seeds must remain stable.
- Validation selects the best epoch; test is evaluated only for the final report.
- Generated data, models, and Python caches remain excluded from synchronization checks.

---

### Task 1: Increase the data split and add dataset invariants

**Files:**
- Modify: `notebooks/colab/apprentissage_automatique_from_scratch.ipynb` data-preparation cell currently containing `TRAIN_SAMPLES`

**Interfaces:**
- Consumes: `train_dataset`, `test_dataset`, `np`, and the existing `to_numpy_subset`.
- Produces: `X_train`, `X_validation`, `X_test`, one-hot labels, and validated split metadata consumed by the model/training cells.

- [ ] **Step 1: Change the benchmark sample constants**

Set:

```python
TRAIN_SAMPLES = 20_000
VALIDATION_SAMPLES = 5_000
TEST_SAMPLES = 10_000
```

Keep the existing deterministic `SEED = 42` and the existing disjoint-source split: training and validation come from `train_dataset`, while test comes from `test_dataset`.

- [ ] **Step 2: Add shape, class, and index checks after preprocessing**

Add assertions immediately after one-hot encoding:

```python
assert len(train_indices) == TRAIN_SAMPLES
assert len(validation_indices) == VALIDATION_SAMPLES
assert len(test_indices) == TEST_SAMPLES
assert set(train_indices).isdisjoint(validation_indices)
assert X_train.shape == (TRAIN_SAMPLES, 32 * 32 * 3)
assert X_validation.shape == (VALIDATION_SAMPLES, 32 * 32 * 3)
assert X_test.shape == (TEST_SAMPLES, 32 * 32 * 3)
assert y_train.shape == (TRAIN_SAMPLES, NUM_CLASSES)
assert y_validation.shape == (VALIDATION_SAMPLES, NUM_CLASSES)
assert y_test.shape == (TEST_SAMPLES, NUM_CLASSES)
assert np.array_equal(np.unique(y_train_idx), np.arange(NUM_CLASSES))
assert np.array_equal(np.unique(y_validation_idx), np.arange(NUM_CLASSES))
assert np.array_equal(np.unique(y_test_idx), np.arange(NUM_CLASSES))
```

Do not assert that train and test integer indices are disjoint because they belong to different datasets and therefore have independent index spaces.

- [ ] **Step 3: Verify the notebook cell syntax**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path
path = Path("notebooks/colab/apprentissage_automatique_from_scratch.ipynb")
json.loads(path.read_text())
print("valid notebook JSON")
PY
```

Expected: `valid notebook JSON`.

- [ ] **Step 4: Commit the split change**

```bash
git add notebooks/colab/apprentissage_automatique_from_scratch.ipynb
git commit -m "feat: use serious CIFAR-10 benchmark split"
```

### Task 2: Make the benchmark configuration and model invariants explicit

**Files:**
- Modify: `notebooks/colab/apprentissage_automatique_from_scratch.ipynb` model-configuration cell currently containing `HIDDEN_UNITS`

**Interfaces:**
- Consumes: split constants and imported `Dense`, `ReLU`, `Softmax`, `Sequential`.
- Produces: the same `model`, `loss`, `optimizer`, and `regularizer` objects with printed/checked benchmark metadata.

- [ ] **Step 1: Add explicit input and output constants**

Add:

```python
INPUT_DIM = 32 * 32 * 3
NUM_CLASSES = 10
```

Use `INPUT_DIM` for the first `Dense` layer and retain:

```python
HIDDEN_UNITS = (512, 256, 128)
LEARNING_RATE = 5e-4
L2_STRENGTH = 1e-4
EPOCHS = 200
BATCH_SIZE = 64
```

- [ ] **Step 2: Add model shape and softmax checks**

After constructing the model, run a small real-batch check:

```python
assert model.layers[0].input_dim == INPUT_DIM
assert model.layers[0].output_dim == HIDDEN_UNITS[0]
assert model.layers[2].input_dim == HIDDEN_UNITS[0]
assert model.layers[2].output_dim == HIDDEN_UNITS[1]
assert model.layers[4].input_dim == HIDDEN_UNITS[1]
assert model.layers[4].output_dim == HIDDEN_UNITS[2]
assert model.layers[6].input_dim == HIDDEN_UNITS[2]
assert model.layers[6].output_dim == NUM_CLASSES

sample_output = model.forward(X_train[:8])
assert sample_output.shape == (8, NUM_CLASSES)
np.testing.assert_allclose(sample_output.sum(axis=1), 1.0, atol=1e-6)
```

Use the repository's actual public layer attributes; if a layer exposes a different public name, inspect that class and use the existing public interface rather than a private attribute.

- [ ] **Step 3: Print a reproducibility summary**

Add one compact display containing sample counts, architecture, optimizer, regularization, epochs, batch size, and seed. This output must appear before the long training cell so a run can be audited from the notebook output.

- [ ] **Step 4: Run focused repository tests**

Run:

```bash
pytest -q tests/test_multiclass_sequential_model.py tests/test_training_history.py tests/test_l2_regularization.py tests/test_training_diagnostics.py
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the model verification change**

```bash
git add notebooks/colab/apprentissage_automatique_from_scratch.ipynb
git commit -m "test: verify MLP benchmark configuration"
```

### Task 3: Add validation-driven reporting and final-run checks

**Files:**
- Modify: `notebooks/colab/apprentissage_automatique_from_scratch.ipynb` training and evaluation cells
- Test: notebook JSON and code-cell syntax validation

**Interfaces:**
- Consumes: `history`, `model`, `X_validation`, `y_validation`, `X_test`, `y_test`, and `evaluate_multiclass`.
- Produces: best validation epoch, final test metrics, and an auditable benchmark report.

- [ ] **Step 1: Keep training validation-aware**

Retain the existing call shape:

```python
history = train_multiclass_classifier(
    model=model,
    X=X_train,
    y=y_train,
    loss=loss,
    optimizer=optimizer,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    validation_data=(X_validation, y_validation),
    regularizer=regularizer,
)
```

- [ ] **Step 2: Assert complete history and diagnostics**

After training, verify:

```python
assert len(history) == EPOCHS
assert all(row.validation_loss is not None for row in history)
assert all(row.validation_accuracy is not None for row in history)
assert all(row.weight_norm is not None for row in history)
assert all(row.gradient_norm is not None for row in history)
assert all(row.relu_zero_fraction is not None for row in history)
best_epoch = history.best_epoch
assert 0 <= best_epoch < EPOCHS
```

- [ ] **Step 3: Evaluate validation and test without using test for selection**

Report the selected validation row, then evaluate test exactly once after selection:

```python
best_metrics = history[best_epoch]
test_metrics = evaluate_multiclass(model, X_test, y_test, loss)
print({
    "best_epoch": best_epoch,
    "best_validation_loss": best_metrics.validation_loss,
    "best_validation_accuracy": best_metrics.validation_accuracy,
    "test_loss": test_metrics.loss,
    "test_accuracy": test_metrics.accuracy,
})
```

The report must label the test score as a final held-out measurement. Do not compare epochs by test accuracy.

- [ ] **Step 4: Validate every notebook code cell without executing Colab-only data loading**

Run:

```bash
python3 - <<'PY'
import ast, json
from pathlib import Path

path = Path("notebooks/colab/apprentissage_automatique_from_scratch.ipynb")
nb = json.loads(path.read_text())
for index, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        ast.parse("".join(cell.get("source", [])), filename=f"{path}:cell-{index}")
print("all code cells parse")
PY
```

Expected: `all code cells parse`.

- [ ] **Step 5: Commit the reporting change**

```bash
git add notebooks/colab/apprentissage_automatique_from_scratch.ipynb
git commit -m "feat: report validation-selected CIFAR-10 benchmark"
```

### Task 4: Final verification and Colab execution

**Files:**
- Verify: `notebooks/colab/apprentissage_automatique_from_scratch.ipynb`
- Verify: `src/training/sequential.py`, `src/training/history.py`, `src/evaluation/classification.py`, `src/regularization/l2.py`

**Interfaces:**
- Consumes: the synchronized `main` checkout in Google Drive.
- Produces: reproducible benchmark output and a final comparison against the earlier 8k/2k/2k run.

- [ ] **Step 1: Synchronize Drive and verify Git identity**

Run the notebook synchronization cells and confirm:

```python
assert subprocess.run(
    ["git", "-C", str(PROJECT_ROOT), "branch", "--show-current"],
    check=True, capture_output=True, text=True,
).stdout.strip() == "main"
```

Confirm the Drive `HEAD` hash equals `origin/main` before importing `src`.

- [ ] **Step 2: Restart or reload imported modules**

Restart the Colab runtime, or reload the `src` modules after synchronization, so Python does not reuse modules imported before the pull.

- [ ] **Step 3: Run the notebook from the first data cell through final report**

Confirm the printed shapes, configuration, assertions, history length, validation diagnostics, and held-out test result.

- [ ] **Step 4: Compare against the previous baseline**

Record the old and new train/validation/test metrics with the same metric definitions. Interpret improvement only after checking that the split, normalization, seeds, architecture, and training budget are the intended ones.

- [ ] **Step 5: Verify the final worktree**

Run:

```bash
git --no-pager diff --check
git --no-pager status --short
```

Expected: no whitespace errors; only intentional notebook changes or generated files excluded by the synchronization policy.
