# Clean NSL-KDD Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move NSL-KDD loading, preprocessing, model construction, and evaluation out of the notebook into focused, tested Python modules.

**Architecture:** Keep the notebook as a thin experiment report: configure the run, call typed helpers, plot results, and display metrics. Put dataset-specific mechanics in `src/data/nsl_kdd.py`, experiment construction in `src/experiments/nsl_kdd.py`, and reusable classification reporting in `src/evaluation/classification.py`.

**Tech Stack:** Python, NumPy, scikit-learn metrics, existing NumPy/CuPy backend, pytest, Jupyter notebooks.

## Global Constraints

- Preserve the NSL-KDD five-class mapping: `normal`, `dos`, `probe`, `r2l`, `u2r`.
- Fit categorical mappings and normalization statistics on the training partition only.
- Keep the article MLP architecture: `41 -> 512 -> 256 -> 128 -> 5`.
- Keep focal-loss defaults `gamma=2.0` and `alpha=0.25`.
- Keep the notebook responsible for experiment-specific configuration and presentation only.
- Do not use the test set for model or hyperparameter selection.
- Preserve the shared `setup_colab.ipynb` bootstrap.

## Article protocol gaps to close before the benchmark

- [x] Implement a tested Adam optimizer with bias-corrected first and second
  moments.
- [x] Support the article learning-rate decay parameter (`decay=0.004`).
- [ ] Use Adam rather than SGD for the NSL-KDD cross-entropy baseline.
- [ ] Keep the article learning rate (`1e-4`) and batch size (`64`).
- [ ] Add early stopping with patience `25` and restore the best validation
  weights.
- [ ] Disable the local-only scheduler and L2 regularization for the strict
  article comparison unless the article documents equivalent behavior.
- [ ] Run cross-entropy first, then focal loss, with the same split, seed,
  architecture, and evaluation protocol.
- [ ] Report accuracy together with per-class precision, recall, F1, macro-F1,
  and the confusion matrix; do not use accuracy alone to claim reproduction.

The current implementation uses SGD with momentum and a local scheduler, so
the existing result is not yet a strict reproduction of the article's Adam
protocol.

---

### Task 1: Add a tested NSL-KDD data module

**Files:**
- Create: `src/data/nsl_kdd.py`
- Modify: `src/data/__init__.py`
- Test: `tests/test_nsl_kdd_data.py`

**Interfaces:**
- `NSLKDD_CLASSES: tuple[str, ...]`
- `NSLKDDData` dataclass containing `X_train`, `X_validation`, `X_test`, `y_train`, `y_validation`, `y_test`, `y_train_idx`, `y_validation_idx`, `y_test_idx`, and `feature_names`.
- `load_nsl_kdd(train_path: Path, test_path: Path, validation_fraction: float = 0.2, seed: int = 42) -> NSLKDDData`.
- `read_nsl_kdd(path: Path) -> tuple[np.ndarray, np.ndarray]`.

- [ ] **Step 1: Write failing tests for label mapping and split invariants**

```python
def test_read_nsl_kdd_maps_attack_labels_to_five_classes(tmp_path):
    path = write_fixture(tmp_path / "train.csv", ["normal.", "neptune.", "ipsweep.", "guess_passwd.", "rootkit."])
    _, labels = read_nsl_kdd(path)
    np.testing.assert_array_equal(labels, [0, 1, 2, 3, 4])

def test_load_nsl_kdd_fits_preprocessing_on_train_and_returns_disjoint_splits(tmp_path):
    train_path, test_path = write_balanced_fixture(tmp_path)
    data = load_nsl_kdd(train_path, test_path, validation_fraction=0.25, seed=42)
    assert len(data.X_train) + len(data.X_validation) == 20
    assert set(data.y_train_idx.tolist()).issubset(range(5))
    assert data.X_train.shape[1] == data.X_validation.shape[1] == data.X_test.shape[1]
    assert data.y_train.shape[1] == data.y_validation.shape[1] == data.y_test.shape[1] == 5
```

- [ ] **Step 2: Run the focused tests and verify the expected missing-module failure**

Run: `PYTHONPATH=. pytest -q tests/test_nsl_kdd_data.py`

Expected: FAIL because `src.data.nsl_kdd` does not yet exist.

- [ ] **Step 3: Implement typed loading and preprocessing**

Implement `read_nsl_kdd` with the article’s attack grouping, parse the three categorical columns, fit category maps and normalization on `train_indices`, map unknown test categories to an explicit unknown bucket, and return one-hot labels plus integer labels.

- [ ] **Step 4: Run focused tests**

Run: `PYTHONPATH=. pytest -q tests/test_nsl_kdd_data.py`

Expected: PASS.

- [ ] **Step 5: Export the public data API**

Export `NSLKDDData`, `NSLKDD_CLASSES`, `load_nsl_kdd`, and `read_nsl_kdd` from `src/data/__init__.py`.

- [ ] **Step 6: Commit**

```bash
git add src/data tests/test_nsl_kdd_data.py
git commit -m "feat: extract NSL-KDD preprocessing"
```

### Task 2: Add a focused NSL-KDD experiment module

**Files:**
- Create: `src/experiments/__init__.py`
- Create: `src/experiments/nsl_kdd.py`
- Test: `tests/test_nsl_kdd_experiment.py`

**Interfaces:**
- `NSLKDDModelConfig` frozen dataclass with `hidden_units`, `dropout_rate`, `learning_rate`, `momentum`, `l2_strength`, `epochs`, `batch_size`, `gamma`, and `alpha`.
- `build_nsl_kdd_model(config: NSLKDDModelConfig, input_dim: int, backend: NumpyBackend | None = None) -> Sequential`.
- `build_nsl_kdd_loss(config: NSLKDDModelConfig, backend: NumpyBackend | None = None) -> MulticlassFocalLoss`.

- [ ] **Step 1: Write failing architecture tests**

```python
def test_build_nsl_kdd_model_uses_article_architecture():
    model = build_nsl_kdd_model(NSLKDDModelConfig(), input_dim=41)
    dense = [layer for layer in model.layers if isinstance(layer, Dense)]
    assert [dense[0].input_dim, *(layer.output_dim for layer in dense)] == [41, 512, 256, 128, 5]

def test_build_nsl_kdd_loss_uses_article_focal_defaults():
    loss = build_nsl_kdd_loss(NSLKDDModelConfig())
    assert loss.gamma == 2.0
    assert loss.alpha == 0.25
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `PYTHONPATH=. pytest -q tests/test_nsl_kdd_experiment.py`

Expected: FAIL because the experiment module does not yet exist.

- [ ] **Step 3: Implement the configuration and builders**

Use the existing `Dense`, `ReLU`, `Dropout`, `Softmax`, `Sequential`, `MulticlassFocalLoss`, and backend conventions. Do not duplicate model assembly in the notebook.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=. pytest -q tests/test_nsl_kdd_experiment.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/experiments tests/test_nsl_kdd_experiment.py
git commit -m "feat: extract NSL-KDD MLP construction"
```

### Task 3: Add reusable classification reporting

**Files:**
- Modify: `src/evaluation/classification.py`
- Test: `tests/test_classification_report.py`

**Interfaces:**
- `ClassificationReport` frozen dataclass containing `accuracy`, `macro_precision`, `macro_recall`, `macro_f1`, and `confusion_matrix`.
- `classification_report(model: Sequential, X: Array, y: Array, class_count: int, batch_size: int | None = None) -> ClassificationReport`.

- [ ] **Step 1: Write a failing report test**

```python
def test_classification_report_returns_macro_metrics_and_confusion_matrix():
    report = classification_report(model, X, y, class_count=5)
    assert 0.0 <= report.macro_f1 <= 1.0
    assert report.confusion_matrix.shape == (5, 5)
```

- [ ] **Step 2: Run the focused test and verify failure**

Run: `PYTHONPATH=. pytest -q tests/test_classification_report.py`

Expected: FAIL because `classification_report` does not yet exist.

- [ ] **Step 3: Implement batched prediction and metrics**

Use host-converted predictions for scikit-learn metrics, preserve backend support, and make the function independent of focal loss so it can compare focal loss and cross-entropy later.

- [ ] **Step 4: Run focused and existing evaluation tests**

Run: `PYTHONPATH=. pytest -q tests/test_classification_report.py tests/test_multiclass_accuracy.py tests/test_multiclass_sequential_model.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/evaluation tests/test_classification_report.py
git commit -m "feat: add classification reporting"
```

### Task 4: Thin the NSL-KDD notebook

**Files:**
- Modify: `notebooks/colab/nsl_kdd_focal_loss_mlp.ipynb`
- Test: notebook syntax and cell-order validation script

**Interfaces:**
- The notebook imports `load_nsl_kdd`, `NSLKDDModelConfig`, `build_nsl_kdd_model`, `build_nsl_kdd_loss`, and `classification_report`.

- [ ] **Step 1: Replace inline dataset logic with a loader call**

Keep only URL/download orchestration and call:

```python
data = load_nsl_kdd(
    train_path=DATA_ROOT / "KDDTrain%2B.txt",
    test_path=DATA_ROOT / "KDDTest%2B.txt",
    validation_fraction=0.2,
    seed=42,
)
```

- [ ] **Step 2: Keep one early visualization cell**

Plot class counts from `data.y_train_idx` immediately after loading. Do not reference variables created in later preprocessing cells.

- [ ] **Step 3: Replace inline model assembly with the experiment builder**

```python
config = NSLKDDModelConfig()
model = build_nsl_kdd_model(config, input_dim=data.X_train.shape[1], backend=backend)
loss = build_nsl_kdd_loss(config, backend=backend)
```

- [ ] **Step 4: Replace inline metric code with the report API**

Evaluate only after validation-based epoch selection and display accuracy, macro precision, macro recall, macro F1, and the confusion matrix.

- [ ] **Step 5: Validate every notebook code cell**

Run a JSON parse and `compile()` for every code cell. Confirm the notebook contains no references to `y_validation_idx`, `raw_train`, or `encode_features` outside the data module.

- [ ] **Step 6: Run the complete test suite**

Run: `PYTHONPATH=. pytest -q`

Expected: all new tests pass; document any pre-existing Dense gradient failures separately.

- [ ] **Step 7: Commit**

```bash
git add notebooks/colab/nsl_kdd_focal_loss_mlp.ipynb
git commit -m "refactor: make NSL-KDD notebook declarative"
```

### Task 5: Record the clean-code boundary

**Files:**
- Create: `docs/superpowers/specs/2026-11-10-clean-nsl-kdd-benchmark-design.md`

- [ ] **Step 1: Document responsibilities**

Record that notebooks orchestrate and visualize, while `src/` owns reusable data, model, training, and evaluation logic.

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-11-10-clean-nsl-kdd-benchmark-design.md
git commit -m "docs: define clean NSL-KDD benchmark boundaries"
```
