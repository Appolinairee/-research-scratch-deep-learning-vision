# Serious MLP CIFAR-10 Benchmark

## Goal

Replace the lightweight development split with an intermediate but serious
benchmark that can measure whether the current NumPy MLP improves when given
substantially more data, while keeping the experiment practical in Colab.
After this baseline, controlled hyperparameter experiments can estimate the
best performance reachable by this MLP family.

## Fixed benchmark configuration

- Dataset: CIFAR-10, flattened RGB images (`32 * 32 * 3 = 3072` features).
- Split: 20,000 training samples, 5,000 validation samples, 10,000 test samples.
- Preprocessing: fit normalization statistics on training data only; apply them
  unchanged to validation and test data.
- Architecture: `3072 -> 512 -> 256 -> 128 -> 10`.
- Hidden activation: ReLU.
- Output activation: softmax.
- Optimizer: plain mini-batch SGD.
- Learning rate: `5e-4`.
- Batch size: `64`.
- Epochs: `200`.
- Regularization: L2 with strength `1e-4`, applied to weights and not biases.
- Seeds: preserve explicit seeds for data sampling, layer initialization, and
  batch shuffling.

## Notebook responsibilities

The notebook will:

1. define the benchmark constants in one configuration cell;
2. create the 20k/5k/10k split reproducibly;
3. print and verify all resulting shapes;
4. train with validation metrics and diagnostics;
5. plot train/validation loss and accuracy;
6. select the best epoch from validation loss;
7. evaluate the untouched test set once for the final report;
8. report the configuration and verification checklist beside the results.

Reusable training, evaluation, history, diagnostics, and regularization logic
remain in `src/`; no dataset-specific logic is moved into those modules.

## Verification checklist

Before accepting a run, verify:

- CIFAR-10 source contains 50,000 training and 10,000 test images.
- Shapes are `(20000, 3072)`, `(5000, 3072)`, and `(10000, 3072)`.
- Label shapes match the corresponding sample counts and contain 10 classes.
- Train, validation, and test indices are disjoint.
- Normalization statistics are fitted on training data only.
- The test set is not used for model selection or hyperparameter tuning.
- Model dimensions are exactly `3072 -> 512 -> 256 -> 128 -> 10`.
- Softmax outputs have 10 columns and each row sums to approximately one.
- Loss and accuracy are recorded for every epoch.
- Validation metrics and weight/gradient/ReLU diagnostics are populated.
- L2 is included in training loss and weight gradients, not bias gradients.
- The best epoch is selected from validation loss, not test accuracy.
- The Drive checkout is on `main` and its commit matches `origin/main`.
- The notebook imports the synchronized source files, not stale modules.
- Results, seeds, and configuration are recorded so the run is reproducible.

## Follow-up optimization phase

Once this baseline is measured, run controlled variants one factor at a time:

- learning rate and schedule;
- L2 strength;
- hidden widths/depth;
- epoch count and early stopping;
- checkpoint restoration at the best validation epoch.

Each variant must use the same split and seeds, select on validation data, and
reserve the test set for the final comparison.
