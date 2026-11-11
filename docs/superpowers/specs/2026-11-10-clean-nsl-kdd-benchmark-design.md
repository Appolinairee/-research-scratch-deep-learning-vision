# Clean NSL-KDD Benchmark Design

## Boundary

The Colab notebook is an experiment report and orchestration layer. It owns
the benchmark choice, runtime backend, plots, and displayed configuration.
Reusable behavior belongs in `src/`.

## Source modules

- `src/data/nsl_kdd.py` owns file parsing, attack grouping, deterministic
  train/validation splitting, categorical encoding, normalization, and
  one-hot labels.
- `src/experiments/nsl_kdd.py` owns the article-aligned MLP configuration,
  model construction, and focal-loss construction.
- `src/optim/adam.py` owns the reusable Adam update rule, including
  bias-corrected moments and optional learning-rate decay.
- `src/evaluation/classification.py` owns backend-compatible batched
  predictions and class-balanced metrics.
- `src/losses/multiclass_focal_loss.py` owns the focal-loss forward and
  backward equations.

Each boundary has focused pytest coverage. The notebook must not duplicate
label maps, preprocessing state, model layer assembly, or metric formulas.

## Notebook flow

Each benchmark notebook runs the shared setup first, then follows this order:

1. import public source APIs;
2. download or locate dataset files;
3. load the prepared dataset;
4. visualize class distribution;
5. construct the configured model and loss;
6. train with validation data;
7. report test metrics and confusion matrix.

The test set is never used for model selection. Data and model artifacts remain
under benchmark-specific subdirectories of the shared Drive project.

## Article-aligned comparison

The baseline comparison must use Adam with `lr=1e-4`, `decay=0.004`,
`batch_size=64`, and `epochs=250`. Cross-entropy is evaluated first; focal
loss is then evaluated with the same model, split, seed, and reporting code.
Accuracy is retained for comparison with the paper, but macro-F1 and
per-class metrics are required to assess minority-class behavior.
