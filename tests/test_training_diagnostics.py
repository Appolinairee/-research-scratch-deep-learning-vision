import numpy as np

from src.activations import ReLU
from src.layers import Dense
from src.models import Sequential
from src.training.diagnostics import collect_diagnostics


def test_collect_diagnostics_reports_weights_gradients_and_relu_zeros():
    dense = Dense(input_dim=2, output_dim=2, seed=0)
    relu = ReLU()
    dense.W = np.ones((2, 2))
    dense.dW = np.full((2, 2), 2.0)
    relu.forward(np.array([[-1.0, 2.0], [0.0, 3.0]]))

    diagnostics = collect_diagnostics(Sequential([dense, relu]))

    assert diagnostics.weight_norm == 2.0
    assert diagnostics.gradient_norm == 4.0
    assert diagnostics.relu_zero_fraction == 0.5
