import numpy as np
import pytest
from types import SimpleNamespace

from src.core.backend import NumpyBackend, get_backend
from src.activations import ReLU, Softmax
from src.losses import MulticlassCrossEntropy
from src.layers import Dense


def test_numpy_backend_preserves_array_operations():
    backend = NumpyBackend()
    values = backend.asarray([[1.0, -2.0]])

    assert backend.is_array(values)
    np.testing.assert_allclose(backend.maximum(values, 0.0), [[1.0, 0.0]])
    np.testing.assert_allclose(backend.to_host(values), values)


def test_get_backend_rejects_unavailable_gpu_backend():
    cupy = pytest.importorskip("cupy")
    del cupy
    backend = get_backend("gpu")
    assert backend.is_gpu


def test_dense_uses_selected_backend_for_forward_and_backward():
    backend = NumpyBackend()
    layer = Dense(input_dim=2, output_dim=2, seed=0, backend=backend)
    X = backend.asarray([[1.0, 0.0], [0.0, 1.0]])

    output = layer.forward(X)
    layer.backward(backend.asarray([[1.0, 0.0], [0.0, 1.0]]))

    assert backend.is_array(output)
    assert backend.is_array(layer.dW)


def test_random_normal_uses_backend_generator_api():
    class RandomState:
        def __init__(self, seed):
            self.seed = seed

        def normal(self, loc, scale, size):
            return np.full(size, scale)

    backend = NumpyBackend(
        xp=SimpleNamespace(
            random=SimpleNamespace(
                RandomState=RandomState
            )
        )
    )

    values = backend.random_normal(seed=0, scale=0.1, shape=(2, 3))

    assert values.shape == (2, 3)
    np.testing.assert_allclose(values, 0.1)


def test_activation_and_loss_preserve_selected_backend():
    backend = NumpyBackend()
    relu = ReLU(backend=backend)
    softmax = Softmax(backend=backend)
    loss = MulticlassCrossEntropy(backend=backend)
    values = backend.asarray([[-1.0, 1.0]])

    hidden = relu.forward(values)
    probabilities = softmax.forward(hidden)
    loss_value = loss.forward(backend.asarray([[0.0, 1.0]]), probabilities)

    assert backend.is_array(hidden)
    assert backend.is_array(probabilities)
    assert isinstance(loss_value, float)
