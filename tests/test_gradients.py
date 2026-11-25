"""Gradient check + sanity tests:  python -m tests.test_gradients"""
import numpy as np
from src.neural_network import NeuralNetwork, cross_entropy
from src.data import generate_dataset, one_hot


def test_gradient_check():
    X, y = generate_dataset(20, seed=3)
    Y = one_hot(y)
    net = NeuralNetwork([100, 8, 10], l2=0.0)
    net.forward(X)
    dW, _ = net.backward(Y)
    eps, W = 1e-5, net.W[0]
    for (i, j) in [(0, 0), (5, 3), (50, 7)]:
        old = W[i, j]
        W[i, j] = old + eps; lp = cross_entropy(net.forward(X), Y)
        W[i, j] = old - eps; lm = cross_entropy(net.forward(X), Y)
        W[i, j] = old
        num = (lp - lm) / (2 * eps)
        assert abs(num - dW[0][i, j]) < 1e-6, (num, dW[0][i, j])
    print("gradient check PASSED")


def test_softmax_sums_to_one():
    net = NeuralNetwork([100, 8, 10])
    X, _ = generate_dataset(5)
    assert np.allclose(net.forward(X).sum(axis=1), 1)
    print("softmax test PASSED")


if __name__ == "__main__":
    test_gradient_check()
    test_softmax_sums_to_one()
