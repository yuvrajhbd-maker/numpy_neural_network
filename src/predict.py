"""Saved model se prediction + ASCII image:  python -m src.predict"""
import numpy as np
from src.data import make_image
from src.neural_network import NeuralNetwork


def show(img):
    for row in img.reshape(10, 10):
        print("".join("#" if v > 0.5 else "." for v in row))


def main(n=5):
    net = NeuralNetwork.load("models/digit_model.npz")
    rng = np.random.default_rng(123)
    for _ in range(n):
        d = int(rng.integers(0, 10))
        img = make_image(d, rng).ravel()
        probs = net.predict_proba(img[None, :])[0]
        print(f"\nAsli digit: {d} | Predicted: {probs.argmax()} "
              f"(confidence {probs.max()*100:.1f}%)")
        show(img)


if __name__ == "__main__":
    main()
