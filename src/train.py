"""Training script:  python -m src.train"""
import numpy as np
from src.data import generate_dataset, train_test_split, one_hot
from src.neural_network import NeuralNetwork, accuracy, confusion_matrix


def main():
    print("Dataset ban raha hai...")
    X, y = generate_dataset(n_samples=15000, seed=1)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_ratio=0.2, seed=1)
    print(f"Train: {X_tr.shape}, Test: {X_te.shape}\n")

    net = NeuralNetwork([100, 64, 32, 10], lr=0.002, l2=1e-3)
    net.fit(X_tr, one_hot(y_tr), X_te, one_hot(y_te), epochs=30, batch_size=64)

    pred = net.predict(X_te)
    print(f"\nFINAL TEST ACCURACY: {accuracy(pred, y_te)*100:.2f}%")
    print("\nConfusion Matrix (rows=true, cols=predicted):")
    print(confusion_matrix(y_te, pred))

    net.save("models/digit_model.npz")
    print("\nModel saved: models/digit_model.npz")


if __name__ == "__main__":
    main()
