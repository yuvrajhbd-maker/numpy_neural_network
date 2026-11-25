"""Neural Network from scratch - sirf NumPy.

Features: multiple hidden layers, ReLU, Softmax, Cross-Entropy loss,
He initialization, mini-batch training, Adam optimizer, L2 regularization,
save/load model.
"""
import numpy as np


def relu(z):
    return np.maximum(0, z)


def relu_grad(z):
    return (z > 0).astype(z.dtype)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)  # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(probs, y_onehot, eps=1e-12):
    return -np.mean(np.sum(y_onehot * np.log(probs + eps), axis=1))


class NeuralNetwork:
    def __init__(self, layer_sizes, lr=0.001, l2=1e-4, seed=42):
        """layer_sizes example: [100, 64, 32, 10]"""
        self.layer_sizes = list(layer_sizes)
        self.lr = lr
        self.l2 = l2
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            self.W.append(rng.normal(0, np.sqrt(2.0 / fan_in),
                                     (fan_in, layer_sizes[i + 1])))
            self.b.append(np.zeros((1, layer_sizes[i + 1])))
        # Adam state
        self._m = [np.zeros_like(p) for p in self.W + self.b]
        self._v = [np.zeros_like(p) for p in self.W + self.b]
        self._t = 0

    # ---------- forward ----------
    def forward(self, X):
        self._z, self._a = [], [X]
        a = X
        for i in range(len(self.W)):
            z = a @ self.W[i] + self.b[i]
            self._z.append(z)
            a = softmax(z) if i == len(self.W) - 1 else relu(z)
            self._a.append(a)
        return a

    # ---------- backward ----------
    def backward(self, y_onehot):
        n = y_onehot.shape[0]
        L = len(self.W)
        dW, db = [None] * L, [None] * L
        delta = (self._a[-1] - y_onehot) / n  # softmax + CE gradient
        for i in reversed(range(L)):
            dW[i] = self._a[i].T @ delta + self.l2 * self.W[i]
            db[i] = delta.sum(axis=0, keepdims=True)
            if i > 0:
                delta = (delta @ self.W[i].T) * relu_grad(self._z[i - 1])
        return dW, db

    # ---------- Adam update ----------
    def _step(self, dW, db, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        params = self.W + self.b
        grads = dW + db
        for k, (p, g) in enumerate(zip(params, grads)):
            self._m[k] = b1 * self._m[k] + (1 - b1) * g
            self._v[k] = b2 * self._v[k] + (1 - b2) * g * g
            m_hat = self._m[k] / (1 - b1 ** self._t)
            v_hat = self._v[k] / (1 - b2 ** self._t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + eps)  # in-place

    # ---------- training ----------
    def fit(self, X, y_onehot, X_val=None, y_val_onehot=None,
            epochs=40, batch_size=64, verbose=True, seed=0):
        rng = np.random.default_rng(seed)
        history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
        n = len(X)
        for ep in range(1, epochs + 1):
            idx = rng.permutation(n)
            for s in range(0, n, batch_size):
                b = idx[s:s + batch_size]
                self.forward(X[b])
                dW, db = self.backward(y_onehot[b])
                self._step(dW, db)
            p = self.forward(X)
            history["loss"].append(cross_entropy(p, y_onehot))
            history["acc"].append(accuracy(p.argmax(1), y_onehot.argmax(1)))
            msg = (f"Epoch {ep:3d}/{epochs} | loss {history['loss'][-1]:.4f} "
                   f"| acc {history['acc'][-1]*100:.2f}%")
            if X_val is not None:
                pv = self.forward(X_val)
                history["val_loss"].append(cross_entropy(pv, y_val_onehot))
                history["val_acc"].append(
                    accuracy(pv.argmax(1), y_val_onehot.argmax(1)))
                msg += (f" | val_loss {history['val_loss'][-1]:.4f} "
                        f"| val_acc {history['val_acc'][-1]*100:.2f}%")
            if verbose:
                print(msg)
        return history

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X):
        return self.forward(X).argmax(axis=1)

    # ---------- save / load ----------
    def save(self, path):
        data = {"layer_sizes": np.array(self.layer_sizes)}
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            data[f"W{i}"], data[f"b{i}"] = w, b
        np.savez(path, **data)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        net = cls(d["layer_sizes"].tolist())
        for i in range(len(net.W)):
            net.W[i], net.b[i] = d[f"W{i}"], d[f"b{i}"]
        return net


def accuracy(pred, true):
    return float(np.mean(pred == true))


def confusion_matrix(true, pred, n_classes=10):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (true, pred), 1)
    return cm
