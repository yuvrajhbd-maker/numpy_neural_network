"""Dataset generator (sirf NumPy).

Internet/dataset download ki zarurat nahi: 5x7 pixel font ke digits 0-9 se
noisy + shifted images banate hain (10x10 canvas par).
"""
import numpy as np

# 5 wide x 7 tall bitmap font
_FONT = {
    0: ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    1: ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    2: ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    3: ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    4: ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    5: ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    6: ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    7: ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    8: ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    9: ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
}
CANVAS = 10  # image size = 10x10
N_CLASSES = 10


def _template(d):
    return np.array([[int(c) for c in row] for row in _FONT[d]], dtype=np.float64)


def make_image(digit, rng, noise=0.15, flip_prob=0.04):
    """Ek noisy digit image (CANVAS x CANVAS) banata hai."""
    img = np.zeros((CANVAS, CANVAS))
    t = _template(digit)
    r = rng.integers(0, CANVAS - 7 + 1)
    c = rng.integers(0, CANVAS - 5 + 1)
    img[r:r + 7, c:c + 5] = t
    flips = rng.random(img.shape) < flip_prob
    img = np.where(flips, 1 - img, img)
    img = img * rng.uniform(0.7, 1.0) + rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 1)


def generate_dataset(n_samples=5000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, N_CLASSES, n_samples)
    X = np.stack([make_image(d, rng).ravel() for d in y])
    return X, y


def train_test_split(X, y, test_ratio=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_ratio)
    te, tr = idx[:n_test], idx[n_test:]
    return X[tr], X[te], y[tr], y[te]


def one_hot(y, n_classes=N_CLASSES):
    out = np.zeros((len(y), n_classes))
    out[np.arange(len(y)), y] = 1
    return out
