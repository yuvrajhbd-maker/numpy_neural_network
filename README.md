# Neural Network from Scratch using only NumPy

Handwritten-style digit (0-9) recognition. Poora neural network sirf **NumPy** se
bana hai - koi TensorFlow / PyTorch / sklearn nahi.

## Features
- Fully-connected multi-layer network (default: 100 -> 64 -> 32 -> 10)
- ReLU hidden activation, Softmax output, Cross-Entropy loss
- He weight initialization
- Mini-batch training with **Adam optimizer** (NumPy se likha)
- L2 regularization
- Gradient checking test (backprop sahi hai ye prove karta hai)
- Model save/load (`.npz`), confusion matrix, accuracy
- Dataset khud generate hota hai (internet ki zarurat nahi)

## Folder Structure
```
numpy_neural_network/
├── src/
│   ├── data.py            # dataset generator, split, one-hot
│   ├── neural_network.py  # NN class: forward, backward, Adam, save/load
│   ├── train.py           # training script
│   └── predict.py         # saved model se prediction
├── tests/test_gradients.py
├── models/                # trained model yahan save hota hai
├── requirements.txt
└── README.md
```

## Run Karne ka Tarika
```bash
pip install numpy
python -m tests.test_gradients   # tests
python -m src.train              # training + accuracy
python -m src.predict            # sample predictions
```

## Theory (Viva ke liye)
**Forward:** z = aW + b, hidden layers me a = ReLU(z), last layer me softmax.

**Loss:** Cross-Entropy = -mean(sum(y * log(p))).

**Backward:** Softmax + CE ka gradient simple hai: delta = (p - y)/N.
Fir chain rule se: dW = a_prev^T · delta, delta_prev = (delta · W^T) * ReLU'(z).

**Adam:** m = b1*m + (1-b1)g, v = b2*v + (1-b2)g^2,
update = lr * m_hat / (sqrt(v_hat) + eps).

**Regularization:** L2 penalty weights ko chhota rakhti hai -> overfitting kam.
