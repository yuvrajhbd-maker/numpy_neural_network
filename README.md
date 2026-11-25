# Neural Network from Scratch using only NumPy

Handwritten-style digit (0-9) recognition.

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

## How to run
```bash
pip install numpy
python -m tests.test_gradients   # tests
python -m src.train              # training + accuracy
python -m src.predict            # sample predictions
```

## Theory

**Forward Propagation:**  
\(z = aW + b\). In hidden layers, the ReLU activation function is applied: \(a = ReLU(z)\). In the final layer, Softmax is used to generate class probabilities.

**Loss Function:**  
Cross-Entropy Loss is calculated as:

\[
L = -\text{mean}\left(\sum y \cdot \log(p)\right)
\]

**Backward Propagation:**  
The gradient of Softmax combined with Cross-Entropy is simple:

\[
\delta = \frac{p-y}{N}
\]

Using the chain rule:

\[
dW = a_{prev}^{T} \cdot \delta
\]

\[
\delta_{prev} = (\delta \cdot W^{T}) \cdot ReLU'(z)
\]

**Adam Optimizer:**  
Adam updates the model's parameters using moving averages of gradients:

\[
m = \beta_1 m + (1-\beta_1)g
\]

\[
v = \beta_2 v + (1-\beta_2)g^2
\]

The parameter update is:

\[
update = lr \cdot \frac{\hat m}{\sqrt{\hat v}+\epsilon}
\]

**Regularization:**  
L2 Regularization keeps the weights small by adding a penalty to large weights. This helps reduce overfitting and improves the model's generalization ability.
