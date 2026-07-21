# AND Gate with a Neural Network (TensorFlow / Keras)

A small feedforward neural network that learns the logical **AND** gate. Same problem as the from-scratch perceptron, but solved with Keras so you can see what the framework abstracts away.

## Requirements

```bash
pip install tensorflow numpy
```

## Architecture

```
Input (2 features)
      |
Dense(4), ReLU          <- hidden layer
      |
Dense(1), Sigmoid       <- output layer, gives probability
```

| Setting | Value | Why |
|---|---|---|
| Optimizer | `adam` | Adaptive learning rate, good default |
| Loss | `binary_crossentropy` | Standard for 0/1 outputs |
| Metric | `accuracy` | Easy to read during training |
| Epochs | `10` | See the note below |

## Dataset

| x1 | x2 | y |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

Only 4 samples, so this is a memorisation task, not a generalisation one. That is fine, the point is to watch the weights move.

## Usage

```bash
python and_gate_nn.py
```

The script prints predictions before training, trains for 10 epochs, then prints predictions again so you can compare.

## How to read the output

The sigmoid gives a probability between 0 and 1. A threshold of 0.5 converts it to a class:

```
pred >= 0.5  ->  1
pred <  0.5  ->  0
```

Before training the weights are random, so predictions hover near 0.5 with no real pattern. After training they should push toward 0 for the first three rows and toward 1 for the last.

## Two things to watch out for

**1. The pre-training class check has a bug**

```python
int(pred >= 0)   # always True, sigmoid output is never negative
```

It should be `int(pred >= 0.5)`, same as the post-training loop. As written, every untrained input reports class 1.

**2. Ten epochs is not enough**

With 4 samples and one gradient step per epoch, the model barely moves. Predictions will still be mush. Bump it up to see real convergence:

```python
model.fit(X, y, epochs=500, verbose=0)
```

At 500 epochs the outputs separate cleanly. At 10 they mostly do not.

## Perceptron vs this

| | Perceptron (NumPy) | This (Keras) |
|---|---|---|
| Layers | 1 | 2 |
| Activation | Step | ReLU + Sigmoid |
| Output | Hard 0/1 | Probability |
| Update rule | Written by hand | Backprop via Adam |
| Solves XOR | No | Yes, the hidden layer makes it possible |

That last row is the real reason this version exists. Swap `y` to `[[0],[1],[1],[0]]` and it still learns, where the single perceptron cannot.
