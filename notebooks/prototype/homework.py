#%%
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Activation functions and derivatives
def relu(x):
    return np.maximum(0, x)

def deriv_relu(x):
    dx = np.zeros_like(x)
    dx[x > 0] = 1
    return dx

def softmax(x):
    # x: (batch, classes)
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def deriv_softmax(x):
    # not used directly
    s = softmax(x)
    return s * (1 - s)

# Loss
def crossentropy_loss(t, y):
    # t,y shape: (batch, classes)
    return -np.sum(t * np.log(np.clip(y, 1e-10, 1.0))) / t.shape[0]

# Dense layer
def xavier_init(in_dim, out_dim):
    limit = np.sqrt(6.0 / (in_dim + out_dim))
    return np.random.uniform(-limit, limit, size=(in_dim, out_dim))

class Dense:
    def __init__(self, in_dim, out_dim):
        self.W = xavier_init(in_dim, out_dim)
        self.b = np.zeros((1, out_dim))
        # gradients
        self.dW = None
        self.db = None
        # cache input
        self.x = None

    def forward(self, x):
        self.x = x  # (batch, in_dim)
        return x.dot(self.W) + self.b  # (batch, out_dim)

    def backward(self, grad):
        # grad: dL/dz where z = xW+b
        self.dW = self.x.T.dot(grad) / self.x.shape[0]
        self.db = np.sum(grad, axis=0, keepdims=True) / self.x.shape[0]
        # propagate to input
        return grad.dot(self.W.T)

# Model
class Model:
    def __init__(self, layers, activations):
        self.layers = layers
        self.activations = activations  # list of funcs
        self.zs = []  # pre-activations
        self.hs = []  # post-activations

    def __call__(self, x):
        h = x
        self.hs = [h]
        self.zs = []
        for layer, act in zip(self.layers, self.activations):
            z = layer.forward(h)
            self.zs.append(z)
            h = act(z)
            self.hs.append(h)
        return h

    def backward(self, t, y):
        # initial gradient: dL/dy with cross-entropy + softmax
        grad = (y - t) / t.shape[0]
        # backprop through layers
        for i in reversed(range(len(self.layers))):
            # derivative of activation
            z = self.zs[i]
            if self.activations[i] == relu:
                grad = grad * deriv_relu(z)
            # softmax derivative is combined in loss grad
            # backprop through dense
            grad = self.layers[i].backward(grad)

    def update(self, lr):
        for layer in self.layers:
            layer.W -= lr * layer.dW
            layer.b -= lr * layer.db

# Hyperparameters
lr = 1.0
n_epochs = 100
batch_size = 128

# Data loading and preprocessing (unchanged)
# assume x_train, t_train, x_test are preloaded as in provided template
# here we reload for completeness
import os
# x_train = np.load('drive/MyDrive/Colab Notebooks/DLBasics2025_colab/Lecture03/data/x_train.npy')
# t_train = np.load('drive/MyDrive/Colab Notebooks/DLBasics2025_colab/Lecture03/data/y_train.npy')
# x_test = np.load('drive/MyDrive/Colab Notebooks/DLBasics2025_colab/Lecture03/data/x_test.npy')
x_train = np.load('../data/x_train.npy')
t_train = np.load('../data/y_train.npy')
x_test = np.load('../data/x_test.npy')
x_train, x_test = x_train/255., x_test/255.
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
t_train = np.eye(10)[t_train.astype('int32').flatten()]
# split
x_train, x_val, t_train, t_val = train_test_split(x_train, t_train, test_size=10000, random_state=42)

# Initialize model: one hidden layer of 128 units
mlp = Model(
    layers=[Dense(784, 128), Dense(128, 10)],
    activations=[relu, softmax]
)

# Training
for epoch in range(n_epochs):
    # shuffle
    x_train, t_train = shuffle(x_train, t_train)
    # mini-batches
    num_batches = int(np.ceil(x_train.shape[0] / batch_size))
    losses = []
    accs = []
    for i in range(num_batches):
        xb = x_train[i*batch_size:(i+1)*batch_size]
        tb = t_train[i*batch_size:(i+1)*batch_size]
        # forward
        yb = mlp(xb)
        # loss
        loss = crossentropy_loss(tb, yb)
        losses.append(loss)
        # accuracy
        accs.append(accuracy_score(tb.argmax(1), yb.argmax(1)))
        # backward
        mlp.backward(tb, yb)
        # update
        mlp.update(lr)
    # validation
    y_val = mlp(x_val)
    val_loss = crossentropy_loss(t_val, y_val)
    val_acc = accuracy_score(t_val.argmax(1), y_val.argmax(1))
    print(f"Epoch {epoch+1}/{n_epochs} - loss: {np.mean(losses):.4f}, acc: {np.mean(accs):.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

# Prediction on test set
t_pred = []
num_batches = int(np.ceil(x_test.shape[0] / batch_size))
for i in range(num_batches):
    xb = x_test[i*batch_size:(i+1)*batch_size]
    yb = mlp(xb)
    t_pred.extend(yb.argmax(1).tolist())

submission = pd.Series(t_pred, name='label')
# submission.to_csv('drive/MyDrive/Colab Notebooks/DLBasics2025_colab/Lecture03/submission_pred.csv', header=True, index_label='id')
submission.to_csv('sub/submission_pred.csv', header=True, index_label='id')
