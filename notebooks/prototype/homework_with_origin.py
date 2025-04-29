#%%
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def np_log(x):
    return np.log(np.clip(x, 1e-10, 1e+10))
def relu(x):
    return np.maximum(0, x)
def deriv_relu(x):
    return (x > 0).astype(x.dtype)
def softmax(x):
    x -= np.max(x, axis=1, keepdims=True)
    x_exp = np.exp(x)
    return x_exp / np.sum(x_exp, axis=1, keepdims=True)
def deriv_softmax(x):
    return softmax(x) * (1 - softmax(x))
def crossentropy_loss(t, y):
    return -np.sum(t * np_log(y)) / t.shape[0]

# Dense layer
class Dense:
    def __init__(self, in_dim, out_dim, function, deriv_function):
        self.W = np.random.uniform(low=-0.08, high=0.08, size=(in_dim, out_dim)).astype("float64")
        self.b = np.zeros(out_dim).astype("float64")
        self.function = function
        self.deriv_function = deriv_function

        self.x = None
        self.u = None

    def __call__(self, x):
        self.x = x
        self.u = np.matmul(self.x, self.W) + self.b
        return self.function(self.u)

    def b_prop(self, delta, W_next):
        self.delta = self.deriv_function(self.u) * np.matmul(delta, W_next.T)
        return self.delta

    def compute_grad(self):
        batch_size = self.delta.shape[0]
        self.dW = np.matmul(self.x.T, self.delta) / batch_size
        self.db = np.matmul(np.ones(batch_size), self.delta) / batch_size

    def get_grads(self):
        return self.dW, self.db

# Model class
class Model:
    def __init__(self, layer_sizes, activations, deriv_activations):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Dense(layer_sizes[i], layer_sizes[i+1], activations[i], deriv_activations[i]))

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, delta):
        for i, layer in enumerate(self.layers[::-1]):
            if i == 0:
                layer.delta = delta
            else:
                delta = layer.b_prop(delta, W)
            layer.compute_grad()
            W = layer.W

    def update(self, lr):
        for layer in self.layers:
            dW, db = layer.get_grads()
            layer.W -= lr * dW
            layer.b -= lr * db

# data loading
def load_data():
    x_train = np.load('../data/x_train.npy')
    t_train = np.load('../data/y_train.npy')
    x_test  = np.load('../data/x_test.npy')
    x_train, x_test = x_train/255., x_test/255.
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test  = x_test.reshape(x_test.shape[0], -1)
    t_train = np.eye(10)[t_train.astype('int32').flatten()]
    return x_train, x_test, t_train

def create_batch(data, batch_size):
    num_batches, mod = divmod(data.shape[0], batch_size)
    batched_data = np.split(data[: batch_size * num_batches], num_batches)
    if mod:
        batched_data.append(data[batch_size * num_batches:])
    return batched_data

def train_model(mlp, x_train, t_train, x_val, t_val, params):
    for epoch in range(params['n_epochs']):
        x_train, t_train = shuffle(x_train, t_train)
        x_batches = create_batch(x_train, params["batch_size"])
        t_batches = create_batch(t_train, params["batch_size"])

        losses_train, accs_train = [], []
        losses_valid, accs_valid = [], []

        for xb, tb in zip(x_batches, t_batches):
            yb = mlp(xb)
            loss = crossentropy_loss(tb, yb)
            losses_train.append(loss)
            acc = accuracy_score(tb.argmax(1), yb.argmax(1))
            accs_train.append(acc)

            delta = (yb - tb)
            mlp.backward(delta)
            mlp.update(params["lr"])

        yv = mlp(x_val)
        val_loss = crossentropy_loss(t_val, yv)
        val_acc = accuracy_score(t_val.argmax(1), yv.argmax(1))
        losses_valid.append(val_loss)
        accs_valid.append(val_acc)

        print(f"EPOCH: {epoch+1}/{params['n_epochs']} | Train Loss: {np.mean(losses_train):.4f} Acc: {np.mean(accs_train):.4f} | Valid Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
    return val_acc

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help="path to JSON config (overrides defaults)")
    args = parser.parse_args()

    default_params = {
        "lr": 0.1,
        "n_epochs": 10,
        "batch_size": 128,
        "seed": 42,
        "layers": [784, 100, 100, 10],
        "activations": [relu, relu, softmax],
        "deriv_activations": [deriv_relu, deriv_relu, deriv_softmax]
    }  

    if args.config:
        with open(args.config) as f:
            params = json.load(f)
    else:
        params = default_params
    return params

def save_sub_and_params(submission, params, val_acc):
    timestamp = datetime.now().strftime("%m%d%H%M")
    acc_str = f"{val_acc:.4f}".replace('.', '')
    base = f"{timestamp}_{acc_str}"
    odir = './sub/'
    submission.to_csv(f"{odir}{base}.csv", index=False)
    with open(f"{odir}{base}.json", 'w') as f:
        json.dump(params, f)

if __name__ == "__main__":
    params = get_params()
    np.random.seed(params["seed"])
    
    x_train, x_test, t_train = load_data()
    x_train, x_val, t_train, t_val = train_test_split(
        x_train, t_train, test_size=10000, random_state=params["seed"]
    )

    mlp = Model(params["layers"], params["activations"], params["deriv_activations"])
    val_acc = train_model(mlp, x_train, t_train, x_val, t_val, params)

    t_pred = []
    x_batches = create_batch(x_test, params["batch_size"])
    for xb in x_batches:
        preds = mlp(xb).argmax(1).tolist()
        t_pred.extend(preds)

    save_sub_and_params(pd.Series(t_pred, name='label'), params, val_acc)
