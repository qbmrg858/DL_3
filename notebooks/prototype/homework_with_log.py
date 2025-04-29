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

# activation functions
def relu(x):
    return np.maximum(0, x)

def deriv_relu(x):
    dx = np.zeros_like(x)
    dx[x > 0] = 1
    return dx

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# loss
def crossentropy_loss(t, y):
    return -np.sum(t * np.log(np.clip(y, 1e-10, 1.0))) / t.shape[0]

# Dense layer
def xavier_init(in_dim, out_dim):
    limit = np.sqrt(6.0 / (in_dim + out_dim))
    return np.random.uniform(-limit, limit, size=(in_dim, out_dim))

class Dense:
    def __init__(self, in_dim, out_dim):
        self.W = xavier_init(in_dim, out_dim)
        self.b = np.zeros((1, out_dim))
    def forward(self, x):
        self.x = x
        return x.dot(self.W) + self.b
    def backward(self, grad):
        batch = self.x.shape[0]
        self.dW = self.x.T.dot(grad) / batch
        self.db = np.sum(grad, axis=0, keepdims=True) / batch
        return grad.dot(self.W.T)
    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db

# Model
class Model:
    def __init__(self, layer_sizes, activation_names):
        # build layers and activations from specs
        self.layers = []
        self.activations = []
        act_map = {'relu': relu, 'softmax': softmax}
        for i in range(len(layer_sizes)-1):
            self.layers.append(Dense(layer_sizes[i], layer_sizes[i+1]))
            self.activations.append(act_map[activation_names[i]])
    def __call__(self, x):
        h = x
        self.zs, self.hs = [], [h]
        for lyr, act in zip(self.layers, self.activations):
            z = lyr.forward(h); self.zs.append(z)
            h = act(z); self.hs.append(h)
        return h
    def backward(self, t, y):
        grad = (y - t) / t.shape[0]
        for i in reversed(range(len(self.layers))):
            if self.activations[i] == relu:
                grad = grad * deriv_relu(self.zs[i])
            grad = self.layers[i].backward(grad)
    def update(self, lr):
        for lyr in self.layers:
            lyr.update(lr)

def load_data():
    x_train = np.load('../data/x_train.npy')
    t_train = np.load('../data/y_train.npy')
    x_test  = np.load('../data/x_test.npy')
    x_train, x_test = x_train/255., x_test/255.
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test  = x_test.reshape(x_test.shape[0], -1)
    t_train = np.eye(10)[t_train.astype('int32').flatten()]
    return x_train, x_test, t_train


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=None,
                    help="path to JSON config (overrides defaults)")
args = parser.parse_args()

# default hyperparameters
default_params = {
    "lr": 1.0,
    "n_epochs": 100,
    "batch_size": 128,
    "seed": 42,
    "layers": [784, 256, 128, 10],
    "activations": ["relu", "relu", "softmax"]
}

# load params from file or use defaults
if args.config:
    with open(args.config) as f:
        params = json.load(f)
else:
    params = default_params

# set random seed
np.random.seed(params["seed"])

x_train, x_test, t_train = load_data()

# split
x_train, x_val, t_train, t_val = train_test_split(
    x_train, t_train, test_size=10000, random_state=params["seed"]
)

# build model
mlp = Model(params["layers"], params["activations"])

# training loop
for epoch in range(params["n_epochs"]):
    x_train, t_train = shuffle(x_train, t_train, random_state=params["seed"]+epoch)
    losses, accs = [], []
    n_batches = int(np.ceil(x_train.shape[0]/params["batch_size"]))
    for i in range(n_batches):
        xb = x_train[i*params["batch_size"]:(i+1)*params["batch_size"]]
        tb = t_train[i*params["batch_size"]:(i+1)*params["batch_size"]]
        yb = mlp(xb)
        losses.append(crossentropy_loss(tb, yb))
        accs.append(accuracy_score(tb.argmax(1), yb.argmax(1)))
        mlp.backward(tb, yb)
        mlp.update(params["lr"])
    yv = mlp(x_val)
    val_loss = crossentropy_loss(t_val, yv)
    val_acc  = accuracy_score(t_val.argmax(1), yv.argmax(1))
    print(f"Epoch {epoch+1}/{params['n_epochs']} | "
          f"train_loss: {np.mean(losses):.4f} train_acc: {np.mean(accs):.4f} | "
          f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}")

# predict test
t_pred = []
n_batches = int(np.ceil(x_test.shape[0]/params["batch_size"]))
for i in range(n_batches):
    xb = x_test[i*params["batch_size"]:(i+1)*params["batch_size"]]
    t_pred.extend(mlp(xb).argmax(1).tolist())

# file naming
timestamp = datetime.now().strftime("%m%d%H%M")
acc_str = f"{val_acc:.4f}".replace('.', '')
base = f"submission_{timestamp}_acc{acc_str}"
odir = './sub/'

# save CSV
pd.Series(t_pred, name='label')\
  .to_csv(os.path.join(odir, f"{base}.csv"), header=True, index_label='id')

# save params JSON
with open(os.path.join(odir, f"{base}.json"), 'w') as fp:
    json.dump(params, fp, indent=4)
