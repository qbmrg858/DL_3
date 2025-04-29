#%%
import os
import sys
import json
import argparse
import math
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# --- activation functions and their derivatives ---
def relu(x):
    return np.maximum(0, x)

def deriv_relu(x):
    return (x > 0).astype(x.dtype)

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=1, keepdims=True)

def deriv_softmax(x):
    s = softmax(x)
    return s * (1 - s)

# map names to functions
ACTIVATION_MAP = {
    "relu": relu,
    "softmax": softmax,
}
DERIV_MAP = {
    "relu": deriv_relu,
    "softmax": deriv_softmax,
}

#--- decay ---
def linear_decay(initial_lr, epoch, total_epochs, **kw):
    return initial_lr * (1 - epoch / total_epochs)

def step_decay(initial_lr, epoch, total_epochs, step_size=10, gamma=0.1, **kw):
    return initial_lr * (gamma ** (epoch // step_size))

def exp_decay(initial_lr, epoch, total_epochs, gamma=0.9, **kw):
    return initial_lr * (gamma ** epoch)

def cosine_annealing(initial_lr, epoch, total_epochs, eta_min=0, **kw):
    return eta_min + (initial_lr - eta_min) * (1 + math.cos(math.pi * epoch / total_epochs)) / 2

SCHEDULERS = {
    "linear": linear_decay,
    "step":   step_decay,
    "exp":    exp_decay,
    "cosine": cosine_annealing,
}

#--- loss ---
def np_log(x):
    return np.log(np.clip(x, 1e-10, 1e+10))
def crossentropy_loss(t, y):
    return -np.sum(t * np_log(y)) / t.shape[0]

#--- Dense layer ---
class Dense:
    def __init__(self, in_dim, out_dim, activation, deriv_activation):
        self.W = np.random.uniform(-0.08, 0.08, (in_dim, out_dim)).astype('float64')
        self.b = np.zeros(out_dim, dtype='float64')
        self.activation = activation
        self.deriv_activation = deriv_activation
        self.x = None
        self.u = None

    def __call__(self, x):
        self.x = x
        self.u = x.dot(self.W) + self.b
        return self.activation(self.u)

    def b_prop(self, delta, W_next):
        self.delta = self.deriv_activation(self.u) * (delta.dot(W_next.T))
        return self.delta

    def compute_grad(self):
        m = self.delta.shape[0]
        self.dW = self.x.T.dot(self.delta) / m
        self.db = np.sum(self.delta, axis=0) / m

    def get_grads(self):
        return self.dW, self.db

#--- Model class ---
class MLP:
    def __init__(self, layer_sizes, activation_names):
        self.layers = []
        # construct layers with matching derivative
        for i in range(len(layer_sizes) - 1):
            act_name = activation_names[i]
            act = ACTIVATION_MAP[act_name]
            deriv = DERIV_MAP[act_name]
            self.layers.append(Dense(layer_sizes[i], layer_sizes[i+1], act, deriv))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, delta):
        for i, layer in enumerate(self.layers[::-1]):
            if i == 0:
                layer.delta = delta
            else:
                delta = layer.b_prop(delta, W_next)
            layer.compute_grad()
            W_next = layer.W

    def update(self, lr):
        for layer in self.layers:
            dW, db = layer.get_grads()
            layer.W -= lr * dW
            layer.b -= lr * db

    def load_model_weight(self, model_state):
        for i, layer in enumerate(self.layers):
            layer.W = model_state[f"W{i}"]
            layer.b = model_state[f"b{i}"]

class Callback:
    def on_epoch_end(self, epoch, metrics, trainer): pass

class ModelCheckpoint(Callback):
    def __init__(self, dirpath, monitor="val_acc", mode="max"):
        os.makedirs(dirpath, exist_ok=True)
        self.dir = dirpath; self.monitor = monitor
        self.best = -np.inf if mode=="max" else np.inf
        self.mode = mode
        self.mdoel_state = None
        self.epoch = None

    def on_epoch_end(self, epoch, metrics, trainer):
        score = metrics[self.monitor]
        improved = (score > self.best) if self.mode=="max" else (score < self.best)
        if improved:
            self.best = score
            self.mdoel_state = trainer.model_state()
            self.epoch = epoch

class Trainer:
    def __init__(self, model, optimizer, loss_fn, callbacks, params):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.callbacks = callbacks
        self.params = params.copy()

        self.metrics = {}

        self.initial_lr = self.params['lr']

    def model_state(self):
        state = {}
        for i, layer in enumerate(self.model.layers):
            state[f"W{i}"] = layer.W
            state[f"b{i}"] = layer.b
        return state
    
    def stop(self):
        self.stop_flag = False

    def fit(self, x_train, t_train, x_val, t_val):
        total_epochs = self.params['n_epochs']
        sched_name = self.params['lr_scheduler']
        sched_params = params.get('lr_scheduler_params', {})

        for epoch in range(self.params['n_epochs']):
            x_train, t_train = shuffle(x_train, t_train)
            x_batches = create_batch(x_train, self.params['batch_size'])
            t_batches = create_batch(t_train, self.params['batch_size'])

            losses, accs = [], []
            for xb, tb in zip(x_batches, t_batches):
                yb = self.model(xb)
                losses.append(crossentropy_loss(tb, yb))
                accs.append(accuracy_score(tb.argmax(1), yb.argmax(1)))
                delta = (yb - tb)
                self.model.backward(delta)
                self.model.update(self.params['lr'])
            yv = self.model(x_val)
            val_loss = crossentropy_loss(t_val, yv)
            val_acc  = accuracy_score(t_val.argmax(1), yv.argmax(1))
            print(f"Epoch {epoch+1}/{self.params['n_epochs']} - Train loss:{np.mean(losses):.4f} acc:{np.mean(accs):.4f} | Val loss:{val_loss:.4f} acc:{val_acc:.4f}")

            old_lr = self.params['lr']
            scheduler = SCHEDULERS.get(sched_name)
            if scheduler is None:
                raise ValueError(f"Unknown lr_scheduler: {sched_name}")
            new_lr = scheduler(self.initial_lr, epoch+1, total_epochs, **sched_params)
            self.params['lr'] = new_lr
            print(f" -- lr updated: {old_lr:.6f} -> {new_lr:.6f}")

            metrics = {
                "train_loss": np.mean(losses),
                "val_loss": val_loss,
                "train_acc": np.mean(accs),
                "val_acc": val_acc,
            }

            for callback in self.callbacks:
                callback.on_epoch_end(epoch, metrics, self)

        return self.model, val_acc

    def get_trained_models(self):
        callbacks = self.callbacks
        model_map = {}
        for callback in callbacks:
            if isinstance(callback, ModelCheckpoint):
                mlp = MLP(self.params['layers'], params['activations']) 
                mlp.load_model_weight(callback.mdoel_state)
                model_map[f'ckpt_{callback.monitor}_{callback.epoch}_{callback.best:.4f}'] = mlp
        return model_map
    
    def get_best_weights(self):
        model_state_map = []
        for callback in self.callbacks:
            if isinstance(callback, ModelCheckpoint):
                model_state_map.append({})
        return model_state_map
    
    def get_best_score(self):
        return self.callbacks[0].best
    
#--- data loading & utils ---
def load_data():
    x_train = np.load('../data/x_train.npy') / 255.
    x_test  = np.load('../data/x_test.npy') / 255.
    t_train = np.eye(10)[np.load('../data/y_train.npy').astype(int).flatten()]
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test  = x_test.reshape(x_test.shape[0], -1)
    return x_train, x_test, t_train

def create_batch(data, batch_size):
    n, mod = divmod(data.shape[0], batch_size)
    chunks = np.split(data[:n*batch_size], n)
    if mod:
        chunks.append(data[n*batch_size:])
    return chunks    

def load_model_from_ckpt(params, ckpt_path):
    mlp = MLP(params['layers'], params['activations'])
    mlp.load_model_weight(ckpt_path)
    return mlp

def save_sub_and_params(sub, prefix):
    timestamp = datetime.now().strftime('%m%d%H%M')
    base = f"{timestamp}_{prefix}"
    odir = 'sub/'
    os.makedirs(odir, exist_ok=True)
    sub.to_csv(f"{odir}{base}.csv", index=False)

    with open(f"{odir}{timestamp}.json", 'w') as f:
        json.dump(params, f, indent=2)

def construct_callbacks(params):
    callbacks = []
    for cb in params['callback']:
        if 'ckpt' in cb:
            callbacks.append(ModelCheckpoint(**cb['ckpt']))
    return callbacks

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    args = parser.parse_args()

    default = {
        'seed': 42,
        
        'lr': 0.5,
        'n_epochs': 100,
        'batch_size': 128,

        'activations': ['relu', 'relu', 'relu', 'softmax'],
        'layers': [784, 200, 100, 10],

        'lr_scheduler': 'linear',
        'lr_scheduler_params': {},

        # 'lr_scheduler': 'step',
        # 'lr_scheduler_params': {'step_size': 10, 'gamma': 0.1},

        # 'lr_scheduler': 'exp',
        # 'lr_scheduler_params': {'gamma': 0.9},

        # 'lr_scheduler': 'cosine',
        # 'lr_scheduler_params': {'eta_min': 0},

        'callback': [
            {'ckpt': {'dirpath': 'ckpt', 'monitor': 'val_acc', 'mode': 'max'}},
            {'ckpt': {'dirpath': 'ckpt', 'monitor': 'val_loss', 'mode': 'min'}},
        ]
    }
    if args.config:
        with open(args.config) as f:
            params = json.load(f)
    else:
        params = default
    return params

if __name__ == '__main__':
    params = get_params()
    np.random.seed(params['seed'])
    x_train, x_test, t_train = load_data()
    x_train, x_val, t_train, t_val = train_test_split(x_train, t_train, test_size=10000, random_state=params['seed'])
    trainer = Trainer(
        model=MLP(params['layers'], params['activations']),
        optimizer=None,
        loss_fn=crossentropy_loss,
        callbacks=construct_callbacks(params),
        params=params
    )
    trainer.fit(x_train, t_train, x_val, t_val)
    model_map = trainer.get_trained_models()

    for key, model in model_map.items():
        print(f"Model {key} - val_acc: {trainer.get_best_score()}")
        preds = []
        for xb in create_batch(x_test, params['batch_size']):
            preds.extend(model(xb).argmax(1).tolist())
        
        save_sub_and_params(pd.Series(preds, name='label'), key)
