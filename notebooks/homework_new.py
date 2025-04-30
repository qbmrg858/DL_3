#%%
import os
import sys
import json
import argparse
import math
from matplotlib import pyplot as plt
import numpy as np
import optuna
import pandas as pd
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# --- preprocessing ---
def gamma_correction(x, gamma=1.0/2.0):
    return np.power(x, gamma)

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None
    def fit(self, x):
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0) + 1e-8
    def transform(self, x):
        return (x - self.mean) / self.std

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

def sgdr_cosine_annealing(initial_lr, epoch, total_epochs, T_0, T_mult=2, eta_min=0, **kwargs):
    T_cur = epoch
    T_i = T_0
    while T_cur >= T_i:
        T_cur -= T_i
        T_i *= T_mult
    return eta_min + 0.5 * (initial_lr - eta_min) * (1 + math.cos(math.pi * T_cur / T_i))

def one_cycle_lr(initial_lr, epoch, total_epochs, max_lr=0.1, pct_start=0.3, **kwargs):
    if epoch / total_epochs < pct_start:
        return initial_lr + (max_lr - initial_lr) * (epoch / (pct_start * total_epochs))
    else:
        return max_lr * (1 - (epoch - pct_start * total_epochs) / ((1 - pct_start) * total_epochs))

SCHEDULERS = {
    "linear": linear_decay,
    "step":   step_decay,
    "exp":    exp_decay,
    "cosine": cosine_annealing,
    "sgdr": sgdr_cosine_annealing,
    "one_cycle": one_cycle_lr,
}

#--- loss ---
def np_log(x):
    return np.log(np.clip(x, 1e-10, 1e+10))
def crossentropy_loss(t, y):
    return -np.sum(t * np_log(y)) / t.shape[0]

class Dense:
    def __init__(self, in_dim, out_dim, activation, deriv_activation, dropout_rate=0.0):
        self.W = np.random.uniform(-0.08, 0.08, (in_dim, out_dim)).astype('float64')
        self.b = np.zeros(out_dim, dtype='float64')
        self.v_W = np.zeros_like(self.W)
        self.v_b = np.zeros_like(self.b)

        self.activation = activation
        self.deriv_activation = deriv_activation
        self.x = None
        self.u = None

        self.dropout_rate = dropout_rate
        self.dropout_mask = None

    def __call__(self, x, training=False):
        self.x = x
        self.u = x.dot(self.W) + self.b
        out = self.activation(self.u)
        if training and self.dropout_rate > 0:
            self.dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, size=out.shape) / (1 - self.dropout_rate)
            out *= self.dropout_mask
        return out 

    def b_prop(self, delta, W_next):
        self.delta = self.deriv_activation(self.u) * (delta.dot(W_next.T))
        return self.delta

    def compute_grad(self):
        m = self.delta.shape[0]
        self.dW = self.x.T.dot(self.delta) / m
        self.db = np.sum(self.delta, axis=0) / m

    def get_grads(self):
        return self.dW, self.db

class MLP:
    def __init__(self, layer_sizes, activation_names):
        self.layers = []
        # construct layers with matching derivative
        for i in range(len(layer_sizes) - 1):
            act_name = activation_names[i]
            act = ACTIVATION_MAP[act_name]
            deriv = DERIV_MAP[act_name]
            self.layers.append(Dense(layer_sizes[i], layer_sizes[i+1], act, deriv))

    def __call__(self, x, training=False):
        for layer in self.layers:
            x = layer(x, training=training)
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
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'lr': [],
        }

        self.initial_lr = self.params['lr']

    def model_state(self):
        state = {}
        for i, layer in enumerate(self.model.layers):
            state[f"W{i}"] = layer.W.copy()
            state[f"b{i}"] = layer.b.copy()
        return state
    
    def stop(self):
        self.stop_flag = False

    def fit(self, x_train, t_train, x_val, t_val):
        total_epochs = self.params['n_epochs']
        sched_name = self.params['lr_scheduler']
        sched_params = self.params.get('lr_scheduler_params', {})

        for epoch in range(self.params['n_epochs']):
            x_train, t_train = shuffle(x_train, t_train)
            x_batches = create_batch(x_train, self.params['batch_size'])
            t_batches = create_batch(t_train, self.params['batch_size'])

            losses, accs = [], []
            for xb, tb in zip(x_batches, t_batches):
                yb = self.model(xb, training=True)
                losses.append(self.loss_fn(tb, yb) + self.compute_l2_penalty())
                accs.append(accuracy_score(tb.argmax(1), yb.argmax(1)))
                delta = (yb - tb)
                self.model.backward(delta)
                self.model.update(self.params['lr'])
            yv = self.model(x_val, training=False)
            val_loss = self.loss_fn(t_val, yv) + self.compute_l2_penalty()
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
                "lr": self.params['lr'],
            }
            for key in self.history:
                self.history[key].append(metrics[key])

            for callback in self.callbacks:
                callback.on_epoch_end(epoch, metrics, self)

        return self.model, val_acc
    def compute_l2_penalty(self):
        l2_lambda = self.params['l2']
        if l2_lambda == 0: return 0
        return l2_lambda * sum(np.sum(layer.W ** 2) for layer in self.model.layers)

    def plot_learning_curves(self):
        epochs = range(1, len(self.history['train_loss']) + 1)

        plt.figure(figsize=(14, 5))

        # --- Loss Curve ---
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history['train_loss'], label='Train Loss')
        plt.plot(epochs, self.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()

        # --- Accuracy Curve ---
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history['train_acc'], label='Train Acc')
        plt.plot(epochs, self.history['val_acc'], label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curve')
        plt.legend()

        plt.show()
    def plot_lr_curve(self):
        epochs = range(1, len(self.history['lr']) + 1)

        plt.figure(figsize=(7, 5))
        plt.plot(epochs, self.history['lr'], label='Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.legend()
        plt.show()

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

def tune_hyperparameters(x_train, t_train, x_val, t_val, n_trials=30):
    best_p = None
    best_acc = 0.0

    def objective(trial):
        nonlocal best_p, best_acc  
        p = {
            'seed': 42,
            'gamma_cor': trial.suggest_float('gamma_cor', 0.1, 2.0),
            'lr': trial.suggest_float('lr', 1e-4, 0.5, log=True),
            'n_epochs': 100,
            'batch_size': trial.suggest_int('batch_size', 32, 512),
            'activations': ['relu', 'relu', 'relu', 'relu', 'softmax'],
            'layers': [784, 512, 256, 128, 10],
            'dropout': [0.0,
                        trial.suggest_float('dropout_1', 0.0, 0.5),
                        trial.suggest_float('dropout_2', 0.0, 0.5),
                        trial.suggest_float('dropout_3', 0.0, 0.5),
                        0.0],
            'l2': trial.suggest_float('l2', 1e-3, 10, log=True),
            'lr_scheduler': 'one_cycle',
            'lr_scheduler_params': {
                'max_lr': trial.suggest_float('max_lr', 0.51, 0.99),
                'pct_start': trial.suggest_float('pct_start', 0.1, 0.8)
            },
            'callback': [
                {'ckpt': {'dirpath': 'ckpt', 'monitor': 'val_acc', 'mode': 'max'}},
                {'ckpt': {'dirpath': 'ckpt', 'monitor': 'val_loss', 'mode': 'min'}}
            ]
        }

        np.random.seed(p['seed'])
        x_train, x_test, t_train = load_data()
        trainer = run_train(x_train, x_test, t_train, p)
        score = trainer.get_best_score()

        if score > best_acc:
            best_acc = score
            best_p = p.copy()

        print(f"[Trial {trial.number}] acc={score:.4f}")
        print(f"     with params: {p}")
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    print("Best params:", best_p)
    print("Best validation accuracy:", best_acc)
    return best_p

def run_train(x_train, x_test, t_train, params):
    np.random.seed(params['seed'])
    x_train, x_val, t_train, t_val = train_test_split(x_train, t_train, test_size=10000, random_state=params['seed'])

    x_train = gamma_correction(x_train, params['gamma_cor'])
    x_val = gamma_correction(x_val, params['gamma_cor'])
    x_test = gamma_correction(x_test, params['gamma_cor'])

    # scaler = StandardScaler()
    # scaler.fit(x_train)
    # x_train = scaler.transform(x_train)
    # x_val = scaler.transform(x_val)
    # x_test = scaler.transform(x_test)

    trainer = Trainer(
        model=MLP(params['layers'], params['activations']),
        optimizer=None,
        loss_fn=crossentropy_loss,
        callbacks=construct_callbacks(params),
        params=params
    )
    trainer.fit(x_train, t_train, x_val, t_val)
    return trainer

DEFAULT_PRAMS = {
    'seed': 42,
    
    'gamma_cor': 0.5,

    'lr': 0.5,
    'n_epochs': 20,
    'batch_size': 128,

    # 'activations': ['relu', 'relu', 'relu', 'relu', 'softmax'],
    # 'layers': [784, 512, 256, 128, 10],

    'activations': ['relu', 'relu', 'relu', 'softmax'],
    'layers': [784, 100, 100, 10],
    'dropout': [0.0, 0.5, 0.5, 0.0],

    'l2': 0.1,

    # 'lr_scheduler': 'linear',
    # 'lr_scheduler_params': {},

    'lr_scheduler': 'step',
    'lr_scheduler_params': {'step_size': 3, 'gamma': 0.5},

    # 'lr_scheduler': 'exp',
    # 'lr_scheduler_params': {'gamma': 0.95},

    # 'lr_scheduler': 'cosine',
    # 'lr_scheduler_params': {'eta_min': 0},

    # 'lr_scheduler': 'sgdr',
    # 'lr_scheduler_params': {'T_0': 10, 'T_mult': 1, 'eta_min': 0.001},

    # 'lr_scheduler': 'one_cycle',
    # 'lr_scheduler_params': {'max_lr': 0.7, 'pct_start': 0.2},

    'callback': [
        {'ckpt': {'dirpath': 'ckpt', 'monitor': 'val_acc', 'mode': 'max'}},
        {'ckpt': {'dirpath': 'ckpt', 'monitor': 'val_loss', 'mode': 'min'}},
    ]
}

if __name__ == '__main__':
    # params = get_params()
    x_train, x_test, t_train = load_data()
    params = DEFAULT_PRAMS.copy()
    # params = tune_hyperparameters(x_train, t_train, x_test, t_train, n_trials=500)
    print("Best params after tuning:", params)

    trainer = run_train(x_train, x_test, t_train, params)
    model_map = trainer.get_trained_models()

    for key, model in model_map.items():
        print(f"Model {key} - val_acc: {trainer.get_best_score()}")
        preds = []
        for xb in create_batch(x_test, params['batch_size']):
            preds.extend(model(xb).argmax(1).tolist())
        
        save_sub_and_params(pd.Series(preds, name='label'), key)
    
    trainer.plot_learning_curves()
    trainer.plot_lr_curve()