# utils.py
import numpy as np
import json
import yaml
from sklearn.model_selection import train_test_split

def load_data_npz(path):
    """.npz 形式で x_train, y_train, x_test をまとめて持っている想定"""
    data = np.load(path)
    return data["x_train"], data["y_train"], data["x_test"]

def load_data():
    x_train = np.load('../../data/x_train.npy')
    t_train = np.load('../../data/y_train.npy')
    x_test  = np.load('../../data/x_test.npy')
    x_train, x_test = x_train/255., x_test/255.
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test  = x_test.reshape(x_test.shape[0], -1)
    t_train = np.eye(10)[t_train.astype('int32').flatten()]
    return x_train, x_test, t_train

def one_hot(labels, num_classes):
    """整数ラベル → one-hot"""
    return np.eye(num_classes)[labels.flatten().astype(int)]

def create_batches(X, y=None, batch_size=128):
    """X, y をバッチリストに分割。y=None の時は X のみ返す。"""
    N = X.shape[0]
    idx = np.arange(N)
    np.random.shuffle(idx)
    Xs = X[idx]
    if y is not None:
        ys = y[idx]
    batches = []
    for i in range(0, N, batch_size):
        xb = Xs[i:i+batch_size]
        if y is None:
            batches.append(xb)
        else:
            batches.append((xb, ys[i:i+batch_size]))
    return batches

def save_yaml(cfg, path):
    """辞書 → YAML"""
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    with open(path) as f:
        return json.load(f)

def loss_fn(t, y):
    return -np.sum(t * np.log(np.clip(y, 1e-10, 1.0))) / t.shape[0]