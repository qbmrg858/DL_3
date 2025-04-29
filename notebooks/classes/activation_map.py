# model/activation_map.py
import numpy as np

# --- まず各活性化関数とその導関数を定義（またはインポート） ---
def relu(x):
    return np.maximum(0, x)
def deriv_relu(x):
    return (x > 0).astype(x.dtype)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def deriv_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)
def deriv_tanh(x):
    return 1 - np.tanh(x) ** 2

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)
def deriv_softmax(x):
    # 通常、cross-entropyと組み合わせるので個別には使わないことが多いですが…
    s = softmax(x)
    return s * (1 - s)

# --- マッピング辞書 ---
_ACTIVATIONS = {
    "relu":    relu,
    "sigmoid": sigmoid,
    "tanh":    tanh,
    "softmax": softmax,
}

_DERIVATIVES = {
    "relu":    deriv_relu,
    "sigmoid": deriv_sigmoid,
    "tanh":    deriv_tanh,
    "softmax": deriv_softmax,
}

# --- 外部呼び出し用のヘルパー関数 ---
def activation_map(cfg):
    """config['model']['activation'] の文字列から活性化関数を返す"""
    names = cfg["model"]["activation"]
    functions = []
    for name in names:
        if name not in _ACTIVATIONS:
            raise KeyError(f"activation_map: unknown activation '{name}'")
        else:
            functions.append(_ACTIVATIONS[name]) 
    return functions

def deriv_map(cfg):
    """config['model']['activation'] の文字列から対応する導関数を返す"""
    names = cfg["model"]["activation"]
    functions = []
    for name in names:
        if name not in _DERIVATIVES:
            raise KeyError(f"deriv_map: unknown activation '{name}'")
        else:
            functions.append(_DERIVATIVES[name])
    return functions
