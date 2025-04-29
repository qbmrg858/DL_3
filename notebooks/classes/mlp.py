# model/mlp.py
import numpy as np

class Dense:
    def __init__(self, in_dim, out_dim, act, d_act):
        self.W = np.random.randn(in_dim, out_dim) * np.sqrt(2.0/in_dim)
        self.b = np.zeros(out_dim)
        self.act, self.d_act = act, d_act
        self.x, self.u, self.delta = None, None, None

    def __call__(self, x):
        self.x = x
        self.u = x @ self.W + self.b
        return self.act(self.u)

    def b_prop(self, delta, W_next):
        self.delta = self.d_act(self.u) * (delta @ W_next.T)
        return self.delta

    def compute_grad(self):
        bs = self.x.shape[0]
        self.dW = self.x.T @ self.delta / bs
        self.db = np.sum(self.delta, axis=0) / bs

    def get_grads(self):
        return self.dW, self.db

class MLP:
    def __init__(self, layers, activations, deriv_acts):
        self.layers = []
        for i in range(len(layers)-1):
            self.layers.append(Dense(layers[i], layers[i+1],
                                     activations[i],
                                     deriv_acts[i]))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, delta):
        W_next = None
        for layer in reversed(self.layers):
            if W_next is None:
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
