# train/trainer.py
import numpy as np
from sklearn.utils import shuffle

class Trainer:
    def __init__(self, model, optimizer, loss_fn, callbacks, cfg):
        self.model, self.opt, self.loss_fn = model, optimizer, loss_fn
        self.callbacks, self.cfg = callbacks, cfg
        self.stop_flag = False
        # for SWA
        self.swa_n = 0
        self.swa_W = None
        self.swa_b = None

    def model_state(self):
        # return list of params to save
        Ws = [lay.W for lay in self.model.layers]
        bs = [lay.b for lay in self.model.layers]
        return Ws + bs

    def swa_accumulate(self):
        # simple average of weights
        Ws_bs = self.model_state()
        if self.swa_W is None:
            self.swa_W = [w.copy() for w in Ws_bs]
        else:
            for i, w in enumerate(Ws_bs):
                self.swa_W[i] = (self.swa_W[i]*self.swa_n + w)/(self.swa_n+1)
        self.swa_n += 1

    def stop(self):
        self.stop_flag = True

    def fit(self, x_train, y_train, x_val, y_val):
        for epoch in range(self.cfg["n_epochs"]):
            # --- train ---
            x_train, y_train = shuffle(x_train, y_train)
            xbatches = np.array_split(x_train, len(x_train)//self.cfg["batch_size"])
            ybatches = np.array_split(y_train, len(y_train)//self.cfg["batch_size"])

            train_losses, train_accs = [], []
            for xb, yb in zip(xbatches, ybatches):
                y_pred = self.model(xb)
                loss = self.loss_fn(yb, y_pred)
                train_losses.append(loss)
                acc = (yb.argmax(1)==y_pred.argmax(1)).mean()
                train_accs.append(acc)

                delta = (y_pred - yb)/yb.shape[0]
                self.model.backward(delta)
                self.model.update(self.opt.lr)

            # --- valid ---
            yv = self.model(x_val)
            val_loss = self.loss_fn(y_val, yv)
            val_acc  = (y_val.argmax(1)==yv.argmax(1)).mean()

            metrics = {
                "train_loss": np.mean(train_losses),
                "train_acc":  np.mean(train_accs),
                "val_loss":   val_loss,
                "val_acc":    val_acc
            }
            for cb in self.callbacks:
                cb.on_epoch_end(epoch, metrics, self)

            if self.stop_flag:
                break

        # at end, if SWA used, swap in swa weights
        if self.swa_W is not None:
            L = len(self.model.layers)
            for i in range(L):
                self.model.layers[i].W = self.swa_W[i]
                self.model.layers[i].b = self.swa_W[L+i]

        return val_acc
