# train/callbacks.py
import os
import numpy as np

class Callback:
    def on_epoch_end(self, epoch, metrics, trainer): pass

class ModelCheckpoint(Callback):
    def __init__(self, dirpath, monitor="val_acc", mode="max"):
        os.makedirs(dirpath, exist_ok=True)
        self.dir = dirpath; self.monitor = monitor
        self.best = -np.inf if mode=="max" else np.inf
        self.mode = mode

    def on_epoch_end(self, epoch, metrics, trainer):
        score = metrics[self.monitor]
        improved = (score > self.best) if self.mode=="max" else (score < self.best)
        if improved:
            self.best = score
            path = os.path.join(self.dir, f"ckpt_epoch{epoch+1}.npz")
            np.savez(path, *trainer.model_state())
            print(f">>> ModelCheckpoint: saved {path}")

class EarlyStopping(Callback):
    def __init__(self, patience=5, monitor="val_loss", mode="min"):
        self.patience = patience; self.monitor = monitor; self.mode = mode
        self.best = np.inf if mode=="min" else -np.inf
        self.wait = 0

    def on_epoch_end(self, epoch, metrics, trainer):
        score = metrics[self.monitor]
        improved = (score < self.best) if self.mode=="min" else (score > self.best)
        if improved:
            self.best = score; self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f">>> EarlyStopping at epoch {epoch+1}")
                trainer.stop()

class SWA(Callback):
    def __init__(self, swa_start=10, swa_freq=5):
        self.swa_start = swa_start; self.freq = swa_freq
        self.n = 0

    def on_epoch_end(self, epoch, metrics, trainer):
        if epoch+1 >= self.swa_start and (epoch+1 - self.swa_start) % self.freq == 0:
            trainer.swa_accumulate()
            self.n += 1
            print(f">>> SWA: accumulated weights #{self.n}")

class CSVLogger(Callback):
    def __init__(self, filepath):
        self.fp = open(filepath, "w")
        self.fp.write("epoch," + ",".join(["train_loss","train_acc","val_loss","val_acc"])+"\n")

    def on_epoch_end(self, epoch, metrics, trainer):
        row = f"{epoch+1},{metrics['train_loss']:.4f},{metrics['train_acc']:.4f},{metrics['val_loss']:.4f},{metrics['val_acc']:.4f}\n"
        self.fp.write(row); self.fp.flush()

    def __del__(self):
        self.fp.close()
