# scripts/run_train.py
import yaml
from preprocessing import Preprocessor
from mlp import MLP
from trainer import Trainer
from callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, SWA
from utils import load_data, train_test_split, loss_fn
from activation_map import activation_map, deriv_map

def main():
    cfg = yaml.safe_load(open("config/default.yaml"))
    print(cfg)
    raw_x, raw_xt, raw_y = load_data()
    x_tr, x_val, y_tr, y_val = train_test_split(raw_x, raw_y, test_size=0.2)

    pre = Preprocessor(cfg)
    X_train = pre.fit_transform(x_tr)
    X_val   = pre.transform(x_val)

    model = MLP(cfg["model"]["layers"], activation_map(cfg), deriv_map(cfg))
    callbacks = []
    if cfg["callbacks"]["checkpoint"]["on"]:
        callbacks.append(ModelCheckpoint(cfg["callbacks"]["checkpoint"]["dir"], **cfg["callbacks"]["checkpoint"]))
    if cfg["callbacks"]["early_stopping"]["on"]:
        callbacks.append(EarlyStopping(**cfg["callbacks"]["early_stopping"]))
    if cfg["callbacks"]["swa"]["on"]:
        callbacks.append(SWA(**cfg["callbacks"]["swa"]))
    callbacks.append(CSVLogger("logs/train_log.csv"))

    trainer = Trainer(model, cfg, loss_fn, callbacks)
    best_acc = trainer.fit(X_train, y_tr, X_val, y_val)
    print("Best validation accuracy:", best_acc)

if __name__=="__main__":
    main()
