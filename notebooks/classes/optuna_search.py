#%% optuna_integration.py
import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from activation_map import activation_map, deriv_map
from model import Model
from data import load_data, create_batch, crossentropy_loss

def train_model(mlp, x_train, t_train, x_val, t_val, params):
    """学習ループ：最後のエポックの検証Accを返すように変更"""
    best_val_acc = 0.0
    for epoch in range(params['n_epochs']):
        x_train, t_train = shuffle(x_train, t_train, random_state=params['seed'])
        x_batches = create_batch(x_train, params["batch_size"])
        t_batches = create_batch(t_train, params["batch_size"])

        for xb, tb in zip(x_batches, t_batches):
            yb = mlp(xb)
            delta = (yb - tb) / tb.shape[0]
            mlp.backward(delta)
            mlp.update(params["lr"])

        # エポックごとに検証
        yv = mlp(x_val)
        val_acc = accuracy_score(t_val.argmax(1), yv.argmax(1))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    return best_val_acc

def optuna_search(trial, x_train, t_train, x_val, t_val, cfg):
    # --- ハイパラサンプリング ---
    lr         = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    n1         = trial.suggest_int("n_units1", 32, 256, step=32)
    n2         = trial.suggest_int("n_units2", 32, 256, step=32)

    # --- params を上書き ---
    params = {
        "lr": lr,
        "n_epochs": cfg["train"]["n_epochs"],
        "batch_size": batch_size,
        "seed": cfg["train"]["seed"],
        "layers": [784, n1, n2, 10],
    }

    # --- 活性化関数マッピング ---
    act  = activation_map(cfg)
    dact = deriv_map(cfg)
    activations       = [act] * (len(params["layers"]) - 2) + [softmax]    # 最後はsoftmax
    deriv_activations = [dact] * (len(params["layers"]) - 2) + [deriv_softmax]

    # --- モデル構築・学習・評価 ---
    mlp = Model(params["layers"], activations, deriv_activations)
    val_acc = train_model(mlp, x_train, t_train, x_val, t_val, params)
    return val_acc

def main():
    # 設定読み込み
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))

    # データ読み込み・分割
    x, x_test, t = load_data()
    x_train, x_val, t_train, t_val = train_test_split(
        x, t, test_size=cfg["train"]["val_size"], random_state=cfg["train"]["seed"]
    )

    # Optuna サーチ
    sampler = optuna.samplers.TPESampler(seed=cfg["train"]["seed"])
    study = optuna.create_study(direction="maximize", sampler=sampler)
    func  = lambda trial: optuna_search(trial, x_train, t_train, x_val, t_val, cfg)
    study.optimize(func, n_trials=cfg["optuna"]["n_trials"])

    # ベストパラメータで再学習
    best = study.best_params
    print("Best hyperparameters:", best)
    cfg["train"].update(best)
    act  = activation_map(cfg)
    dact = deriv_map(cfg)
    layers = [784, best["n_units1"], best["n_units2"], 10]
    mlp = Model(layers,
                [act]*(len(layers)-2)+[softmax],
                [dact]*(len(layers)-2)+[deriv_softmax])
    _ = train_model(mlp, x_train, t_train, x_val, t_val, cfg["train"])

    # テスト予測・保存
    t_pred = []
    for xb in create_batch(x_test, cfg["train"]["batch_size"]):
        t_pred += mlp(xb).argmax(1).tolist()
    pd.Series(t_pred, name="label").to_csv("submission.csv", index_label="id")

    # サーチ結果・設定保存
    with open("optuna_study.json", "w") as f:
        json.dump(study.best_params, f)

if __name__ == "__main__":
    main()
