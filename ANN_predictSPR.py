# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 16:35:48 2025

@author: Kanai
"""

import os
from datetime import datetime
import numpy as np
import tkinter as tk
import matplotlib
import json
import shutil
import inspect
import sys
matplotlib.use("TkAgg")   # ここで指定（pyplotより前）
import matplotlib.pyplot as plt
from viewer_3d_gui import launch_viewer
from ANNmodel import build_full_model, HybridLoss, compute_zn_volume
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import Callback, CSVLogger


class LiveLossPlotter(Callback):
    def __init__(self, with_val=True, save_dir=None, fname="loss_curve.png", dpi=150):
        super().__init__()
        self.tr, self.va = [], []
        self.with_val = with_val
        self.save_dir = save_dir
        self.fname = fname
        self.dpi = dpi

    def on_train_begin(self, logs=None):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Loss / Val Loss")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        (self.l1,) = self.ax.plot([], [], "o-", label="loss")
        (self.l2,) = self.ax.plot([], [], "o-", label="val_loss")
        self.ax.legend()

        # ▼ ここを追加：原点(0,0)で軸が交差するようにする
        self.ax.spines["left"].set_position(("data", 0.0))    # x=0 で y軸
        self.ax.spines["bottom"].set_position(("data", 0.0))  # y=0 で x軸
        self.ax.spines["right"].set_color("none")
        self.ax.spines["top"].set_color("none")
        self.ax.xaxis.set_ticks_position("bottom")
        self.ax.yaxis.set_ticks_position("left")

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if logs.get("loss") is not None:
            self.tr.append(float(logs["loss"]))
        if logs.get("val_loss") is not None:
            self.va.append(float(logs["val_loss"]))

        x_tr = np.arange(1, len(self.tr) + 1)
        self.l1.set_data(x_tr, self.tr)
        if self.va:
            x_va = np.arange(1, len(self.va) + 1)
            self.l2.set_data(x_va, self.va)

        # ▼ ここを変更：必ず 0 を含む範囲に固定（原点を軸交点に）
        xmax = max(2.0, len(self.tr) + 0.5)
        self.ax.set_xlim(0.0, xmax)

        y_all = np.array(self.tr + (self.va if self.va else []), dtype=float)
        if y_all.size:
            ymax = float(np.nanmax(y_all))
            pad = max(1e-12, 0.02 * ymax)
            self.ax.set_ylim(0.0, ymax + pad)  # 下限を 0 に固定

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def on_train_end(self, logs=None):
        # 追加：最後にJPGで保存
        if self.save_dir is not None and hasattr(self, "fig"):
            os.makedirs(self.save_dir, exist_ok=True)
            out_path = os.path.join(self.save_dir, self.fname)
            self.fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.ioff()
        plt.show()

class EpochIntervalSaver(Callback):
    def __init__(self, save_dir, every=5):
        super().__init__()
        self.save_dir = save_dir
        self.every = int(every)
    def on_epoch_end(self, epoch, logs=None):
        # epochは0始まりなので+1
        if (epoch + 1) % self.every == 0:
            path = os.path.join(self.save_dir, f"model_epoch_{epoch+1:04d}.h5")
            self.model.save(path)

def build_xy_from_slices(slc, ct_target, 
                         ct_e1, ct_e2, ct_e3, mr_pd, mr_pdw,
                         rho_gt, z362, z186):
    # (D,H,W) -> (N,5,1) と (N,4)
    feat = np.stack([ct_e1[slc], ct_e2[slc], ct_e3[slc], mr_pd[slc], mr_pdw[slc]], axis=-1)  # (d,h,w,5)
    X = feat.reshape(-1, 5).astype(np.float32)[:, :, None]  # (N,5,1)
    targ = np.stack([rho_gt[slc], ct_target[slc], z362[slc], z186[slc]], axis=-1)           # (d,h,w,4)
    Y = targ.reshape(-1, 4).astype(np.float32)                                              # (N,4)
    return X, Y

# Atomic data for H C N O Na Mg Si P S Cl K Ca Fe I Ar
# 
A_arr = np.array([1.00784, 12.0096, 14.00643, 15.99903, 22.98976, 24.304, 28.0855, 30.97376, 32.059, 35.446, 39.0983, 40.078, 55.845, 126.90447,39.948], dtype=np.float32)
Z_arr = np.array([1,6,7,8,11,12,14,15,16,17,19,20,26,53,18], dtype=np.float32)

# --- ハイパーパラメータ ---
delta = 0    # 1: physics-constrained loss, 0: standard loss
n_vox_batch = 16384 
epochs = 20
save_every = 5 # epochs to save model
train_slc = slice(0, 40) # training slices
val_slc   = slice(40, 50) # validation slices
print("Loading images...")
par_dir = r'C:/Users/Kanai/Synology/TWMU/0_張先生_共同研究_MRI阻止能/FromIzo/data_summarized/'
save_root = r"C:/Users/Kanai/Desktop/MRtoSPR/Chang_Results"   # 結果を保存したい親フォルダ

# input data
ct_e1 = np.load(par_dir + 'simCT_80kVp.npy').astype(np.float32) 
ct_e2 = np.load(par_dir + 'simCT_120kVp.npy').astype(np.float32) 
ct_e3 = np.load(par_dir + 'simCT_140kVp.npy').astype(np.float32)
mr_pd = np.load(par_dir + 'mr_pd.npy').astype(np.float32)
mr_pdw = np.load(par_dir + 'mr_pdw.npy').astype(np.float32)
wf = np.load(par_dir + 'wf.npy').astype(np.float32)

# normalize data
ct_e1 = ct_e1 - 1000
ct_e2 = ct_e2 - 1000
ct_e3 = ct_e3 - 1000

# ground truth
rho_gt = np.load(par_dir + 'rho_elem.npy')
dim = ct_e1.shape

# --- Physics Constrained Loss用事前計算 ---
z_3_62 = compute_zn_volume(wf, Z_arr, A_arr, n=3.62, normalize_w=False)
z_1_86 = compute_zn_volume(wf, Z_arr, A_arr, n=1.86, normalize_w=False)

# devide dataset into training and validation
X_train, Y_train = build_xy_from_slices(train_slc, ct_e1, ct_e1, ct_e2, ct_e3, mr_pd, mr_pdw, rho_gt, z_3_62, z_1_86)
X_val,   Y_val   = build_xy_from_slices(val_slc,   ct_e1, ct_e1, ct_e2, ct_e3, mr_pd, mr_pdw, rho_gt, z_3_62, z_1_86)

# x = np.stack([ct_e1.ravel(), ct_e2.ravel(), ct_e3.ravel(), mr_pd.ravel(), mr_pdw.ravel()], axis=-1)
# x = x[..., np.newaxis].astype(np.float32)  # shape = (N_vox, n_ch, 1)

# n_ch = x.shape[-2] # MRI+DECTチャネル数: T1DF, T1DW, T2-STIR, HighE, LowE, ρₑ, VMI



# --- モデルコンパイル ---

seq_len, n_ch = X_train.shape[1], X_train.shape[2]   # -> 5, 1
model = build_full_model(input_shape=(seq_len, n_ch))
loss_fn = HybridLoss(delta=delta)  # 実験値で調整
model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=loss_fn)
y = np.stack([rho_gt.ravel().astype(np.float32), 
              ct_e1.ravel().astype(np.float32), 
              z_3_62.ravel().astype(np.float32), 
              z_1_86.ravel().astype(np.float32)
              ], axis=-1)

# make directory to save results
run_dir = os.path.join(save_root, f"ANN_{datetime.now().strftime('%Y%m%d_%H%S')}")
os.makedirs(run_dir, exist_ok=True)

# save current script and ANNmodel.py
try:
    # この実行スクリプト（ANN_predictSPR.py）
    this_script = os.path.abspath(__file__)
    shutil.copy2(this_script, os.path.join(run_dir, os.path.basename(this_script)))
except Exception as e:
    print(f"[warn] failed to save current script: {e}")

try:
    import ANNmodel as _annmod
    annmodel_path = os.path.abspath(_annmod.__file__)
    shutil.copy2(annmodel_path, os.path.join(run_dir, os.path.basename(annmodel_path)))
except Exception as e:
    print(f"[warn] failed to save ANNmodel.py: {e}")

# execute training
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=1, batch_size=n_vox_batch, shuffle=True,
    callbacks=[
        LiveLossPlotter(save_dir=run_dir),                 # ← ここを修正
        CSVLogger(os.path.join(run_dir, "history.csv"), append=False),
        EpochIntervalSaver(run_dir, every=save_every),
    ],
)

# Save final model and history
model.save(os.path.join(run_dir, "model_final.h5"))
with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as f:
    json.dump(history.history, f, ensure_ascii=False, indent=2)
np.savez_compressed(
    os.path.join(run_dir, "history.npz"),
    **{k: np.asarray(v) for k, v in history.history.items()}
)
