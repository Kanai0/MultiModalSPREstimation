# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 16:35:48 2025

@author: Kanai
"""

import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from viewer_3d_gui import SliceViewer3D
from ANNmodel import build_full_model, HybridLoss, compute_zn_volume
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import Callback

def launch_viewer(volume, title= "3D Volume Slice Viewer"):
    root = tk.Tk()
    app = SliceViewer3D(root, volume, title=title)
    root.mainloop()


class LiveLossPlotter(Callback):
    def __init__(self):
        super().__init__()
        self.tr, self.va = [], []
    def on_train_begin(self, logs=None):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Loss / Val Loss"); self.ax.set_xlabel("Epoch"); self.ax.set_ylabel("Loss")
        (self.l1,) = self.ax.plot([], [], label="loss")
        (self.l2,) = self.ax.plot([], [], label="val_loss")
        self.ax.legend(); self.fig.canvas.draw()
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.tr.append(logs.get("loss"))
        self.va.append(logs.get("val_loss"))
        x = range(1, len(self.tr)+1)
        self.l1.set_data(x, self.tr); self.l2.set_data(x, self.va)
        self.ax.relim(); self.ax.autoscale_view()
        self.fig.canvas.draw(); plt.pause(0.01)
    def on_train_end(self, logs=None):
        plt.ioff(); plt.show()

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
delta = 1e-6    # 1: physics-constrained loss, 0: standard loss
n_vox_batch = 262144 # 512*512
train_slc = slice(0, 40) # training slices
val_slc   = slice(40, 50) # validation slices
print("Loading images...")
par_dir = r'C:/Users/Kanai/Synology/TWMU/0_張先生_共同研究_MRI阻止能/FromIzo/data_summarized/'

# input data
ct_e1 = np.load(par_dir + 'simCT_80kVp.npy') 
ct_e2 = np.load(par_dir + 'simCT_120kVp.npy') 
ct_e3 = np.load(par_dir + 'simCT_140kVp.npy') 
mr_pd = np.load(par_dir + 'mr_pd.npy')
mr_pdw = np.load(par_dir + 'mr_pdw.npy')
wf = np.load(par_dir + 'wf.npy')

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

# --- 学習 ---
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=5, batch_size=n_vox_batch, shuffle=True,
    callbacks=[LiveLossPlotter()]
)
