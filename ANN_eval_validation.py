#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate a trained model on the validation split (interactive-first).

- Parameters are set directly in this file (no argparse).
- Designed for VS Code Interactive Window: variables persist after runs.
- Also runnable as a plain script; uses the same parameters.
"""

import os
import json
from datetime import datetime
from typing import Optional, Tuple, Dict
from viewer_3d_gui import launch_viewer
import numpy as np
import tensorflow as tf

from ANNmodel import (
    build_full_model,  # kept for compatibility / reference
    HybridLoss,
    compute_zn_volume,
    physics_ct_loss,
    kph, kcoh, kincoh, z3_62_water, z1_86_water,
)


def build_xy_from_slices(slc, ct_target,
                         ct_e1, ct_e2, ct_e3, mr_pd, mr_pdw,
                         rho_gt, z362, z186):
    """
    Replicates training script's data shaping:
      - Features: [ct_e1, ct_e2, ct_e3, mr_pd, mr_pdw] -> (N, 5, 1)
      - Targets:  [rho_true, ct_target, z3.62, z1.86]  -> (N, 4)
    """
    feat = np.stack([
        ct_e1[slc], ct_e2[slc], ct_e3[slc], mr_pd[slc], mr_pdw[slc]
    ], axis=-1)  # (d,h,w,5)
    X = feat.reshape(-1, 5).astype(np.float32)[:, :, None]  # (N,5,1)

    targ = np.stack([
        rho_gt[slc], ct_target[slc], z362[slc], z186[slc]
    ], axis=-1)  # (d,h,w,4)
    Y = targ.reshape(-1, 4).astype(np.float32)  # (N,4)
    return X, Y


def compute_ct_from_rho(rho_pred, z362, z186):
    """Compute CT (HU) from predicted rho using the same physics formula."""
    num = rho_pred * (kph * z362 + kcoh * z186 + kincoh)
    den = (kph * z3_62_water + kcoh * z1_86_water + kincoh)
    return 1000.0 * (num / den - 1.0)



model_dir = r"C:/Users/Kanai/Desktop/MRtoSPR/Chang_Results/ANN_20250911_1118/"
model_path = model_dir + r"model_epoch_0020.h5"
data_dir = r'C:/Users/Kanai/Synology/TWMU/0_張先生_共同研究_MRI阻止能/FromIzo/data_summarized/'
val_slc = slice(40, 50)
batch_size = 16384
out_dir = model_dir + "EvaluationResults"
try_compiled = True


model_path = os.path.abspath(model_path)

# Output directory organization
if out_dir is None:
    base = os.path.dirname(model_path)
    stamp = datetime.now().strftime("eval_%Y%m%d_%H%M%S")
    out_dir = os.path.join(base, stamp)
os.makedirs(out_dir, exist_ok=True)

print("[Info] Loading input arrays from:", data_dir)
# Inputs
ct_e1 = np.load(os.path.join(data_dir, 'simCT_80kVp.npy')).astype(np.float32)
ct_e2 = np.load(os.path.join(data_dir, 'simCT_120kVp.npy')).astype(np.float32)
ct_e3 = np.load(os.path.join(data_dir, 'simCT_140kVp.npy')).astype(np.float32)
mr_pd = np.load(os.path.join(data_dir, 'mr_pd.npy')).astype(np.float32)
mr_pdw = np.load(os.path.join(data_dir, 'mr_pdw.npy')).astype(np.float32)
wf = np.load(os.path.join(data_dir, 'wf.npy')).astype(np.float32)
rho_gt = np.load(os.path.join(data_dir, 'rho_elem.npy')).astype(np.float32)

# Normalize CT inputs like training (CT-1000)
ct_e1 = ct_e1 - 1000.0
ct_e2 = ct_e2 - 1000.0
ct_e3 = ct_e3 - 1000.0

# Precompute z_n volumes for physics constraint
print("[Info] Computing z_n volumes (3.62, 1.86)...")
# Atomic data arrays (must match training)
A_arr = np.array([1.00784, 12.0096, 14.00643, 15.99903, 22.98976, 24.304, 28.0855, 30.97376, 32.059, 35.446,
                    39.0983, 40.078, 55.845, 126.90447, 39.948], dtype=np.float32)
Z_arr = np.array([1, 6, 7, 8, 11, 12, 14, 15, 16, 17, 19, 20, 26, 53, 18], dtype=np.float32)

z_3_62 = compute_zn_volume(wf, Z_arr, A_arr, n=3.62, normalize_w=False)
z_1_86 = compute_zn_volume(wf, Z_arr, A_arr, n=1.86, normalize_w=False)

# Build validation set
X_val, Y_val = build_xy_from_slices(val_slc, ct_e1, ct_e1, ct_e2, ct_e3, mr_pd, mr_pdw,
                                    rho_gt, z_3_62, z_1_86)

# Load model
print("[Info] Loading model:", model_path)
model = None
compiled_eval = None
if try_compiled:
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'HybridLoss': HybridLoss,
                'physics_ct_loss': physics_ct_loss,
            },
            compile=True,
        )
        # Run evaluate with the original compiled loss if available
        print("[Info] Running model.evaluate with compiled loss...")
        compiled_eval = model.evaluate(X_val, Y_val, batch_size=batch_size, return_dict=True, verbose=1)
    except Exception as e:
        print(f"[Warn] Compiled load/evaluate failed: {e}\n[Info] Falling back to manual metric computation.")

if model is None:
    # Load model weights/graph without compilation
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'HybridLoss': HybridLoss,
            'physics_ct_loss': physics_ct_loss,
        },
        compile=False,
    )

# Predict rho on validation set
print("[Info] Predicting on validation set...")
y_pred = model.predict(X_val, batch_size=batch_size, verbose=1).astype(np.float32).reshape(-1)

# Extract ground truth arrays for metrics
rho_true = Y_val[:, 0].reshape(-1)
ct_true = Y_val[:, 1].reshape(-1)
z362_v = Y_val[:, 2].reshape(-1)
z186_v = Y_val[:, 3].reshape(-1)

# Metrics for rho
diff_rho = y_pred - rho_true
mse_rho = float(np.mean(np.square(diff_rho)))
rmse_rho = float(np.sqrt(mse_rho))
mae_rho = float(np.mean(np.abs(diff_rho)))

# Physics CT comparison
ct_pred = compute_ct_from_rho(y_pred, z362_v, z186_v)
diff_ct = ct_pred - ct_true
mse_ct = float(np.mean(np.square(diff_ct)))
rmse_ct = float(np.sqrt(mse_ct))
mae_ct = float(np.mean(np.abs(diff_ct)))

# Optional R^2 for rho
ss_res = float(np.sum(np.square(diff_rho)))
ss_tot = float(np.sum(np.square(rho_true - float(np.mean(rho_true)))))
r2_rho = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')

results = {
    'compiled_evaluate': compiled_eval,  # may be None
    'rho_metrics': {
        'mse': mse_rho,
        'rmse': rmse_rho,
        'mae': mae_rho,
        'r2': r2_rho,
    },
    'ct_metrics': {
        'mse': mse_ct,
        'rmse': rmse_ct,
        'mae': mae_ct,
    },
    'counts': {
        'n_vox_val': int(Y_val.shape[0])
    }
}

# Save metrics and arrays
metrics_path = os.path.join(out_dir, 'eval_metrics.json')
with open(metrics_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# Save predictions reshaped to (D,H,W)
d_all, h, w = ct_e1.shape
d_val = val_slc.stop - val_slc.start
pred_vol = y_pred.reshape(d_val, h, w)
true_vol = rho_true.reshape(d_val, h, w)
np.save(os.path.join(out_dir, 'rho_pred_val.npy'), pred_vol)
np.save(os.path.join(out_dir, 'rho_true_val.npy'), true_vol)
np.save(os.path.join(out_dir, 'ct_pred_from_rho_val.npy'), ct_pred.reshape(d_val, h, w))
np.save(os.path.join(out_dir, 'ct_true_val.npy'), ct_true.reshape(d_val, h, w))
