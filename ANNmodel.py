# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input, optimizers, backend as K

kph = 9.094e-6
kcoh = 1.064e-3
kincoh = 0.5988
rho_e_water = 3.343e23  # e-/cm3
z3_62_water = 7.522
z1_86_water = 7.115

# Residual Block:
# Residual Block A: kernel size=3 (default)
# Residual Block B: kernel size=2 (specified in input arguments)
# strides=2 will not function properly, so used strides=1
def residual_block(x, filters, kernel_size=3, strides=1, name_prefix="A"):

    """
    Projection shortcut:
      - 自動判定: (strides != 1) or (in_channels != filters) のとき 1x1 Conv を適用
    """
    in_ch = K.int_shape(x)[-1]
    use_proj = (strides != 1) or (in_ch != filters)
    
    # shortcut path
    shortcut = x
    if use_proj:
        shortcut = layers.Conv1D(filters, 1, strides=strides, padding="same",
                                 name=f"{name_prefix}_proj")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name_prefix}_proj_bn")(shortcut)

    # main path
    x = layers.Conv1D(filters, kernel_size, strides=strides, padding='same', name=f"{name_prefix}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(filters, kernel_size, strides=1, padding='same', name=f"{name_prefix}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    x = layers.Add(name=f"{name_prefix}_add")([x, shortcut])
    x = layers.ReLU()(x)
    return x

# B-type branch without convA/B (B1-B2-B3-B2-B1)
def build_branch_B(input_tensor, name_prefix="B"):
    x = residual_block(input_tensor, filters=64, kernel_size=2, strides=1, name_prefix=f"{name_prefix}1") # B1
    x = residual_block(x, filters=128, kernel_size=2, strides=1, name_prefix=f"{name_prefix}2") # B2
    x = residual_block(x, filters=256, kernel_size=2, strides=1, name_prefix=f"{name_prefix}3") # B3
    x = residual_block(x, filters=128, kernel_size=2, strides=1, name_prefix=f"{name_prefix}2_back") # B2
    x = residual_block(x, filters=64, kernel_size=2, strides=1, name_prefix=f"{name_prefix}1_back") # B1
    return x

# A-type branch (A1-A2-A3-A2-A1, no final conv)
def build_branch_A(input_tensor, name_prefix="A"):
    x = residual_block(input_tensor, filters=64, kernel_size=3, strides=1, name_prefix=f"{name_prefix}1")
    x = residual_block(x, filters=128, kernel_size=3, strides=1, name_prefix=f"{name_prefix}2")
    x = residual_block(x, filters=256, kernel_size=3, strides=1, name_prefix=f"{name_prefix}3")
    x = residual_block(x, filters=128, kernel_size=3, strides=1, name_prefix=f"{name_prefix}2_back")
    x = residual_block(x, filters=64, kernel_size=3, strides=1, name_prefix=f"{name_prefix}1_back")
    x = layers.GlobalAveragePooling1D()(x)
    return x

# Full model with shared ConvA + 4 branches (3B + 1A)
def build_full_model(input_shape=(40, 1)):
    inputs = Input(shape=input_shape, name="model_input")

    # Shared initial ConvA + ReLU
    x = layers.Conv1D(64, kernel_size=7, strides=2, padding='same', name="ConvA_shared")(inputs)
    x = layers.ReLU(name="relu_shared")(x)

    # 3 B-type and 1 A-type branches
    branch1 = build_branch_B(x, name_prefix="B1")
    branch1 = layers.Conv1D(64, kernel_size=3, strides=1, padding='same', name=f"convB_final1")(branch1)
    branch1 = layers.GlobalAveragePooling1D()(branch1)

    branch2 = build_branch_B(x, name_prefix="B2")
    branch2 = layers.Conv1D(64, kernel_size=3, strides=1, padding='same', name=f"convB_final2")(branch2)
    branch2 = layers.GlobalAveragePooling1D()(branch2)

    branch3 = build_branch_B(x, name_prefix="B3")
    branch3 = layers.Conv1D(64, kernel_size=3, strides=1, padding='same', name=f"convB_final3")(branch3)
    branch3 = layers.GlobalAveragePooling1D()(branch3)
    
    branch4 = build_branch_A(x, name_prefix="A1")

    # Concatenate and FC layers
    merged = layers.Concatenate(name="concat")([branch1, branch2, branch3, branch4])
    x = layers.Dense(64, activation='relu', name="fc1")(merged)
    x = layers.Dense(64, activation='relu', name="fc2")(x)
    output = layers.Dense(1, activation='linear', name="rho_m")(x)

    model = models.Model(inputs=inputs, outputs=output, name="Chang_ResNet_Model_Corrected")
    return model

# --- Calculation of effective atomic number using Eq.(5) ---
def compute_z_n(Z, A, w, n, normalize_w=True, eps=1e-12):
    """
    Eq.(5): z_n = ( sum_i [ w_i * (Z_i/A_i) * Z_i^n ] / sum_i [ w_i * (Z_i/A_i) ] )^(1/n)

    Parameters
    ----------
    Z : array-like
        Atomic numbers (e.g., [1, 8] for H2O).
    A : array-like
        Atomic masses in g/mol (same length as Z).
    w : array-like
        Weight fractions (same length as Z). If not normalized, set normalize_w=True.
    n : float
        Exponent (e.g., 3.62 or 1.86).
    normalize_w : bool
        If True, w is normalized to sum to 1.
    eps : float
        Small number to avoid division-by-zero.

    Returns
    -------
    float
        z_n value.
    """
    Z = np.asarray(Z, dtype=float)
    A = np.asarray(A, dtype=float)
    w = np.asarray(w, dtype=float)

    if not (Z.shape == A.shape == w.shape):
        raise ValueError("Z, A, w must have the same shape.")

    if normalize_w:
        s = w.sum()
        if s <= 0:
            raise ValueError("Sum of weight fractions must be positive.")
        w = w / s

    # weights' = w_i * (Z_i / A_i)
    w_prime = w * (Z / A)
    den = w_prime.sum()
    if abs(den) < eps:
        raise ZeroDivisionError("Denominator is too small in Eq.(5).")

    num = np.sum(w_prime * (Z ** n))
    return (num / den) ** (1.0 / n)

def zn_from_composition(Z, A, w):
    """組成 (Z, A, w) から z_3.62, z_1.86 を計算して返す"""
    z_3_62 = compute_z_n(Z, A, w, n=3.62)
    z_1_86 = compute_z_n(Z, A, w, n=1.86)
    return float(z_3_62), float(z_1_86)

# --- Physics-based CT値予測 ---
def physics_ct_loss(y_meas_ct, y_pred_rho, z_3_62, z_1_86):
    # y_pred_rho: [batch, 1] 質量密度
    num = y_pred_rho * (kph * z_3_62 + kcoh * z_1_86 + kincoh)
    den = kph * z3_62_water + kcoh * z1_86_water + kincoh
    hu_pred = 1000.0 * (num / den - 1.0)
    return tf.reduce_mean(tf.square(hu_pred - y_meas_ct))

# wf: shape = (D, H, W, C) = (50, 512, 512, 15)
# 例）実際は読み込み済みの配列を使ってください
# wf = np.load("wf.npy")  # など

def compute_zn_volume(wf: np.ndarray, Z: np.ndarray, A: np.ndarray, n: float,
                      normalize_w: bool = False, eps: float = 1e-12,
                      dtype=np.float32) -> np.ndarray:
    """
    Eq.(5) をボクセルごとに適用して z_n を返す（shape=(D,H,W)）
      z_n = ( sum_i [ w_i*(Z_i/A_i)*Z_i^n ] / sum_i [ w_i*(Z_i/A_i) ] )^(1/n)

    Parameters
    ----------
    wf : np.ndarray
        重量分率の4次元配列 (D, H, W, C)。最後の軸Cが元素チャンネル。
    Z, A : np.ndarray
        原子番号と原子量（長さC）。dtypeはfloat推奨。
    n : float
        指数（例：3.62, 1.86）
    normalize_w : bool
        True の場合、各ボクセルで wf を軸=-1で正規化する（合計1にする）。
    eps : float
        0除算回避用の小さい数。
    dtype : type
        返り値のdtype（既定: float32）。

    Returns
    -------
    zn : np.ndarray
        z_n（shape = (D, H, W)）
    """
    if wf.ndim != 4:
        raise ValueError(f"wf must be 4D (D,H,W,C), got {wf.shape}")
    if Z.shape != A.shape or Z.shape[-1] != wf.shape[-1]:
        raise ValueError("Z, A, wf のチャンネル数が一致している必要があります。")

    wf = wf.astype(np.float32, copy=False)  # 計算安定化
    Z = Z.astype(np.float32, copy=False)
    A = A.astype(np.float32, copy=False)

    # 必要に応じて重量分率を各ボクセルで正規化
    if normalize_w:
        s = wf.sum(axis=-1, keepdims=True)  # (D,H,W,1)
        wf = np.divide(wf, s, out=np.zeros_like(wf), where=s > 0)

    ratio = Z / A                     # (C,)
    Z_pow = Z ** float(n)             # (C,)

    # 分母: sum_i wf * (Z/A)
    den = np.tensordot(wf, ratio, axes=([3], [0]))     # → (D,H,W)

    # 分子: sum_i wf * (Z/A) * Z^n
    num = np.tensordot(wf, ratio * Z_pow, axes=([3], [0]))  # → (D,H,W)

    # z_n
    zn = np.power(np.maximum(num / (den + eps), 0.0), 1.0 / float(n), dtype=np.float32)
    return zn.astype(dtype)

# ---- 使い方例 ----
# z_3.62 と z_1.86 をまとめて計算
# zn362 = compute_zn_volume(wf, Z_arr, A_arr, n=3.62, normalize_w=False)
# zn186 = compute_zn_volume(wf, Z_arr, A_arr, n=1.86, normalize_w=False)
# np.save("z362.npy", zn362)
# np.save("z186.npy", zn186)

# --- カスタム損失関数 ---
class HybridLoss(tf.keras.losses.Loss):
    def __init__(self, delta):
        super().__init__()
        self.delta = delta

    def call(self, y_true, y_pred):
        # y_true: [rho_true, ct_true, z_3.62, z_1.86]
        rho_true, ct_true, z362, z186 = tf.split(y_true, 4, axis=-1)
        loss_mass = tf.reduce_mean(tf.square(y_pred - rho_true))
        loss_phys = physics_ct_loss(ct_true, y_pred, z362, z186)
        return (1.0 - self.delta) * loss_mass + self.delta * loss_phys
