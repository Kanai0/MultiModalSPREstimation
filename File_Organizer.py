# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import pydicom




def load_dicom_series(folder_path):
    dicom_files = [pydicom.dcmread(os.path.join(folder_path, f)) 
                   for f in os.listdir(folder_path) if f.endswith(".dcm")]

    # スライス位置でソート（ImagePositionPatientがない場合はInstanceNumberで）
    try:
        dicom_files.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except:
        dicom_files.sort(key=lambda x: int(x.InstanceNumber))

    # 画像サイズと枚数を取得して空の配列を作成
    image_shape = dicom_files[0].pixel_array.shape
    volume_shape = (len(dicom_files), image_shape[0], image_shape[1])
    volume = np.zeros(volume_shape, dtype=dicom_files[0].pixel_array.dtype)

    for i, dcm in enumerate(dicom_files):
        volume[i, :, :] = dcm.pixel_array

    return volume

### deprecated ###
# def load_npz_as_array(folder_path):

#     npz_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npz')])
#     if not npz_files:
#         print("npzファイルが見つかりません。")
#         return

#     array_list = []
#     for fname in npz_files:
#         path = os.path.join(folder_path, fname)
#         data = np.load(path)
#         array_list.append(data['arr_0'])
#     return np.stack(array_list, axis=0)  # shape: (スライス数, H, W)


def natural_key(s: str):
    """Windowsの '1, 2, 10' を自然順に並べるためのキー."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def scan_npz_keys(folder_path):
    """
    フォルダ内の全 .npz を走査して {key: 出現ファイル数} を返す。
    また各ファイルごとのキー一覧も返す。
    """
    npz_files = sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith('.npz')],
        key=natural_key
    )
    if not npz_files:
        raise FileNotFoundError("npzファイルが見つかりません。")

    key_counts = {}
    file_keys = {}

    for fname in npz_files:
        path = os.path.join(folder_path, fname)
        with np.load(path) as data:
            keys = list(data.files)   # data.files にキー一覧が入っている
            file_keys[fname] = keys
            for k in keys:
                key_counts[k] = key_counts.get(k, 0) + 1

    return npz_files, key_counts, file_keys


def choose_key_interactively(key_counts, total_files):
    """
    キー候補を表示して、ユーザーにひとつ選んでもらう。
    すべてのファイルに共通するキーが1つだけなら自動選択。
    """
    common_keys = [k for k, c in key_counts.items() if c == total_files]

    if len(common_keys) == 1:
        chosen = common_keys[0]
        print(f"すべてのファイルに共通するキーが見つかったため自動選択: '{chosen}'")
        return chosen

    print("=== 読み込み可能なキー候補（出現ファイル数）===")
    sorted_items = sorted(key_counts.items(), key=lambda x: (-x[1], x[0].lower()))
    for i, (k, c) in enumerate(sorted_items, 1):
        mark = " (全ファイル共通)" if c == total_files else ""
        print(f"[{i}] {k} : {c}/{total_files}{mark}")

    while True:
        sel = input("使用するキー番号を入力してください: ").strip()
        if sel.isdigit():
            idx = int(sel)
            if 1 <= idx <= len(sorted_items):
                return sorted_items[idx - 1][0]
        print("無効な入力です。もう一度番号を入力してください。")


def load_npz_as_array(folder_path, selected_key=None, skip_missing=True):

    npz_files, key_counts, file_keys = scan_npz_keys(folder_path)
    total_files = len(npz_files)

    if selected_key is None:
        selected_key = choose_key_interactively(key_counts, total_files)
    else:
        # スクリプト上で事前指定する場合のバリデーション
        if selected_key not in key_counts:
            raise KeyError(f"指定キー '{selected_key}' はいずれの .npz にも存在しません。")
        
    
    npz_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npz')])
    if not npz_files:
        print("npzファイルが見つかりません。")
        return

    array_list = []
    for fname in npz_files:
        path = os.path.join(folder_path, fname)
        data = np.load(path)
        array_list.append(data[selected_key])

    return np.stack(array_list, axis=0)  # shape: (スライス数, H, W)



dir_path = r'C:/Users/Kanai/Synology/TWMU/0_張先生_共同研究_MRI阻止能/FromIzo/weight_fraction'
output_path = 'wf.npy'
# DICOM版（必要に応じて）
# arr = load_dicom_series(dir_path)

# NPZ版：selected_key=None なら実行時にキー選択を促す
arr = load_npz_as_array(dir_path, selected_key=None, skip_missing=True)

# 保存
np.save(output_path, arr)
print("保存しました: " + output_path)


