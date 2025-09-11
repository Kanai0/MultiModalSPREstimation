# %%
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 19:07:29 2025

@author: Kanai
"""

import numpy as np
import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class SliceViewer3D:
    def __init__(self, master, volume, title="3D Volume Slice Viewer"):
        self.master = master
        self.volume = volume
        self.shape = volume.shape
        
        self.master.title(title)

        self.max_val = float(np.max(volume))
        self.min_val = float(np.min(volume))

        self.axial_idx = self.shape[0] // 2
        self.coronal_idx = self.shape[1] // 2
        self.sagittal_idx = self.shape[2] // 2

        self.vmin = tk.DoubleVar(value=self.min_val)
        self.vmax = tk.DoubleVar(value=self.max_val)

        # Colorbar handles (one per view)
        self.cb1 = None
        self.cb2 = None
        self.cb3 = None

        # Save settings
        self.save_dpi = 300

        self.setup_ui()
        self.update_images()

    def setup_ui(self):
        self.fig = Figure(figsize=(8, 8))
        self.ax1 = self.fig.add_subplot(221)
        self.ax2 = self.fig.add_subplot(222)
        self.ax3 = self.fig.add_subplot(223)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=4)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)

        # ステータスラベル追加（右下に）
        self.status_label = tk.Label(self.master, text="Pixel Value: ", anchor='w')
        self.status_label.grid(row=5, column=0, columnspan=4, sticky='w')
        
        self.setup_controls()

    def setup_controls(self):
        # Axial (Left Top)
        tk.Label(self.master, text="Axial Slice").grid(row=1, column=0)
        self.axial_slider = tk.Scale(self.master, from_=0, to=self.shape[0]-1,
                                     orient=tk.HORIZONTAL, command=self.on_axial_change)
        self.axial_slider.set(self.axial_idx)
        self.axial_slider.grid(row=2, column=0)
        self.axial_entry = tk.Entry(self.master, width=5)
        self.axial_entry.insert(0, str(self.axial_idx))
        self.axial_entry.grid(row=3, column=0)
        self.axial_entry.bind('<Return>', self.on_axial_entry)

        # Coronal (Right Top)
        tk.Label(self.master, text="Coronal Slice").grid(row=1, column=1)
        self.coronal_slider = tk.Scale(self.master, from_=0, to=self.shape[1]-1,
                                       orient=tk.HORIZONTAL, command=self.on_coronal_change)
        self.coronal_slider.set(self.coronal_idx)
        self.coronal_slider.grid(row=2, column=1)
        self.coronal_entry = tk.Entry(self.master, width=5)
        self.coronal_entry.insert(0, str(self.coronal_idx))
        self.coronal_entry.grid(row=3, column=1)
        self.coronal_entry.bind('<Return>', self.on_coronal_entry)

        # Sagittal (Left Bottom)
        tk.Label(self.master, text="Sagittal Slice").grid(row=1, column=2)
        self.sagittal_slider = tk.Scale(self.master, from_=0, to=self.shape[2]-1,
                                        orient=tk.HORIZONTAL, command=self.on_sagittal_change)
        self.sagittal_slider.set(self.sagittal_idx)
        self.sagittal_slider.grid(row=2, column=2)
        self.sagittal_entry = tk.Entry(self.master, width=5)
        self.sagittal_entry.insert(0, str(self.sagittal_idx))
        self.sagittal_entry.grid(row=3, column=2)
        self.sagittal_entry.bind('<Return>', self.on_sagittal_entry)

        # Window Level/Width (Bottom Right)
        tk.Label(self.master, text="Min Value").grid(row=1, column=3)
        self.vmin_entry = tk.Entry(self.master, textvariable=self.vmin, width=8)
        self.vmin_entry.grid(row=2, column=3)
#        self.vmin_entry.insert(tk.END, str(self.min_val))
        self.vmin_entry.bind('<Return>', lambda e: self.update_images())

        tk.Label(self.master, text="Max Value").grid(row=3, column=3)
        self.vmax_entry = tk.Entry(self.master, textvariable=self.vmax, width=8)
        self.vmax_entry.grid(row=4, column=3)
#        self.vmax_entry.insert(tk.END, str(self.max_val))
        self.vmax_entry.bind('<Return>', lambda e: self.update_images())

        # Save PNG button (bottom left)
        self.save_btn = tk.Button(self.master, text="Save PNG", command=self.save_png)
        self.save_btn.grid(row=4, column=0, sticky='w')

    def update_images(self):
        try:
            vmin = float(self.vmin_entry.get())
            vmax = float(self.vmax_entry.get())
        except ValueError:
            print("数値を入力してください")
            return
    
        # Clear previous content and colorbars
        self.ax1.clear()
        if self.cb1 is not None:
            try:
                self.cb1.remove()
            except Exception:
                pass
            self.cb1 = None

        im1 = self.ax1.imshow(self.volume[self.axial_idx, :, :], cmap='gray', vmin=vmin, vmax=vmax)
        self.ax1.set_title(f'Axial: {self.axial_idx}')
        self.ax1.axis('off')
        self.cb1 = self.fig.colorbar(im1, ax=self.ax1, fraction=0.046, pad=0.04)
        # Show only min/max on the colorbar
        if vmin == vmax:
            self.cb1.set_ticks([vmin])
            self.cb1.set_ticklabels([f"{vmin:g}"])
        else:
            self.cb1.set_ticks([vmin, vmax])
            self.cb1.set_ticklabels([f"{vmin:g}", f"{vmax:g}"])
    
        self.ax2.clear()
        if self.cb2 is not None:
            try:
                self.cb2.remove()
            except Exception:
                pass
            self.cb2 = None

        im2 = self.ax2.imshow(self.volume[:, self.coronal_idx, :], cmap='gray', vmin=vmin, vmax=vmax)
        self.ax2.set_title(f'Coronal: {self.coronal_idx}')
        self.ax2.axis('off')
        self.cb2 = self.fig.colorbar(im2, ax=self.ax2, fraction=0.046, pad=0.04)
        if vmin == vmax:
            self.cb2.set_ticks([vmin])
            self.cb2.set_ticklabels([f"{vmin:g}"])
        else:
            self.cb2.set_ticks([vmin, vmax])
            self.cb2.set_ticklabels([f"{vmin:g}", f"{vmax:g}"])
    
        self.ax3.clear()
        if self.cb3 is not None:
            try:
                self.cb3.remove()
            except Exception:
                pass
            self.cb3 = None

        im3 = self.ax3.imshow(self.volume[:, :, self.sagittal_idx], cmap='gray', vmin=vmin, vmax=vmax)
        self.ax3.set_title(f'Sagittal: {self.sagittal_idx}')
        self.ax3.axis('off')
        self.cb3 = self.fig.colorbar(im3, ax=self.ax3, fraction=0.046, pad=0.04)
        if vmin == vmax:
            self.cb3.set_ticks([vmin])
            self.cb3.set_ticklabels([f"{vmin:g}"])
        else:
            self.cb3.set_ticks([vmin, vmax])
            self.cb3.set_ticklabels([f"{vmin:g}", f"{vmax:g}"])

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def save_png(self):
        # Ask path and save the entire viewer figure as PNG
        path = filedialog.asksaveasfilename(
            title="Save viewer as PNG",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png")]
        )
        if not path:
            return
        try:
            self.fig.tight_layout()
            self.fig.savefig(path, dpi=self.save_dpi, bbox_inches='tight')
            self.status_label.config(text=f"Saved: {path}")
        except Exception as e:
            self.status_label.config(text=f"Save failed: {e}")
        
    def on_axial_change(self, val):
        self.axial_idx = int(val)
        self.axial_entry.delete(0, tk.END)
        self.axial_entry.insert(0, val)
        self.update_images()

    def on_coronal_change(self, val):
        self.coronal_idx = int(val)
        self.coronal_entry.delete(0, tk.END)
        self.coronal_entry.insert(0, val)
        self.update_images()

    def on_sagittal_change(self, val):
        self.sagittal_idx = int(val)
        self.sagittal_entry.delete(0, tk.END)
        self.sagittal_entry.insert(0, val)
        self.update_images()

    def on_axial_entry(self, event):
        val = int(self.axial_entry.get())
        self.axial_slider.set(val)

    def on_coronal_entry(self, event):
        val = int(self.coronal_entry.get())
        self.coronal_slider.set(val)

    def on_sagittal_entry(self, event):
        val = int(self.sagittal_entry.get())
        self.sagittal_slider.set(val)

    def on_scroll(self, event):
        delta = int(event.step)  # 上スクロール：1、下スクロール：-1
    
        if event.inaxes == self.ax1:
            self.axial_idx = np.clip(self.axial_idx + delta, 0, self.shape[0] - 1)
            self.axial_slider.set(self.axial_idx)
        elif event.inaxes == self.ax2:
            self.coronal_idx = np.clip(self.coronal_idx + delta, 0, self.shape[1] - 1)
            self.coronal_slider.set(self.coronal_idx)
        elif event.inaxes == self.ax3:
            self.sagittal_idx = np.clip(self.sagittal_idx + delta, 0, self.shape[2] - 1)
            self.sagittal_slider.set(self.sagittal_idx)
    
        self.update_images()
    
    def on_mouse_move(self, event):
        if event.inaxes is None:
            self.status_label.config(text="Pixel Value: ")
            return
    
        try:
            x, y = int(event.xdata), int(event.ydata)
        except (TypeError, ValueError):
            self.status_label.config(text="Pixel Value: ")
            return
    
        if event.inaxes == self.ax1:  # Axial
            if 0 <= x < self.shape[2] and 0 <= y < self.shape[1]:
                val = self.volume[self.axial_idx, y, x]
                self.status_label.config(text=f"Axial [{x}, {y}] = {val}")
        elif event.inaxes == self.ax2:  # Coronal
            if 0 <= x < self.shape[2] and 0 <= y < self.shape[0]:
                val = self.volume[y, self.coronal_idx, x]
                self.status_label.config(text=f"Coronal [{x}, {y}] = {val}")
        elif event.inaxes == self.ax3:  # Sagittal
            if 0 <= x < self.shape[1] and 0 <= y < self.shape[0]:
                val = self.volume[y, x, self.sagittal_idx]
                self.status_label.config(text=f"Sagittal [{x}, {y}] = {val}")


def load_npy_file():
    tk.Tk().withdraw()
    file_path = filedialog.askopenfilename(title="3次元配列（.npy）を選択", filetypes=[("NumPy Files", "*.npy")])
    if file_path:
        return np.load(file_path)
    return None

def launch_viewer(volume, title= "3D Volume Slice Viewer"):
    root = tk.Tk()
    app = SliceViewer3D(root, volume, title=title)
    root.mainloop()
 


def main():
    volume = load_npy_file()
    if volume is not None and volume.ndim == 3:
        root = tk.Tk()
        app = SliceViewer3D(root, volume)
        root.mainloop()
    else:
        print("有効な3次元NumPy配列が選択されませんでした。")


if __name__ == "__main__":
    main()
