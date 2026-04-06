import torch
import math
from PIL import Image
import os
import numpy as np
import io
import matplotlib
matplotlib.use("Agg")  # use a headless backend (no tkinter)
import matplotlib.pyplot as plt

def visualize_and_save(
    sem,
    depth,
    filename,
    margin=10,
    base_dir="save_renders/save_images",
):
    # --- directories ---
    sem_dir = os.path.join(base_dir, "sem")
    depth_dir = os.path.join(base_dir, "depth")
    os.makedirs(sem_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    # --- semantic preprocessing ---
    try:
        if sem.ndim >= 3 and sem.shape[1] == 17:
            sem = convert_zero_empty_index(sem)
    except Exception:
        pass
    if sem.ndim == 4 and sem.shape[1] == 1:
        sem = sem[:, 0]
    sem = sem.clone().detach().cpu().long()
    num_images = sem.shape[0]

    # --- depth preprocessing ---
    if depth.ndim == 4 and depth.shape[1] == 1:
        depth = depth[:, 0]
    depth = depth.clone().detach().cpu().float()

    # fixed range of depth
    vmin, vmax = 0.1, 40.0

    # --- helpers ---
    def colorize_sem(img_np):
        color_img = np.zeros((*img_np.shape, 3), dtype=np.uint8)
        for i in range(len(my_colors)):
            color_img[img_np == i] = my_colors[i]
        return Image.fromarray(color_img)

    def depth_to_pil(arr):
        if not np.isfinite(arr).all():
            finite = np.isfinite(arr)
            fill = np.nanmin(arr[finite]) if finite.any() else 0.0
            arr = np.where(np.isfinite(arr), arr, fill)
        cmap = plt.get_cmap("Spectral")
        normed = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
        rgba = cmap(normed)
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
        return Image.fromarray(rgb)

    def grid_shape(n):
        if n == 6:
            return 2, 3
        elif n % 6 == 0:
            return n // 6, 6
        else:
            cols = math.ceil(math.sqrt(n))
            rows = math.ceil(n / cols)
            return rows, cols

    def make_collage(pil_list, margin=10, bg=(0, 0, 0)):
        img_w, img_h = pil_list[0].size
        rows, cols = grid_shape(len(pil_list))
        W = cols * img_w + (cols + 1) * margin
        H = rows * img_h + (rows + 1) * margin
        canvas = Image.new("RGB", (W, H), color=bg)
        for idx, img in enumerate(pil_list):
            c = idx % cols
            r = idx // cols
            x = margin + c * (img_w + margin)
            y = margin + r * (img_h + margin)
            canvas.paste(img, (x, y))
        return canvas

    def add_single_colorbar_to_collage(collage_img):
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        ax.axis("off")
        ax.imshow(collage_img)
        # Add shared colorbar
        sm = plt.cm.ScalarMappable(cmap="magma", norm=plt.Normalize(vmin=vmin, vmax=vmax))
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Depth (m)")
        buf = io.BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    # --- build PIL images ---
    sem_pils = [colorize_sem(sem[i].numpy()) for i in range(num_images)]
    depth_pils = [depth_to_pil(depth[i].numpy()) for i in range(num_images)]

    if num_images == 1:
        sem_pils[0].save(os.path.join(sem_dir, f"{filename}.png"))
        depth_with_cbar = add_single_colorbar_to_collage(depth_pils[0])
        depth_with_cbar.save(os.path.join(depth_dir, f"{filename}.png"))
        return

    # collages
    sem_collage = make_collage(sem_pils, margin=margin, bg=(0, 0, 0))
    depth_collage = make_collage(depth_pils, margin=margin, bg=(0, 0, 0))
    # depth_collage_with_cbar = add_single_colorbar_to_collage(depth_collage)

    # save
    sem_collage.save(os.path.join(sem_dir, f"{filename}.png"))
    depth_collage.save(os.path.join(depth_dir, f"{filename}.png"))

def convert_zero_empty_index(pred):
    zero_mask = (pred == 0).all(dim=1)
    pred = pred.argmax(dim=1)
    pred[zero_mask] = 0

    return pred

my_colors = np.array(
    [
        [255, 255, 255],        # Black background for class 0
        [255, 120, 50],   # barrier 1
        [255, 192, 203],  # bicycle 2
        [255, 255, 0],    # bus 3
        [0, 150, 245],    # car 4
        [0, 255, 255],    # construction_vehicle 5
        [200, 180, 0],    # motorcycle 6
        [255, 0, 0],      # pedestrian 7
        [255, 240, 150],  # traffic_cone 8
        [135, 60, 0],     # trailer 9 
        [160, 32, 240],   # truck 10
        [255, 0, 255],    # drivable_surface 11
        [139, 137, 137],  # other_flat 12
        [75, 0, 75],      # sidewalk 13
        [150, 240, 80],   # terrain 14
        [230, 230, 250],  # manmade 15
        [0, 175, 0],      # vegetation 16 
        [0, 0, 0],  # empty class 17
    ]
).astype(np.uint8)
