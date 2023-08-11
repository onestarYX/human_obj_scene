import os
from pathlib import Path
import json
import numpy as np
import shutil

if __name__ == "__main__":
    data_dir = Path('data/nerf_synthetic/chair')
    out_cam_dir = Path('data/nerf_synthetic/chair/cameras')
    out_cam_dir.mkdir(exist_ok=True)
    out_img_dir = Path('data/nerf_synthetic/chair/images')
    out_img_dir.mkdir(exist_ok=True)

    splits = ['train', 'val', 'test']
    file_count = 0
    for split in splits:
        json_file = data_dir / f"transforms_{split}.json"
        with open(json_file, 'r') as f:
            meta = json.load(f)
        img_dir = data_dir / f"{split}"
        w = 800
        h = 800
        focal = 0.5 * 800 / np.tan(0.5 * meta['camera_angle_x'])
        K = np.eye(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = w / 2
        K[1, 2] = h / 2

        for frame_meta in meta['frames']:
            frame_idx = int(frame_meta['file_path'].split('_')[-1])
            transform_matrix = np.array(frame_meta['transform_matrix'])
            R = transform_matrix[:3, :3]
            t = transform_matrix[:3, 3]
            cur_file_idx = file_count + frame_idx
            out_cam_path = out_cam_dir / f"{cur_file_idx:05d}.npz"
            np.savez(out_cam_path, K=K, R=R, t=t)

            img_path = img_dir / (frame_meta['file_path'].split('/')[-1] + '.png')
            out_img_path = out_img_dir / f"{cur_file_idx:05d}.png"
            shutil.copyfile(img_path, out_img_path)

        file_count += len(meta['frames'])