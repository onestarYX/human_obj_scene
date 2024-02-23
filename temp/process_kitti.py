from pathlib import Path
from PIL import Image
import numpy as np

if __name__ == '__main__':
    in_dir = Path('data/kitti360/drive_0000_250_300')
    out_dir = in_dir.with_name(in_dir.name + '_cars')
    in_rgb_dir = in_dir / 'rgb'
    in_sem_dir = in_dir / 'semantic'
    out_rgb_dir = out_dir / in_rgb_dir.name
    out_rgb_dir.mkdir(parents=True, exist_ok=True)
    out_sem_dir = out_dir / in_sem_dir.name
    out_sem_dir.mkdir(parents=True, exist_ok=True)

    target_sem_idx = 26
    for sem_file in in_sem_dir.iterdir():
        sem_img = np.array(Image.open(sem_file))
        mask = sem_img == target_sem_idx
        new_sem_img = np.zeros_like(sem_img)
        new_sem_img[mask] = target_sem_idx
        new_sem_img = Image.fromarray(new_sem_img)
        new_sem_img.save(out_sem_dir / sem_file.name)

        rgb_file = in_rgb_dir / sem_file.name
        rgb_img = np.array(Image.open(rgb_file))
        new_rgb_img = np.zeros_like(rgb_img)
        new_rgb_img[mask] = rgb_img[mask]
        new_rgb_img = Image.fromarray(new_rgb_img)
        new_rgb_img.save(out_rgb_dir / rgb_file.name)