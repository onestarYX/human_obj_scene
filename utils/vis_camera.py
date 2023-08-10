import os
from opt import get_opts
import torch
from datasets.sitcom3D import Sitcom3DDataset
from datasets.blender import BlenderDataset
import numpy as np
from simple_3dviz import Mesh
from simple_3dviz.window import show
import open3d as o3d

# models
from models.rendering_nerflet import (
    get_input_from_rays,
)

@torch.no_grad()
def get_input_xyz(rays, N_samples, use_disp):
    """Do batched inference on rays using chunk."""
    xyz, rays_d, z_vals = get_input_from_rays(rays, N_samples, use_disp)
    return xyz


def compute_iou(pred, gt, num_cls):
    iou = []
    for cls_idx in range(num_cls):
        denom = np.logical_or((pred == cls_idx), (gt == cls_idx)).astype(int).sum()
        if denom == 0:
            iou.append(1)
        else:
            numer = np.logical_and((pred == cls_idx), (gt == cls_idx)).astype(int).sum()
            iou.append(numer / denom)
    return np.mean(iou)


if __name__ == '__main__':
    args = get_opts()
    if args.dataset_name == 'sitcom3D':
        kwargs = {'environment_dir': args.environment_dir,
                  'near_far_version': args.near_far_version}
        # kwargs['img_downscale'] = args.img_downscale
        kwargs['val_num'] = 5
        kwargs['use_cache'] = args.use_cache
        dataset = Sitcom3DDataset(split='test_train', img_downscale=args.img_downscale_val, **kwargs)
    elif args.dataset_name == 'blender':
        kwargs = {}
        dataset = BlenderDataset(root_dir=args.environment_dir,
                                 img_wh=args.img_wh, split='test_train', test_imgs=args.test_imgs)

    os.makedirs(args.out_dir, exist_ok=True)

    ray_colors = np.random.rand(10, 3)

    geo = []
    for i, sample in enumerate(dataset):
        rays = sample['rays']
        xyz = get_input_xyz(rays.cuda(), args.N_samples, args.use_disp) # (H * W, N_sample, 3)
        xyz = xyz[::4, ::2]
        xyz = xyz.reshape(-1, 3).cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        colors = np.tile(ray_colors[i], (len(xyz), 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        geo.append(pcd)

    o3d.visualization.draw_geometries(geo)