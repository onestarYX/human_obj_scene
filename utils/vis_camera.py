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
    xyz, rays_d, z_vals = get_input_from_rays(rays, N_samples, use_disp, perturb=0)
    return xyz

if __name__ == '__main__':
    hparams = get_opts()
    if hparams.dataset_name == 'sitcom3D':
        kwargs = {'environment_dir': hparams.environment_dir,
                  'near_far_version': hparams.near_far_version}
        # kwargs['img_downscale'] = self.hparams.img_downscale
        kwargs['val_num'] = hparams.num_gpus
        kwargs['use_cache'] = hparams.use_cache
        kwargs['num_limit'] = hparams.num_limit
        dataset = Sitcom3DDataset(split='test_train',
                                  img_downscale=hparams.img_downscale,
                                  near=hparams.near, **kwargs)
    else:
        raise NotImplementedError

    os.makedirs(hparams.out_dir, exist_ok=True)

    ray_colors = np.random.rand(10, 3)

    geo = []
    bbox_min = torch.tensor([100, 100, 100])
    bbox_max = torch.tensor([-100, -100, -100])
    for i, sample in enumerate(dataset):
        rays = sample['rays']
        xyz = get_input_xyz(rays.cuda(), hparams.N_samples, hparams.use_disp) # (H * W, N_sample, 3)
        xyz = xyz[::4, ::2]
        xyz = xyz.reshape(-1, 3).cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        colors = np.tile(ray_colors[i], (len(xyz), 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        geo.append(pcd)

    o3d.visualization.draw_geometries(geo)