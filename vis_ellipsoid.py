import os
from opt import get_opts
import torch
from collections import defaultdict

# models
from models.nerf import (
    PosEmbedding,
    NeRF
)
from models.nerflet import Nerflet
from models.rendering_nerflet import (
    get_input_from_rays,
    get_nerflet_pred
)
from models.model_utils import quaternions_to_rotation_matrices

from datasets.sitcom3D import Sitcom3DDataset
from datasets.blender import BlenderDataset
from utils import load_ckpt
import numpy as np
from simple_3dviz import Mesh
from simple_3dviz.window import show
from simple_3dviz.utils import render
from simple_3dviz.behaviours.misc import LightToCamera
from simple_3dviz.behaviours.io import SaveFrames


@torch.no_grad()
def estimate_ellipsoid(model, embeddings, rays, ts, N_samples, use_disp, chunk):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)
    ellipsoid_keys = ['part_rotations', 'part_translations', 'part_scales']

    rays_ = rays[:chunk]
    ts_ = ts[:chunk]
    xyz, rays_d, z_vals = get_input_from_rays(rays_, N_samples, use_disp)
    pred = get_nerflet_pred(model, embeddings, xyz, rays_d, ts_)
    ellipsoid_pred = {}
    for k in ellipsoid_keys:
        ellipsoid_pred[k] = torch.clone(pred[k])

    for k, v in ellipsoid_pred.items():
        results[k] += [v.cpu()]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results


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
                                 img_wh=args.img_wh, split='test_train')

    embedding_xyz = PosEmbedding(args.N_emb_xyz - 1, args.N_emb_xyz)
    embedding_dir = PosEmbedding(args.N_emb_dir - 1, args.N_emb_dir)
    embeddings = {'xyz': embedding_xyz, 'dir': embedding_dir}

    embedding_a = torch.nn.Embedding(args.N_vocab, args.N_a).cuda()
    load_ckpt(embedding_a, args.ckpt_path, model_name='embedding_a')
    embeddings['a'] = embedding_a

    if args.encode_t:
        embedding_t = torch.nn.Embedding(args.N_vocab, args.N_tau).cuda()
        load_ckpt(embedding_t, args.ckpt_path, model_name='embedding_t')
        embeddings['t'] = embedding_t

    nerflet = Nerflet(N_emb_xyz=args.N_emb_xyz, N_emb_dir=args.N_emb_dir,
                      encode_t=args.encode_t, predict_label=args.predict_label,
                      num_classes=args.num_classes, M=args.num_parts).cuda()

    load_ckpt(nerflet, args.ckpt_path, model_name='nerflet')

    imgs, psnrs = [], []
    run_name = args.ckpt_path.split("/")[-4]
    dir_name = f'{args.environment_dir}/rendering/{run_name}'
    os.makedirs(dir_name, exist_ok=True)

    label_colors = np.random.rand(args.num_classes, 3)
    part_colors = np.random.rand(16, 3)

    iou_combined = []
    iou_static = []

    sample = dataset[0]
    rays = sample['rays']
    ts = sample['ts']
    results = estimate_ellipsoid(nerflet, embeddings, rays.cuda(), ts.cuda(),
                                 args.N_samples, args.use_disp, args.chunk)

    part_colors = np.random.rand(args.num_parts, 3)
    rotations = quaternions_to_rotation_matrices(results['part_rotations']).numpy()
    rotations = np.linalg.inv(rotations)
    translations = results['part_translations'].numpy()
    scales = results['part_scales'].numpy()
    eps = np.ones((scales.shape[0], 2))

    m = Mesh.from_superquadrics(alpha=scales, epsilon=eps, translation=translations, rotation=rotations, colors=part_colors)

    cam_pos = sample['c2w'][:, 3].numpy()
    cam_target = np.array([0, 0, 0])
    light = (-60, -160, 120)
    show(m, camera_position=cam_pos, camera_target=cam_target, light=light)
    # render(m,
    #        behaviours=[
    #            CameraTrajectory(
    #                BackAndForth(Lines([-60, -160, 120], [-60, -80, 120])),
    #                speed=0.005
    #            )
    #            LightToCamera(),
    #            SaveFrames("/tmp/frame_{:03d}.png", every_n=5)
    #        ],
    #        n_frames=512,
    #        camera_position=(-60., -160, 120), camera_target=(0., 0, 40),
    #        light=(-60, -160, 120)
    # )