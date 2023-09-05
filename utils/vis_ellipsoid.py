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
import argparse
import csv
import json
from omegaconf import OmegaConf
from pathlib import Path

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
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_path', type=str)
    parser.add_argument('--output_dir', type=str, default='results/ellipsoids')
    parser.add_argument('--use_ckpt', type=str)
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_path)
    config = dict()
    if (ckpt_dir / 'meta_tags.csv').exists():
        config_path = ckpt_dir / 'meta_tags.csv'
        with open(config_path, newline='') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                dict[row['key']] = row['value']
    else:
        config_path = ckpt_dir / 'config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)

    config = OmegaConf.create(config)

    if config.dataset_name == 'sitcom3D':
        kwargs = {'environment_dir': config.environment_dir,
                  'near_far_version': config.near_far_version}
        # kwargs['img_downscale'] = args.img_downscale
        kwargs['val_num'] = 5
        kwargs['use_cache'] = config.use_cache
        dataset = Sitcom3DDataset(split='test_train', img_downscale=config.img_downscale_val, **kwargs)
    elif config.dataset_name == 'blender':
        kwargs = {}
        dataset = BlenderDataset(root_dir=config.environment_dir,
                                 img_wh=config.img_wh, split='test_train')

    embedding_xyz = PosEmbedding(config.N_emb_xyz - 1, config.N_emb_xyz)
    embedding_dir = PosEmbedding(config.N_emb_dir - 1, config.N_emb_dir)
    embeddings = {'xyz': embedding_xyz, 'dir': embedding_dir}

    if config.encode_a:
        embedding_a = torch.nn.Embedding(config.N_vocab, config.N_a).cuda()
        load_ckpt(embedding_a, config.ckpt_path, model_name='embedding_a')
        embeddings['a'] = embedding_a

    if config.encode_t:
        embedding_t = torch.nn.Embedding(config.N_vocab, config.N_tau).cuda()
        load_ckpt(embedding_t, config.ckpt_path, model_name='embedding_t')
        embeddings['t'] = embedding_t

    nerflet = Nerflet(N_emb_xyz=config.N_emb_xyz, N_emb_dir=config.N_emb_dir,
                      encode_a=config.encode_a, encode_t=config.encode_t, predict_label=config.predict_label,
                      num_classes=config.num_classes, M=config.num_parts).cuda()

    load_ckpt(nerflet, config.ckpt_path, model_name='nerflet')


    label_colors = np.random.rand(config.num_classes, 3)
    part_colors = np.random.rand(16, 3)

    sample = dataset[0]
    rays = sample['rays']
    ts = sample['ts']
    results = estimate_ellipsoid(nerflet, embeddings, rays.cuda(), ts.cuda(),
                                 config.N_samples, config.use_disp, config.chunk)

    part_colors = np.random.rand(config.num_parts, 3)
    rotations = quaternions_to_rotation_matrices(results['part_rotations']).numpy()
    rotations = np.linalg.inv(rotations)
    translations = results['part_translations'].numpy()
    scales = results['part_scales'].numpy()
    eps = np.ones((scales.shape[0], 2))

    m = Mesh.from_superquadrics(alpha=scales, epsilon=eps, translation=translations, rotation=rotations, colors=part_colors)

    cam_pos = sample['c2w'][:, 3].numpy()
    # cam_rot = sample['c2w'][:, :3].numpy()
    # init_target = np.array([1, 0, 0])[..., np.newaxis]
    # cam_target = np.squeeze(cam_rot @ init_target) + cam_pos
    cam_target = np.array([0, 0, 0])
    light = (-60, -160, 120)
    show(m, camera_position=cam_pos, camera_target=cam_target, light=light)