import os
import torch
from collections import defaultdict

# models
from models.nerf import PosEmbedding
from models.nerflet import Nerflet
from models.rendering_nerflet import (
    get_input_from_rays,
    get_nerflet_pred
)
from models.model_utils import quaternions_to_rotation_matrices

from datasets.sitcom3D import Sitcom3DDataset
from datasets.blender import BlenderDataset
from datasets.replica import ReplicaDataset
from datasets.front import ThreeDFrontDataset
from utils import load_ckpt
import numpy as np
from simple_3dviz import Mesh
from simple_3dviz.window import show
from simple_3dviz.utils import render
from simple_3dviz.behaviours.io import SaveFrames
import argparse
import csv
import json
from omegaconf import OmegaConf
from pathlib import Path
import imageio

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
    parser.add_argument('--exp_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results/ellipsoids')
    parser.add_argument('--use_ckpt', type=str)
    parser.add_argument('--render_every', type=int, default=5)
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    output_dir = exp_dir / args.output_dir

    config_path = exp_dir / 'config.json'
    with open(config_path, 'r') as f:
        file_config = json.load(f)
    file_config = OmegaConf.create(file_config)
    cli_config = OmegaConf.from_dotlist(args.opts)
    config = OmegaConf.merge(file_config, cli_config)

    if config.dataset_name == 'sitcom3D':
        kwargs = {'environment_dir': config.environment_dir, 'near_far_version': config.near_far_version, 'val_num': 5,
                  'use_cache': config.use_cache}
        # kwargs['img_downscale'] = args.img_downscale
        dataset = Sitcom3DDataset(split=args.split, img_downscale=config.img_downscale_val, **kwargs)
    elif config.dataset_name == 'blender':
        kwargs = {}
        dataset = BlenderDataset(root_dir=config.environment_dir,
                                 img_wh=config.img_wh, split=args.split)
    elif config.dataset_name == 'replica':
        dataset = ReplicaDataset(root_dir=config.environment_dir,
                                 img_downscale=config.img_downscale, split=args.split)
    elif config.dataset_name == '3dfront':
        dataset = ThreeDFrontDataset(root_dir=config.environment_dir,
                                     img_downscale=config.img_downscale, split=args.split)

    embedding_xyz = PosEmbedding(config.N_emb_xyz - 1, config.N_emb_xyz)
    embedding_dir = PosEmbedding(config.N_emb_dir - 1, config.N_emb_dir)
    embeddings = {'xyz': embedding_xyz, 'dir': embedding_dir}

    np.random.seed(19)
    part_colors = np.random.rand(config.num_parts, 3)

    ckpt_paths = []
    if args.use_ckpt:
        ckpt_paths.append(Path(args.use_ckpt))
    else:
        ckpt_dir = exp_dir / 'ckpts'
        for ckpt_path in ckpt_dir.iterdir():
            ckpt_paths.append(ckpt_path)
        def get_step_from_path(path):
            step = path.stem.split('=')[-1]
            return int(step)
        ckpt_paths.sort(key=get_step_from_path)
    ckpt_paths = ckpt_paths[::args.render_every]

    original_render_dir = output_dir / 'original'; original_render_dir.mkdir(parents=True, exist_ok=True)
    zoom_out_render_dir = output_dir / 'zoom_out'; zoom_out_render_dir.mkdir(parents=True, exist_ok=True)
    for ckpt_path in ckpt_paths:
        print(f"Working on {ckpt_path}")
        ckpt_name = ckpt_path.stem
        original_img_path = original_render_dir / f"{ckpt_name}.png"
        zoom_out_img_path = zoom_out_render_dir / f"{ckpt_name}.png"
        if not args.overwrite and original_img_path.exists() and zoom_out_img_path.exists():
            continue

        if config.encode_a:
            embedding_a = torch.nn.Embedding(config.N_vocab, config.N_a).cuda()
            load_ckpt(embedding_a, ckpt_path, model_name='embedding_a')
            embeddings['a'] = embedding_a

        if config.encode_t:
            embedding_t = torch.nn.Embedding(config.N_vocab, config.N_tau).cuda()
            load_ckpt(embedding_t, ckpt_path, model_name='embedding_t')
            embeddings['t'] = embedding_t

        nerflet = Nerflet(N_emb_xyz=config.N_emb_xyz, N_emb_dir=config.N_emb_dir,
                          encode_a=config.encode_a, encode_t=config.encode_t, predict_label=config.predict_label,
                          num_classes=config.num_classes, M=config.num_parts,
                          disable_ellipsoid=config.disable_ellipsoid,
                          scale_min=config.scale_min, scale_max=config.scale_max,
                          use_spread_out_bias=config.use_spread_out_bias
                          ).cuda()
        load_ckpt(nerflet, ckpt_path, model_name='nerflet')

        sample = dataset[0]
        rays = sample['rays']
        ts = sample['ts']
        results = estimate_ellipsoid(nerflet, embeddings, rays.cuda(), ts.cuda(),
                                     config.N_samples, config.use_disp, config.chunk)


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

        cam_pos_zoom_out = cam_pos * 10

        # show(m, camera_position=cam_pos, camera_target=cam_target, light=light)
        render(m,
               behaviours=[
                   SaveFrames(str(original_img_path), every_n=1)
               ],
               n_frames=1, camera_position=cam_pos, camera_target=cam_target, light=light
        )
        render(m,
               behaviours=[
                   SaveFrames(str(zoom_out_img_path), every_n=1)
               ],
               n_frames=1, camera_position=cam_pos_zoom_out, camera_target=cam_target, light=light
        )

    # Generate a video for rendered images
    original_img_list = []
    writer = imageio.get_writer(original_render_dir / 'original_video.mp4', fps=10)
    for ckpt_path in ckpt_paths:
        ckpt_name = ckpt_path.stem
        img_path = original_render_dir / f"{ckpt_name}.png"
        im = imageio.imread(img_path)
        writer.append_data(im)
    writer.close()
    zoom_out_img_list = []
    writer = imageio.get_writer(zoom_out_render_dir / 'zoom_out_video.mp4', fps=10)
    for ckpt_path in ckpt_paths:
        ckpt_name = ckpt_path.stem
        img_path = zoom_out_render_dir / f"{ckpt_name}.png"
        im = imageio.imread(img_path)
        writer.append_data(im)
    writer.close()
