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
import argparse
import csv
import json
from omegaconf import OmegaConf
from pathlib import Path
import imageio
from tqdm import tqdm
import open3d as o3d

@torch.no_grad()
def inference_pts_occ(model, embeddings, xyz, chunk):
    """Do batched inference on 3D points using chunk."""
    occ_list = []

    for i in tqdm(range(0, len(xyz), chunk)):
        xyz_ = xyz[i:i + chunk]
        # Make dummy rays direction and ts
        rays_d = torch.zeros(len(xyz_), 1, 3).to(xyz.device)
        rays_d[:, 0, 0] = 1
        ts = torch.zeros(len(xyz_)).to(torch.int32).to(xyz.device)
        pred = get_nerflet_pred(model, embeddings, xyz_, rays_d, ts)
        occ_list.append(pred['static_occ'].cpu())

    occ_list = torch.cat(occ_list, dim=0)
    return occ_list


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
    parser.add_argument('--output_dir', type=str, default='results/pcd_cam')
    parser.add_argument('--use_ckpt', type=str)
    parser.add_argument('--split', type=str, default='test_train')
    parser.add_argument('--num_test', type=int, default=5)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # Get config
    exp_dir = Path(args.exp_dir)
    output_dir = exp_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = exp_dir / 'config.json'
    with open(config_path, 'r') as f:
        file_config = json.load(f)
    file_config = OmegaConf.create(file_config)
    cli_config = OmegaConf.from_dotlist(args.opts)
    config = OmegaConf.merge(file_config, cli_config)

    # Create dataset
    if config.dataset_name == 'sitcom3D':
        kwargs = {'environment_dir': config.environment_dir, 'near_far_version': config.near_far_version, 'val_num': 5,
                  'use_cache': config.use_cache}
        # kwargs['img_downscale'] = args.img_downscale
        dataset = Sitcom3DDataset(split=args.split, img_downscale=config.img_downscale, near=config.near, **kwargs)
    elif config.dataset_name == 'blender':
        kwargs = {}
        dataset = BlenderDataset(root_dir=config.environment_dir,
                                 img_wh=config.img_wh, split=args.split)
    elif config.dataset_name == 'replica':
        dataset = ReplicaDataset(root_dir=config.environment_dir,
                                 img_downscale=config.img_downscale, split=args.split)
    elif config.dataset_name == '3dfront':
        dataset = ThreeDFrontDataset(root_dir=config.environment_dir,
                                     img_downscale=config.img_downscale, split=args.split,
                                     near=config.near, far=config.far)

    # Construct and load model
    embedding_xyz = PosEmbedding(config.N_emb_xyz - 1, config.N_emb_xyz)
    embedding_dir = PosEmbedding(config.N_emb_dir - 1, config.N_emb_dir)
    embeddings = {'xyz': embedding_xyz, 'dir': embedding_dir}

    if args.use_ckpt:
        ckpt_path = Path(args.use_ckpt)
    else:
        ckpt_paths = []
        ckpt_dir = exp_dir / 'ckpts'
        for ckpt_path in ckpt_dir.iterdir():
            ckpt_paths.append(ckpt_path)
        def get_step_from_path(path):
            step = path.stem.split('=')[-1]
            return int(step)
        ckpt_paths.sort(key=get_step_from_path)
        ckpt_path = ckpt_paths[-1]
    print(f"Working on {ckpt_path}")
    ckpt_name = ckpt_path.stem

    if config.encode_a:
        embedding_a = torch.nn.Embedding(config.N_vocab, config.N_a).cuda()
        load_ckpt(embedding_a, ckpt_path, model_name='embedding_a')
        embeddings['a'] = embedding_a
    if config.encode_t:
        embedding_t = torch.nn.Embedding(config.N_vocab, config.N_tau).cuda()
        load_ckpt(embedding_t, ckpt_path, model_name='embedding_t')
        embeddings['t'] = embedding_t
    bbox = dataset.bbox if hasattr(dataset, 'bbox') else None
    nerflet = Nerflet(N_emb_xyz=config.N_emb_xyz, N_emb_dir=config.N_emb_dir,
                      encode_a=config.encode_a, encode_t=config.encode_t, predict_label=config.predict_label,
                      num_classes=config.num_classes, M=config.num_parts,
                      disable_ellipsoid=config.disable_ellipsoid,
                      scale_min=config.scale_min, scale_max=config.scale_max,
                      use_spread_out_bias=config.use_spread_out_bias,
                      bbox=bbox).cuda()
    load_ckpt(nerflet, ckpt_path, model_name='nerflet')

    # Prepare colors for each part
    np.random.seed(19)
    part_colors = np.random.rand(config.num_parts, 3)

    # Uniformly sample points in the 3D space and make inference
    for i in range(args.num_test):
        print(f"Checking image {i}")
        sample = dataset[i]
        rays = sample['rays']
        xyz, _, _ = get_input_from_rays(rays, config.N_samples, config.use_disp)
        xyz = xyz.cuda()
        results = inference_pts_occ(nerflet, embeddings, xyz, config.chunk)

        xyz = xyz.reshape(-1, 3).cpu()
        results = results.reshape(-1, config.num_parts)

        geo = []
        occ_threshold = 0.5
        pt_max_occ, pt_association = results.max(dim=-1)
        pt_occupied_mask = pt_max_occ > occ_threshold
        pt_to_show = xyz[pt_occupied_mask]
        pt_to_show_association = pt_association[pt_occupied_mask]
        for idx in range(config.num_parts):
            pt_part_mask = pt_to_show_association == idx
            pt_part = pt_to_show[pt_part_mask]
            if len(pt_part) == 0:
                continue
            part_color = part_colors[idx]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pt_part)
            colors = np.tile(part_color, (len(pt_part), 1))
            pcd.colors = o3d.utility.Vector3dVector(colors)
            geo.append(pcd)

        # For the rest points that have low occupancy
        pt_rest = xyz[np.invert(pt_occupied_mask)]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pt_rest)
        colors = np.tile(np.array([1e-3, 1e-3, 1e-3]), (len(pt_rest), 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd = pcd.voxel_down_sample(voxel_size=0.1)
        geo.append(pcd)

        if hasattr(dataset, 'bbox'):
            bbox = dataset.bbox
            mesh_sphere_0 = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
            mesh_sphere_0.paint_uniform_color([0.1, 0.1, 0.7])
            mesh_sphere_0.translate(bbox[0])
            geo.append(mesh_sphere_0)

            mesh_sphere_1 = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
            mesh_sphere_1.paint_uniform_color([0.7, 0.1, 0.1])
            mesh_sphere_1.translate(bbox[1])
            geo.append(mesh_sphere_1)

            # mesh_sphere_2 = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
            # mesh_sphere_2.paint_uniform_color([0.1, 0.7, 0.1])
            # geo.append(mesh_sphere_2)

            min_pt = bbox[0]
            max_pt = bbox[1]
            bbox_points = [[min_pt[0], min_pt[1], min_pt[2]], [min_pt[0], max_pt[1], min_pt[2]],
                           [max_pt[0], min_pt[1], min_pt[2]], [max_pt[0], max_pt[1], min_pt[2]],
                           [min_pt[0], min_pt[1], max_pt[2]], [min_pt[0], max_pt[1], max_pt[2]],
                           [max_pt[0], min_pt[1], max_pt[2]], [max_pt[0], max_pt[1], max_pt[2]]]
            lines = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
            line_colors = [[0, 1, 0] for i in range(len(lines))]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(bbox_points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(line_colors)
            geo.append(line_set)

        o3d.visualization.draw_geometries(geo)
