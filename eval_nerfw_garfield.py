import os
from opt import get_opts
import torch
from collections import defaultdict

from torch.utils.data import DataLoader

from tqdm import tqdm

# models
from models.nerf import (
    PosEmbedding,
    NeRFWG
)
from models.nerflet import Nerflet
from models.rendering_garfield import (
    render_rays
)

from datasets.sitcom3D import Sitcom3DDataset
from datasets.blender import BlenderDataset
from datasets.replica import ReplicaDataset
from datasets.front import ThreeDFrontDataset
from datasets.kitti360 import Kitti360Dataset
import numpy as np

from metrics import psnr

from utils.visualization import get_image_summary_from_vis_data, np_visualize_depth

import cv2
from utils import load_ckpt
import imageio
import argparse
from pathlib import Path
import json
from omegaconf import OmegaConf
# from sklearn.cluster import HDBSCAN
from cuml.cluster.hdbscan import HDBSCAN


@torch.no_grad()
def batched_inference(models, embeddings,
                      rays, ts, do_grouping, N_samples, N_importance, use_disp,
                      chunk, white_back, **kwargs):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)
    for chunk_idx in range(0, B, chunk):
        inputs = {'rays': rays[chunk_idx:chunk_idx+chunk],
                  'ts': ts[chunk_idx:chunk_idx+chunk]}
        rendered_ray_chunks = \
            render_rays(models=models,
                        embeddings=embeddings,
                        inputs=inputs,
                        N_samples=N_samples,
                        use_disp=use_disp,
                        perturb=0,
                        N_importance=N_importance,
                        chunk=chunk,
                        white_back=white_back,
                        test_time=True,
                        **kwargs)

        for k, v in rendered_ray_chunks.items():
            results[k].append(v)

        if do_grouping:
            weights = rendered_ray_chunks['weights_fine_static']
            pt_encodings = rendered_ray_chunks['pt_encodings']
            garfield_predictor = models['garfield_predictor']

            scales_0 = torch.ones(weights.shape[0], device=weights.device)
            garfield_0 = garfield_predictor.infer_garfield(pt_encodings, weights, scales_0)  # (B, dim_feat)
            results['garfield_0'].append(garfield_0)
            scales_1 = 0.5 * torch.ones(weights.shape[0], device=weights.device)
            garfield_1 = garfield_predictor.infer_garfield(pt_encodings, weights, scales_1)  # (B, dim_feat)
            results['garfield_1'].append(garfield_1)
            scales_2 = 0.1 * torch.ones(weights.shape[0], device=weights.device)
            garfield_2 = garfield_predictor.infer_garfield(pt_encodings, weights, scales_2)  # (B, dim_feat)
            results['garfield_2'].append(garfield_2)

    for k, v in results.items():
        results[k] = torch.cat(v, 0).cpu()

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

def cluster_2d_features(feats, clusterer, label_colors):
    clusters = clusterer.fit_predict(feats.numpy())
    cluster_map = label_colors[clusters]
    cluster_map = (cluster_map * 255).astype(np.uint8)
    return cluster_map


def render_to_path(path, dataset, idx, models, embeddings, config, do_grouping):
    sample = dataset[idx]
    rays = sample['rays']
    ts = sample['ts'].squeeze()
    results = batched_inference(models=models, embeddings=embeddings, rays=rays.cuda(), ts=ts.cuda(),
                                do_grouping=do_grouping, N_samples=config.N_samples, N_importance=config.N_importance,
                                use_disp=config.use_disp, chunk=config.chunk, white_back=dataset.white_back)

    rows = []
    metrics = {}

    # GT image and predicted image
    if config.dataset_name == 'sitcom3D':
        w, h = sample['img_wh']
    elif config.dataset_name == 'blender':
        w, h = config.img_wh
    elif config.dataset_name in ['replica', '3dfront', 'kitti360']:
        w, h = dataset.img_wh

    # GT image and predicted combined image
    img_pred = np.clip(results['rgb_fine_combined'].view(h, w, 3).cpu().numpy(), 0, 1)
    img_pred_ = (img_pred * 255).astype(np.uint8)
    rgbs = sample['rgbs']
    img_gt = rgbs.view(h, w, 3)
    psnr_ = psnr(img_gt, img_pred).item()
    print(f"PSNR: {psnr_}")
    metrics['psnr'] = psnr_
    img_gt_ = np.clip(img_gt.cpu().numpy(), 0, 1)
    img_gt_ = (img_gt_ * 255).astype(np.uint8)
    rows.append(np.concatenate([img_gt_, img_pred_], axis=1))

    # Predicted static image and predicted static depth
    if config.encode_t:
        img_static = np.clip(results['rgb_fine_static'].view(h, w, 3).cpu().numpy(), 0, 1)
        img_static_ = (img_static * 255).astype(np.uint8)
        static_depth = results['depth_fine_static'].cpu().numpy()
    else:
        img_static_ = np.zeros((h, w, 3), dtype=np.ubyte)
        static_depth = results['depth_fine'].cpu().numpy()
    depth_static = np.array(np_visualize_depth(static_depth, cmap=cv2.COLORMAP_BONE))
    depth_static = depth_static.reshape(h, w, 1)
    depth_static_ = np.repeat(depth_static, 3, axis=2)
    rows.append(np.concatenate([img_static_, depth_static_], axis=1))

    if do_grouping:
        clusterer = HDBSCAN(
            cluster_selection_epsilon=0.1,
            min_samples=30,
            min_cluster_size=30,
            allow_single_cluster=True,
        )
        label_colors = np.random.rand(200, 3)
        garfield_0 = results['garfield_0']
        cluster_map_0 = cluster_2d_features(garfield_0, clusterer, label_colors)

        cluster_map_0 = cluster_map_0.reshape(h, w, 3)
        garfield_1 = results['garfield_1']
        cluster_map_1 = cluster_2d_features(garfield_1, clusterer, label_colors)
        cluster_map_1 = cluster_map_1.reshape(h, w, 3)

        garfield_2 = results['garfield_2']
        cluster_map_2 = cluster_2d_features(garfield_2, clusterer, label_colors)
        cluster_map_2 = cluster_map_2.reshape(h, w, 3)

        placeholder = np.zeros_like(cluster_map_2)
        rows.append(np.concatenate([cluster_map_0, cluster_map_1], axis=1))
        rows.append(np.concatenate([cluster_map_2, placeholder], axis=1))

    res_img = np.concatenate(rows, axis=0)
    # imageio.imwrite(path, res_img)

    return metrics, res_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str)
    parser.add_argument('--output_dir', type=str, default='results/rendering')
    parser.add_argument('--use_ckpt', type=str)
    parser.add_argument('--select_part_idx', type=int)
    parser.add_argument('--sweep_parts', action='store_true', default=False)
    parser.add_argument('--num_parts', type=int, default=-1)
    parser.add_argument('--num_images', type=int, default=-1)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    output_dir = exp_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = exp_dir / 'config.json'
    with open(config_path, 'r') as f:
        file_config = json.load(f)
    file_config = OmegaConf.create(file_config)
    cli_config = OmegaConf.from_dotlist(args.opts)
    config = OmegaConf.merge(file_config, cli_config)

    kwargs = {}
    if config.dataset_name == 'sitcom3D':
        kwargs.update({'environment_dir': config.environment_dir,
                      'near_far_version': config.near_far_version})
        # kwargs['img_downscale'] = config.img_downscale
        kwargs['val_num'] = 5
        kwargs['use_cache'] = config.use_cache
        dataset = Sitcom3DDataset(split=args.split, img_downscale=config.img_downscale, **kwargs)
    elif config.dataset_name == 'blender':
        dataset = BlenderDataset(root_dir=config.environment_dir,
                                 img_wh=config.img_wh, split=args.split)
    elif config.dataset_name == 'replica':
        dataset = ReplicaDataset(root_dir=config.environment_dir,
                                 img_downscale=config.img_downscale, split=args.split,
                                 things_only=config.things_only if 'things_only' in config else False)
    elif config.dataset_name == '3dfront':
        dataset = ThreeDFrontDataset(root_dir=config.environment_dir,
                                     img_downscale=config.img_downscale, split=args.split,
                                     near=config.near, far=config.far)
    elif config.dataset_name == 'kitti360':
        dataset = Kitti360Dataset(root_dir=config.environment_dir, split=args.split,
                                  img_downscale=config.img_downscale,
                                  near=config.near, far=config.far, scene_bound=config.scene_bound)

    embedding_xyz = PosEmbedding(config.N_emb_xyz - 1, config.N_emb_xyz)
    embedding_dir = PosEmbedding(config.N_emb_dir - 1, config.N_emb_dir)
    embeddings = {'xyz': embedding_xyz, 'dir': embedding_dir}
    if config.encode_a:
        embedding_a = torch.nn.Embedding(config.N_vocab, config.N_a).cuda()
        load_ckpt(embedding_a, args.use_ckpt, model_name='embedding_a')
        embeddings['a'] = embedding_a
    if config.encode_t:
        embedding_t = torch.nn.Embedding(config.N_vocab, config.N_tau).cuda()
        load_ckpt(embedding_t, args.use_ckpt, model_name='embedding_t')
        embeddings['t'] = embedding_t

    nerf_coarse = NeRFWG('coarse',
                       D=config.num_hidden_layers,
                       W=config.dim_hidden_layers,
                       skips=config.skip_layers,
                       in_channels_xyz=6 * config.N_emb_xyz + 3,
                       in_channels_dir=6 * config.N_emb_dir + 3,
                       encode_appearance=False,
                       use_view_dirs=config.use_view_dirs).cuda()
    nerf_fine = NeRFWG('fine',
                     D=config.num_hidden_layers,
                     W=config.dim_hidden_layers,
                     skips=config.skip_layers,
                     in_channels_xyz=6 * config.N_emb_xyz + 3,
                     in_channels_dir=6 * config.N_emb_dir + 3,
                     encode_appearance=config.encode_a,
                     in_channels_a=config.N_a,
                     encode_transient=config.encode_t,
                     in_channels_t=config.N_tau,
                     beta_min=config.beta_min,
                     use_view_dirs=config.use_view_dirs).cuda()
    load_ckpt(nerf_coarse, args.use_ckpt, model_name='nerf_coarse')
    load_ckpt(nerf_fine, args.use_ckpt, model_name='nerf_fine')
    models = {'coarse': nerf_coarse, 'fine': nerf_fine}

    psnrs = []
    iou_combined = []
    iou_static = []
    label_colors = np.random.rand(config.num_classes, 3)

    for i in tqdm(range(len(dataset))):
        if args.num_images != -1 and i >= args.num_images:
            continue

        path = output_dir / f"{i:03d}.png"
        metrics, _ = render_to_path(path=path, dataset=dataset, idx=i, models=models, embeddings=embeddings,
                                    config=config, label_colors=label_colors)
        psnrs.append(metrics['psnr'])
        if 'iou_combined' in metrics:
            iou_combined.append(metrics['iou_combined'])
        if 'iou_static' in metrics:
            iou_static.append(metrics['iou_static'])

    mean_psnr = np.mean(psnrs)
    print(f'Mean PSNR : {mean_psnr:.2f}')