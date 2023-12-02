import os
from opt import get_opts
import torch
from collections import defaultdict

from torch.utils.data import DataLoader

from tqdm import tqdm

# models
from models.nerf import (
    PosEmbedding,
    NeRF
)
from models.nerflet import Nerflet
from models.rendering import (
    render_rays
)

from datasets.sitcom3D import Sitcom3DDataset
from datasets.blender import BlenderDataset
from datasets.replica import ReplicaDataset
from datasets.front import ThreeDFrontDataset
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


@torch.no_grad()
def batched_inference(models, embeddings,
                      rays, ts, predict_label, num_classes, N_samples, N_importance, use_disp,
                      chunk, white_back, **kwargs):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)
    for chunk_idx in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays(models,
                        embeddings,
                        rays[chunk_idx:chunk_idx+chunk],
                        ts[chunk_idx:chunk_idx+chunk] if ts is not None else None,
                        predict_label,
                        num_classes,
                        N_samples,
                        use_disp,
                        perturb=0,
                        N_importance=N_importance,
                        chunk=chunk,
                        white_back=white_back,
                        test_time=True,
                        **kwargs)

        for k, v in rendered_ray_chunks.items():
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



def render_to_path(path, dataset, idx, models, embeddings, config,
                   label_colors):
    sample = dataset[idx]
    rays = sample['rays']
    if 'ts2' in sample:
        ts = sample['ts2']
    else:
        ts = sample['ts']
    results = batched_inference(models=models, embeddings=embeddings, rays=rays.cuda(), ts=ts.cuda(),
                                predict_label=config.predict_label, num_classes=config.num_classes,
                                N_samples=config.N_samples, N_importance=config.N_importance,
                                use_disp=config.use_disp, chunk=config.chunk, white_back=dataset.white_back)

    rows = []
    metrics = {}

    # GT image and predicted image
    if config.dataset_name == 'sitcom3D':
        w, h = sample['img_wh']
    elif config.dataset_name == 'blender':
        w, h = config.img_wh
    elif config.dataset_name == 'replica' or config.dataset_name == '3dfront':
        w, h = dataset.img_wh

    # GT image and predicted combined image
    img_pred = np.clip(results['rgb_fine'].view(h, w, 3).cpu().numpy(), 0, 1)
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
        static_depth = results['depth_fine_static_exp'].cpu().numpy()
    else:
        img_static_ = np.zeros((h, w, 3), dtype=np.ubyte)
        static_depth = results['depth_fine'].cpu().numpy()
    depth_static = np.array(np_visualize_depth(static_depth, cmap=cv2.COLORMAP_BONE))
    depth_static = depth_static.reshape(h, w, 1)
    depth_static_ = np.repeat(depth_static, 3, axis=2)
    rows.append(np.concatenate([img_static_, depth_static_], axis=1))

    # gt label and pred label
    if config.predict_label:
        label_gt = sample['labels'].to(torch.long).cpu().numpy()
        label_map_gt = label_colors[label_gt].reshape((h, w, 3))
        label_map_gt = (label_map_gt * 255).astype(np.uint8)
        label_pred = results['label_fine']
        label_pred = torch.argmax(label_pred, dim=1).to(torch.long).cpu().numpy()
        label_map_pred = label_colors[label_pred].reshape((h, w, 3))
        label_map_pred = (label_map_pred * 255).astype(np.uint8)
        iou = compute_iou(label_pred, label_gt, config.num_classes)
        metrics['iou_combined'] = iou
        print(f"Semantic iou: {iou}")
        rows.append(np.concatenate([label_map_gt, label_map_pred], axis=1))

        if config.encode_t:
            label_static_pred = torch.argmax(results['static_label'], dim=1).to(torch.long).cpu().numpy()
            label_map_static_pred = label_colors[label_static_pred].reshape((h, w, 3))
            label_map_static_pred = (label_map_static_pred * 255).astype(np.uint8)
            metrics['iou_static'] = compute_iou(label_static_pred, label_gt, config.num_classes)

            label_transient_pred = torch.argmax(results['transient_label'], dim=1).to(torch.long).cpu().numpy()
            label_map_transient_pred = label_colors[label_transient_pred].reshape((h, w, 3))
            label_map_transient_pred = (label_map_transient_pred * 255).astype(np.uint8)
            rows.append(np.concatenate([label_map_static_pred, label_map_transient_pred], axis=1))

    res_img = np.concatenate(rows, axis=0)
    imageio.imwrite(path, res_img)

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

    nerf_coarse = NeRF('coarse',
                        in_channels_xyz=6 * config.N_emb_xyz + 3,
                        in_channels_dir=6 * config.N_emb_dir + 3,
                        encode_appearance=False,
                        predict_label=config.predict_label,
                        num_classes=config.num_classes,
                        use_view_dirs=config.use_view_dirs).cuda()
    nerf_fine = NeRF('fine',
                      in_channels_xyz=6 * config.N_emb_xyz + 3,
                      in_channels_dir=6 * config.N_emb_dir + 3,
                      encode_appearance=config.encode_a,
                      in_channels_a=config.N_a,
                      encode_transient=config.encode_t,
                      in_channels_t=config.N_tau,
                      predict_label=config.predict_label,
                      num_classes=config.num_classes,
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

    if config.predict_label:
        print('Mean IoU combined', np.mean(iou_combined))
        print('Mean IoU static', np.mean(iou_static))