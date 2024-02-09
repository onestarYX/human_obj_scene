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
from models.nerflet import Nerflet, BgNeRF
from models.rendering_nerflet import (
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
import pickle


@torch.no_grad()
def batched_inference(models, embeddings,
                      rays, ts, predict_label, num_classes, N_samples, N_importance, use_disp,
                      chunk, use_bg_nerf, obj_mask, white_back, predict_density, use_fine_nerf, use_associated, **kwargs):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)
    for chunk_idx in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays(models=models,
                        embeddings=embeddings,
                        rays=rays[chunk_idx:chunk_idx+chunk],
                        ts=ts[chunk_idx:chunk_idx+chunk] if ts is not None else None,
                        predict_label=predict_label,
                        num_classes=num_classes,
                        N_samples=N_samples,
                        use_disp=use_disp,
                        N_importance=N_importance,
                        use_bg_nerf=use_bg_nerf,
                        obj_mask=obj_mask[chunk_idx:chunk_idx+chunk],
                        white_back=white_back,
                        predict_density=predict_density,
                        use_fine_nerf=use_fine_nerf,
                        perturb=0,
                        use_associated=use_associated,
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
                   label_colors, part_colors, write_to_path=True,
                   save_results=False, save_path=None, select_part_idx=None):
    sample = dataset[idx]
    rays = sample['rays']
    if 'ts2' in sample:
        ts = sample['ts2']
    else:
        ts = sample['ts']
    obj_mask = sample['obj_mask']
    # TODO: the arguments of this function can be simplified to only pass config
    results = batched_inference(models=models, embeddings=embeddings, rays=rays.cuda(), ts=ts.cuda(),
                                predict_label=config.predict_label, num_classes=config.num_classes,
                                N_samples=config.N_samples, N_importance=config.N_importance,
                                use_disp=config.use_disp, chunk=config.chunk,
                                use_bg_nerf=config.use_bg_nerf, obj_mask=obj_mask, white_back=dataset.white_back,
                                predict_density=config.predict_density, use_fine_nerf=config.use_fine_nerf,
                                use_associated=config.use_associated)

    if save_results:
        res_to_save = {}
        res_to_save['static_label'] = torch.argmax(results['static_label'], dim=1).to(torch.long).cpu().numpy()
        res_to_save['static_ray_associations_fine'] = results['static_ray_associations_fine'].cpu().numpy()
        res_to_save['static_positive_rays_fine'] = results['static_positive_rays_fine'].cpu().numpy()
        with open(save_path, 'wb') as f:
            pickle.dump(res_to_save, f)

    rows = []
    metrics = {}

    # GT image and predicted image
    if config.dataset_name == 'sitcom3D':
        w, h = sample['img_wh']
    elif config.dataset_name == 'blender':
        w, h = config.img_wh
    elif config.dataset_name in ['replica', '3dfront', 'kitti360']:
        w, h = dataset.img_wh

    # TODO: For now only consider fine nerf, might need to support coarse only

    if config.use_bg_nerf:
        # GT image and predicted combined image
        ray_associations = results['static_ray_associations_fine'].cpu().numpy()
        positive_rays_mask = results['static_positive_rays_fine'].cpu().numpy()
        if config.encode_t:
            img_pred_obj = results['combined_rgb_map']
        else:
            img_pred_obj = results['static_rgb_map_fine']
        img_pred_bg = results['rgb_bg']
        img_pred = torch.zeros(h * w, 3, device=img_pred_obj.device)
        img_pred[obj_mask] = img_pred_obj
        img_pred[~obj_mask] = img_pred_bg
        img_pred_ = (img_pred.cpu().numpy() * 255).astype(np.uint8).reshape(h, w, 3)

        rgbs = sample['rgbs']
        img_gt = rgbs.view(h, w, 3)
        psnr_ = psnr(img_gt, img_pred.view(h, w, 3)).item()
        print(f"PSNR: {psnr_}")
        metrics['psnr'] = psnr_
        img_gt_ = np.clip(img_gt.cpu().numpy(), 0, 1)
        img_gt_ = (img_gt_ * 255).astype(np.uint8)
        rows.append(np.concatenate([img_gt_, img_pred_], axis=1))

        # obj img and bg img
        img_part_obj = torch.zeros(h * w, 3, device=img_pred_obj.device)
        img_part_obj[obj_mask] = img_pred_obj
        img_part_obj = (img_part_obj.cpu().numpy() * 255).astype(np.uint8).reshape(h, w, 3)
        img_part_bg = torch.zeros(h * w, 3, device=img_pred_obj.device)
        img_part_bg[~obj_mask] = img_pred_bg
        img_part_bg = (img_part_bg.cpu().numpy() * 255).astype(np.uint8).reshape(h, w, 3)
        rows.append(np.concatenate([img_part_obj, img_part_bg], axis=1))

        # Predicted static image and predicted static depth
        placeholder = np.zeros((h, w, 3), dtype=np.uint8)
        depth_obj = results['static_depth_fine']
        depth_bg = results['depth_bg']
        static_depth = torch.zeros(h * w, device=depth_obj.device)
        static_depth[obj_mask] = depth_obj
        static_depth[~obj_mask] = depth_bg
        static_depth = static_depth.cpu().numpy()
        static_depth_map = np.array(np_visualize_depth(static_depth, cmap=cv2.COLORMAP_BONE))
        static_depth_map = static_depth_map.reshape(h, w, 1)
        static_depth_map = np.repeat(static_depth_map, 3, axis=2)
        rows.append(np.concatenate([placeholder, static_depth_map], axis=1))

        # gt label and pred label
        if config.predict_label:
            label_gt = sample['labels'].to(torch.long).cpu().numpy()
            label_map_gt = label_colors[label_gt].reshape((h, w, 3))
            label_map_gt = (label_map_gt * 255).astype(np.uint8)
            if config.encode_t:
                label_pred_obj = results['combined_label']
            else:
                label_pred_obj = results['static_label']
            label_pred_obj = torch.argmax(label_pred_obj, dim=1).to(torch.long).cpu().numpy()
            label_pred_obj = label_colors[label_pred_obj]
            label_pred_obj[~positive_rays_mask] = 0
            label_pred_bg = torch.argmax(results['label_bg'], dim=1).to(torch.long).cpu().numpy()
            label_pred_bg = label_colors[label_pred_bg]
            label_map_pred = np.zeros((h * w, 3))
            label_map_pred[obj_mask.cpu().numpy()] = label_pred_obj
            label_map_pred[~obj_mask.cpu().numpy()] = label_pred_bg

            label_map_pred = (label_map_pred * 255).astype(np.uint8).reshape(h, w, 3)
            rows.append(np.concatenate([label_map_gt, label_map_pred], axis=1))

            if config.encode_t:
                raise NotImplementedError

        ray_association_map = np.zeros((h * w, 3))
        ray_associations_ = part_colors[ray_associations]
        ray_associations_[~positive_rays_mask] = 0
        ray_association_map[obj_mask.cpu().numpy()] = ray_associations_
        ray_association_map = (ray_association_map * 255).astype(np.uint8).reshape(h, w, 3)
        rows.append(np.concatenate([ray_association_map, placeholder], axis=1))

    else:   # No bg nerf involved
        # GT image and predicted combined image
        ray_associations = results['static_ray_associations_fine'].cpu().numpy().reshape((h, w))
        positive_rays_mask = results['static_positive_rays_fine'].cpu().numpy().reshape((h, w))
        if config.encode_t:
            img_pred = results['combined_rgb_map'].view(h, w, 3).cpu().numpy()
        else:
            img_pred = results['static_rgb_map_fine'].view(h, w, 3).cpu().numpy()
        img_pred_ = (img_pred * 255).astype(np.uint8)
        if select_part_idx is not None:
            non_selected_part_mask = np.logical_and(ray_associations != select_part_idx, np.logical_not(positive_rays_mask))
            img_pred_[non_selected_part_mask] = 255
        rgbs = sample['rgbs']
        img_gt = rgbs.view(h, w, 3)
        psnr_ = psnr(img_gt, img_pred).item()
        print(f"PSNR: {psnr_}")
        metrics['psnr'] = psnr_
        img_gt_ = np.clip(img_gt.cpu().numpy(), 0, 1)
        img_gt_ = (img_gt_ * 255).astype(np.uint8)
        rows.append(np.concatenate([img_gt_, img_pred_], axis=1))

        # Predicted static image and predicted static depth
        img_static = np.clip(results['static_rgb_map_fine'].view(h, w, 3).cpu().numpy(), 0, 1)
        img_static_ = (img_static * 255).astype(np.uint8)
        static_depth = results['static_depth_fine'].cpu().numpy()
        static_depth_map = np.array(np_visualize_depth(static_depth, cmap=cv2.COLORMAP_BONE))
        static_depth_map = static_depth_map.reshape(h, w, 1)
        static_depth_map = np.repeat(static_depth_map, 3, axis=2)
        rows.append(np.concatenate([img_static_, static_depth_map], axis=1))

        # gt label and pred label
        if config.predict_label:
            label_gt = sample['labels'].to(torch.long).cpu().numpy()
            label_map_gt = label_colors[label_gt].reshape((h, w, 3))
            label_map_gt = (label_map_gt * 255).astype(np.uint8)
            if config.encode_t:
                label_pred = results['combined_label']
            else:
                label_pred = results['static_label']
            label_pred = torch.argmax(label_pred, dim=1).to(torch.long).cpu().numpy()
            label_map_pred = label_colors[label_pred].reshape((h, w, 3))
            label_map_pred[~positive_rays_mask] = 0
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

        ray_association_map = part_colors[ray_associations]
        ray_association_map[~positive_rays_mask] = 0
        ray_association_map = (ray_association_map * 255).astype(np.uint8)
        obj_mask = results['static_mask_fine'].cpu().numpy()
        obj_mask = obj_mask[..., np.newaxis] * np.array([[1, 1, 1]])
        obj_mask = obj_mask.reshape((h, w, 3))
        obj_mask = (obj_mask * 255).astype(np.uint8)
        rows.append(np.concatenate([ray_association_map, obj_mask], axis=1))

    res_img = np.concatenate(rows, axis=0)
    if write_to_path:
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
    parser.add_argument('--save_results', action='store_true', default=False)
    parser.add_argument('--save_dir', type=str, default='results/eval')
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    output_dir = exp_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_results:
        save_dir = exp_dir / args.save_dir
        save_dir.mkdir(parents=True, exist_ok=True)

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
        dataset = Sitcom3DDataset(split=args.split, img_downscale=config.img_downscale, near=config.near, **kwargs)
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

    disable_ellipsoid = config.disable_ellipsoid if 'disable_ellipsoid' in config else False
    disable_tf = config.disable_tf if 'disable_tf' in config else False
    bbox = dataset.bbox if hasattr(dataset, 'bbox') else None
    sharpness = config.sharpness if 'sharpness' in config else 100
    models = {}
    nerflet = Nerflet(D=config.num_hidden_layers, W=config.dim_hidden_layers, skips=config.skip_layers,
                      N_emb_xyz=config.N_emb_xyz, N_emb_dir=config.N_emb_dir,
                      encode_a=config.encode_a, encode_t=config.encode_t, predict_label=config.predict_label,
                      num_classes=config.num_classes, M=config.num_parts,
                      disable_ellipsoid=disable_ellipsoid,
                      scale_min=config.scale_min, scale_max=config.scale_max,
                      use_spread_out_bias=config.use_spread_out_bias, bbox=bbox,
                      label_only=config.label_only, disable_tf=disable_tf,
                      sharpness=sharpness, predict_density=config.predict_density).cuda()
    load_ckpt(nerflet, args.use_ckpt, model_name='nerflet')
    models['nerflet'] = nerflet
    if config.use_bg_nerf:
        bg_nerf = BgNeRF(D=config.num_hidden_layers,
                         W=config.dim_hidden_layers,
                         skips=config.skip_layers,
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
        load_ckpt(bg_nerf, args.use_ckpt, model_name="bg_nerf")
        models['bg_nerf'] = bg_nerf

    psnrs = []
    np.random.seed(19)
    label_colors = np.random.rand(config.num_classes, 3)
    part_colors = np.random.rand(config.num_parts, 3)
    iou_combined = []
    iou_static = []

    for i in tqdm(range(len(dataset))):
        if args.num_images != -1 and i >= args.num_images:
            continue
        if args.sweep_parts:
            for j in range(config.num_parts):
                if args.num_parts != -1 and j >= args.num_parts:
                    continue
                print(f"Rendering part {j}")
                render_img_path = output_dir / f"frame_{i:03d}"
                render_img_path.mkdir(exist_ok=True)
                render_img_path = render_img_path / f"part_{j}.png"
                render_to_path(render_img_path, dataset=dataset, idx=i, models=models, embeddings=embeddings,
                               config=config, label_colors=label_colors, part_colors=part_colors,
                               select_part_idx=j)
        elif args.select_part_idx is not None:
            render_img_path = output_dir / f"{i:03d}"
            render_img_path.mkdir(exist_ok=True)
            render_img_path = render_img_path / f"{args.select_part_idx}.png"
            render_to_path(render_img_path, dataset=dataset, idx=i, models=models, embeddings=embeddings,
                           config=config, label_colors=label_colors, part_colors=part_colors,
                           select_part_idx=args.select_part_idx)
        else:
            render_img_path = output_dir / f"{i:03d}.png"
            save_results_path = save_dir / f"{i:03d}.pkl"
            metrics, _ = render_to_path(render_img_path, dataset=dataset, idx=i, models=models, embeddings=embeddings,
                                        config=config, label_colors=label_colors, part_colors=part_colors,
                                        save_results=args.save_results, save_path=save_results_path)
            psnrs.append(metrics['psnr'])
            if 'iou_combined' in metrics:
                iou_combined.append(metrics['iou_combined'])
            if 'iou_static' in metrics:
                iou_static.append(metrics['iou_static'])

    if config.predict_label:
        print('Mean IoU combined', np.mean(iou_combined))
        print('Mean IoU static', np.mean(iou_static))

    if psnrs:
        mean_psnr = np.mean(psnrs)
        print(f'Mean PSNR : {mean_psnr:.2f}')