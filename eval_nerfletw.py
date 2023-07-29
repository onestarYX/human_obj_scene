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
from models.rendering_nerflet import (
    render_rays
)

from datasets.sitcom3D import Sitcom3DDataset
from datasets.blender import BlenderDataset
import numpy as np

from metrics import psnr

from utils.visualization import get_image_summary_from_vis_data, np_visualize_depth

import cv2
from utils import load_ckpt
import imageio


@torch.no_grad()
def batched_inference(models, embeddings,
                      rays, ts, predict_label, num_classes, N_samples, N_importance, use_disp,
                      chunk, white_back, **kwargs):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays(models,
                        embeddings,
                        rays[i:i+chunk],
                        ts[i:i+chunk] if ts is not None else None,
                        predict_label,
                        num_classes,
                        N_samples,
                        use_disp,
                        N_importance,
                        chunk,
                        white_back,
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

    models = {'nerflet': nerflet}

    imgs, psnrs = [], []
    run_name = args.ckpt_path.split("/")[-4]
    dir_name = f'{args.environment_dir}/rendering/{run_name}'
    os.makedirs(dir_name, exist_ok=True)

    label_colors = np.random.rand(args.num_classes, 3)
    part_colors = np.random.rand(16, 3)

    iou_combined = []
    iou_static = []
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        rays = sample['rays']
        ts = sample['ts']
        results = batched_inference(models, embeddings, rays.cuda(), ts.cuda(),
                                    args.predict_label, args.num_classes,
                                    args.N_samples, args.N_importance, args.use_disp,
                                    args.chunk,
                                    dataset.white_back,
                                    **kwargs)

        rows = []

        # GT image and predicted image
        if args.dataset_name == 'sitcom3D':
            w, h = sample['img_wh']
        else:
            w, h = args.img_wh
        if args.encode_t:
            img_pred = np.clip(results['combined_rgb_map'].view(h, w, 3).cpu().numpy(), 0, 1)
        else:
            img_pred = np.clip(results['static_rgb_map'].view(h, w, 3).cpu().numpy(), 0, 1)
        img_pred_ = (img_pred * 255).astype(np.uint8)
        rgbs = sample['rgbs']
        img_gt = rgbs.view(h, w, 3)
        psnrs += [psnr(img_gt, img_pred).item()]
        img_gt_ = np.clip(img_gt.cpu().numpy(), 0, 1)
        img_gt_ = (img_gt_ * 255).astype(np.uint8)
        rows.append(np.concatenate([img_gt_, img_pred_], axis=1))

        # Predicted static image and predicted static depth
        img_static = np.clip(results['static_rgb_map'].view(h, w, 3).cpu().numpy(), 0, 1)
        img_static_ = (img_static * 255).astype(np.uint8)
        depth_static = np.array(np_visualize_depth(results['static_depth'].cpu().numpy(), cmap=cv2.COLORMAP_BONE))
        depth_static = depth_static.reshape(h, w, 1)
        depth_static_ = np.repeat(depth_static, 3, axis=2)
        rows.append(np.concatenate([img_static_, depth_static_], axis=1))

        # gt label and pred label
        if args.predict_label:
            label_gt = sample['labels'].to(torch.long).cpu().numpy()
            label_map_gt = label_colors[label_gt].reshape((h, w, 3))
            label_map_gt = (label_map_gt * 255).astype(np.uint8)
            if args.encode_t:
                label_pred = results['combined_label']
            else:
                label_pred = results['static_label']
            label_pred = torch.argmax(label_pred, dim=1).to(torch.long).cpu().numpy()
            label_map_pred = label_colors[label_pred].reshape((h, w, 3))
            label_map_pred = (label_map_pred * 255).astype(np.uint8)
            iou_combined.append(compute_iou(label_pred, label_gt, args.num_classes))
            rows.append(np.concatenate([label_map_gt, label_map_pred], axis=1))

            if args.encode_t:
                label_static_pred = torch.argmax(results['static_label'], dim=1).to(torch.long).cpu().numpy()
                label_map_static_pred = label_colors[label_static_pred].reshape((h, w, 3))
                label_map_static_pred = (label_map_static_pred * 255).astype(np.uint8)
                iou_static.append(compute_iou(label_static_pred, label_gt, args.num_classes))

                label_transient_pred = torch.argmax(results['transient_label'], dim=1).to(torch.long).cpu().numpy()
                label_map_transient_pred = label_colors[label_transient_pred].reshape((h, w, 3))
                label_map_transient_pred = (label_map_transient_pred * 255).astype(np.uint8)
                rows.append(np.concatenate([label_map_static_pred, label_map_transient_pred], axis=1))

        ray_associations = results['static_ray_associations'].cpu().numpy()
        ray_association_map = part_colors[ray_associations].reshape((h, w, 3))
        ray_association_map = (ray_association_map * 255).astype(np.uint8)
        rows.append(np.concatenate([ray_association_map, np.zeros_like(ray_association_map)], axis=1))

        res_img = np.concatenate(rows, axis=0)
        imageio.imwrite(os.path.join(dir_name, f'{i:03d}.png'), res_img)

    if args.predict_label:
        print('Mean IoU combined', iou_combined)
        print('Mean IoU static', iou_static)

    if psnrs:
        mean_psnr = np.mean(psnrs)
        print(f'Mean PSNR : {mean_psnr:.2f}')