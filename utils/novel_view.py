import os
import torch
from collections import defaultdict

# models
from models.nerf import PosEmbedding
from models.nerflet import Nerflet, BgNeRF
from models.rendering_nerflet import (
    get_input_from_rays,
    get_nerflet_pred,
    compose_nerflet_bgnerf
)
from models.model_utils import quaternions_to_rotation_matrices

from datasets.sitcom3D import Sitcom3DDataset, get_rays
# from datasets.blender import BlenderDataset
# from datasets.replica import ReplicaDataset
# from datasets.front import ThreeDFrontDataset
from utils import load_ckpt
import numpy as np
import argparse
import json
from omegaconf import OmegaConf
from pathlib import Path
import imageio
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import copy
from PIL import Image

def move_cam(input, translation_delta, rotation_delta=None):
    c2w = input['c2w']
    c2w[:, -1] += translation_delta
    directions = input['directions']
    img_w = input['img_wh'][0]
    img_h = input['img_wh'][1]
    directions = directions.reshape(img_h, img_w, -1)
    rays_o, rays_d = get_rays(directions, c2w)
    input['rays'][:, :3] = rays_o
    input['rays'][:, 3:6] = rays_d

def update_cam(input, new_cam_pose):
    directions = input['directions']
    img_w = input['img_wh'][0]
    img_h = input['img_wh'][1]
    directions = directions.reshape(img_h, img_w, -1)
    rays_o, rays_d = get_rays(directions, new_cam_pose)
    input['rays'][:, :3] = rays_o
    input['rays'][:, 3:6] = rays_d

def gen_pan_cam_traj(first_view, num_novel_views):
    pan_radius = 0.1
    step = np.pi * 2 / num_novel_views

    img_w = first_view['img_wh'][0]
    img_h = first_view['img_wh'][1]
    directions = first_view['directions'].reshape(img_h, img_w, -1)
    n = directions[0, 0] + directions[img_h - 1, img_w - 1]
    v = torch.tensor([0, 0, 1], dtype=torch.float32)
    proj_v_on_n = torch.dot(n, v) / (torch.norm(n)**2) * n
    start_vec = torch.nn.functional.normalize(v - proj_v_on_n, dim=0)
    t = start_vec * pan_radius
    rotation = R.from_rotvec(step * n)

    translation_list = [torch.zeros(3), t]
    for idx in range(num_novel_views - 1):
        t_np = t.numpy()
        t_np = rotation.apply(t_np)
        t = torch.tensor(t_np)
        translation_list.append(t)

    return translation_list

def interpolate_transformation(input0, input1, num_steps):
    transformation_list = []
    key_rots = torch.stack([input0[:, :3], input1[:, :3]], dim=0).numpy()
    key_rots = R.from_matrix(key_rots)
    key_times = [0, 1]
    slerp = Slerp(key_times, key_rots)
    steps = np.linspace(start=0, stop=1, num=num_steps)
    interp_rots = slerp(steps)
    interp_rots = interp_rots.as_matrix()

    key_shift0 = input0[:, 3]
    key_shift1 = input1[:, 3]
    for idx in range(num_steps):
        new_tf = torch.zeros_like(input0)
        # translation
        step = steps[idx]
        new_shift = (1 - step) * key_shift0 + step * key_shift1
        new_rot = interp_rots[idx]
        new_tf[:, :3] = torch.tensor(new_rot)
        new_tf[:, 3] = torch.tensor(new_shift)
        transformation_list.append(new_tf)

    return transformation_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results/novel_view')
    parser.add_argument('--use_ckpt', type=str)
    parser.add_argument('--split', type=str, default='test_train')
    parser.add_argument('--num_test_img', type=int, default=5)
    parser.add_argument('--num_novel_views', type=int, default=20)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # Get config
    exp_dir = Path(args.exp_dir)
    output_dir = exp_dir / args.output_dir
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
    else:
        raise NotImplementedError

    # Construct and load model
    embedding_xyz = PosEmbedding(config.N_emb_xyz - 1, config.N_emb_xyz)
    embedding_dir = PosEmbedding(config.N_emb_dir - 1, config.N_emb_dir)
    embeddings = {'xyz': embedding_xyz, 'dir': embedding_dir}

    ckpt_path = Path(args.use_ckpt)
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
    sharpness = config.sharpness if 'sharpness' in config else 100
    models = {}
    nerflet = Nerflet(D=config.num_hidden_layers, W=config.dim_hidden_layers, skips=config.skip_layers,
                      N_emb_xyz=config.N_emb_xyz, N_emb_dir=config.N_emb_dir,
                      encode_a=config.encode_a, encode_t=config.encode_t,
                      predict_label=config.predict_label, num_classes=config.num_classes,
                      M=config.num_parts, disable_ellipsoid=config.disable_ellipsoid,
                      scale_min=config.scale_min, scale_max=config.scale_max,
                      use_spread_out_bias=config.use_spread_out_bias,
                      bbox=bbox, label_only=config.label_only, disable_tf=config.disable_tf,
                      sharpness=sharpness, predict_density=config.predict_density).cuda()
    load_ckpt(nerflet, ckpt_path, model_name='nerflet')
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
        load_ckpt(bg_nerf, ckpt_path, model_name="bg_nerf")
        models['bg_nerf'] = bg_nerf

    # Prepare colors for each part
    np.random.seed(19)
    label_colors = np.random.rand(config.num_classes, 3)
    part_colors = np.random.rand(config.num_parts, 3)

    sample0 = dataset[0]
    sample1 = dataset[1]
    img_output_dir = output_dir / f"view_0_1"
    img_output_dir.mkdir(parents=True, exist_ok=True)

    # Construct a cam trajectory
    transformation_list = interpolate_transformation(sample0['c2w'], sample1['c2w'], args.num_novel_views)

    sample = copy.deepcopy(sample0)
    img_list = []
    for view_idx, t in tqdm(enumerate(transformation_list)):
        update_cam(sample, new_cam_pose=t)
        rays = sample['rays'].cuda()
        ts = sample['ts'].cuda()
        B = rays.shape[0]

        results = defaultdict(list)
        with torch.no_grad():
            for idx in range(0, B, config.chunk):
                results_chunk = compose_nerflet_bgnerf(
                    models=models,
                    embeddings=embeddings,
                    rays=rays[idx:idx+config.chunk],
                    ts=ts[idx:idx+config.chunk],
                    predict_label=config.predict_label,
                    num_classes=config.num_classes,
                    N_samples=config.N_samples,
                    use_disp=config.use_disp,
                    N_importance=config.N_importance,
                    white_back=False,
                    use_fine_nerf=config.use_fine_nerf
                )

                for k, v in results_chunk.items():
                    results[k] += [v]

            for k, v in results.items():
                results[k] = torch.cat(v, 0)

        # Rendering
        rows = []
        w, h = sample['img_wh']
        img_gt0 = sample0['rgbs']
        img_gt0 = img_gt0.view(h, w, 3)
        img_gt0 = np.clip(img_gt0.cpu().numpy(), 0, 1)
        img_gt0 = (img_gt0 * 255).astype(np.uint8)
        img_gt1 = sample1['rgbs']
        img_gt1 = img_gt1.view(h, w, 3)
        img_gt1 = np.clip(img_gt1.cpu().numpy(), 0, 1)
        img_gt1 = (img_gt1 * 255).astype(np.uint8)
        rows.append(np.concatenate([img_gt0, img_gt1], axis=1))

        label_gt0 = sample0['labels'].to(torch.long).cpu().numpy()
        label_map_gt0 = label_colors[label_gt0].reshape((h, w, 3))
        label_map_gt0 = (label_map_gt0 * 255).astype(np.uint8)
        label_gt1 = sample1['labels'].to(torch.long).cpu().numpy()
        label_map_gt1 = label_colors[label_gt1].reshape((h, w, 3))
        label_map_gt1 = (label_map_gt1 * 255).astype(np.uint8)
        rows.append(np.concatenate([label_map_gt0, label_map_gt1], axis=1))

        img_pred = results['comp_rgb'].view(h, w, 3).cpu().numpy()
        img_pred_ = (img_pred * 255).astype(np.uint8)

        label_pred = results['comp_label']
        label_pred = torch.argmax(label_pred, dim=1).to(torch.long).cpu().numpy()
        label_map_pred = label_colors[label_pred].reshape((h, w, 3))
        label_map_pred = (label_map_pred * 255).astype(np.uint8)
        rows.append(np.concatenate([img_pred_, label_map_pred], axis=1))

        placeholder = np.zeros((h, w, 3), dtype=np.uint8)
        nerflet_mask = results['nerflet_mask'].to(torch.int).cpu().numpy()
        nerflet_mask = np.repeat(nerflet_mask[..., np.newaxis], 3, axis=1)
        nerflet_mask = (nerflet_mask.reshape(h, w, 3) * 255).astype(np.uint8)
        rows.append(np.concatenate([placeholder, nerflet_mask], axis=1))

        res_img = np.concatenate(rows, axis=0)
        img_list.append(res_img)

        output_path = img_output_dir / f"view_{view_idx}.png"
        imageio.imwrite(output_path, res_img)

    gif_path = img_output_dir / 'result.gif'
    imageio.mimsave(gif_path, img_list, duration=0.25)
