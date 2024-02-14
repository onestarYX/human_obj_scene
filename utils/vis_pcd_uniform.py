import os
import torch
from collections import defaultdict

# models
from models.nerf import PosEmbedding
from models.nerflet import Nerflet
from models.rendering_nerflet import (
    get_nerflet_pred,
    get_input_from_rays
)
from models.model_utils import quaternions_to_rotation_matrices

from datasets.sitcom3D import Sitcom3DDataset
from datasets.blender import BlenderDataset
from datasets.replica import ReplicaDataset
from datasets.front import ThreeDFrontDataset
from datasets.kitti360 import Kitti360Dataset
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
import wandb

@torch.no_grad()
def inference_pts_occ(model, embeddings, xyz, chunk):
    """Do batched inference on 3D points using chunk."""
    occ_list = []

    for i in tqdm(range(0, len(xyz), chunk)):
        xyz_ = xyz[i:i + chunk]
        # Make dummy rays direction and ts
        rays_d = torch.zeros(len(xyz_), 1, 3).to(xyz.device)
        rays_d[:, 0, 0] = 1
        ts = torch.zeros(len(xyz_)).to(xyz.device).to(torch.int)
        pred = get_nerflet_pred(model, embeddings, xyz_, rays_d, ts)
        occ_list.append(pred['static_occ'].cpu())

    occ_list = torch.cat(occ_list, dim=0)
    return occ_list

@torch.no_grad()
def estimate_ellipsoid(model, embeddings, rays, ts, N_samples, use_disp, chunk):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    ellipsoid_keys = ['part_rotations', 'part_translations', 'part_scales']

    rays_ = rays[:chunk]
    ts_ = ts[:chunk]
    xyz, rays_d, z_vals = get_input_from_rays(rays_, N_samples, use_disp, perturb=0)
    pred = get_nerflet_pred(model, embeddings, xyz.cuda(), rays_d.cuda(), ts_.cuda())
    ellipsoid_pred = {}
    for k in ellipsoid_keys:
        ellipsoid_pred[k] = torch.clone(pred[k]).cpu()
    return ellipsoid_pred

def visualize_nerflets(config, dataset, nerflet, embeddings, output_dir):
    # Prepare colors for each part
    np.random.seed(19)
    part_colors = np.random.rand(config.num_parts, 3)
    writer = imageio.get_writer(output_dir / 'out.mp4', fps=10)
    # Collect a bunch of cameras
    cams = []
    for i in range(0, len(dataset), 2):
        sample = dataset[i]
        if config.dataset_name == 'sitcom3D':
            w, h = sample['img_wh']
        elif config.dataset_name == 'blender':
            w, h = config.img_wh
        elif config.dataset_name in ['replica', '3dfront', 'kitti360']:
            w, h = dataset.img_wh

        # Get ray directions (in world coordinate)
        rays = sample['rays'].reshape(h, w, -1)
        corner_ray0 = rays[0, 0]
        corner_ray1 = rays[0, w - 1]
        corner_ray2 = rays[h - 1, 0]
        corner_ray3 = rays[h - 1, w - 1]
        cam = {'rays_o': corner_ray0[:3].numpy(),
               'rays_d': [corner_ray0[3:6].numpy(),
                          corner_ray1[3:6].numpy(),
                          corner_ray2[3:6].numpy(),
                          corner_ray3[3:6].numpy()
                          ]}
        cams.append(cam)

    # Render cameras
    geo = []
    for cam in cams:
        rays_o = cam['rays_o']
        cam_center = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        cam_center.paint_uniform_color([0.7, 0.1, 0.1])
        cam_center.translate(rays_o)
        geo.append(cam_center)

        rays_d = cam['rays_d']
        ray_end_points = [rays_o,
                          rays_o + rays_d[0],
                          rays_o + rays_d[1],
                          rays_o + rays_d[2],
                          rays_o + rays_d[3]]
        rays_connections = [[0, 1], [0, 2], [0, 3], [0, 4]]
        rays_colors = [[0, 0, 1] for _ in range(len(rays_connections))]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(ray_end_points)
        line_set.lines = o3d.utility.Vector2iVector(rays_connections)
        line_set.colors = o3d.utility.Vector3dVector(rays_colors)
        geo.append(line_set)

    # # Render nerflet ellipsoids
    # sample = dataset[0]
    # rays = sample['rays']
    # if 'ts2' in sample:
    #     ts = sample['ts2']
    # else:
    #     ts = sample['ts']
    # obj_mask = sample['obj_mask']
    # # TODO: we didn't use obj_mask here
    # # xyz, rays_d, z_vals = get_input_from_rays(rays, config.N_samples, config.use_disp, perturb=0)
    # # with torch.no_grad():
    # #     pred = get_nerflet_pred(nerflet, embeddings, xyz[:1000].cuda(), rays_d[:1000].cuda(), ts[:1000].cuda())
    # ellipsoid_pred = estimate_ellipsoid(nerflet, embeddings, rays, ts, config.N_samples, config.use_disp, config.chunk)
    # rotations = quaternions_to_rotation_matrices(ellipsoid_pred['part_rotations'])
    # rotations = torch.linalg.inv(rotations)
    # translations = ellipsoid_pred['part_translations']
    # scales = ellipsoid_pred['part_scales']
    # for i in range(len(rotations)):
    #     rot = rotations[i]
    #     trans = translations[i]
    #     scale = scales[i]
    #     ell_end_points = torch.tensor([[-scale[0], 0, 0], [scale[0], 0, 0],
    #                                    [0, -scale[1], 0], [0, scale[1], 0],
    #                                    [0, 0, -scale[2]], [0, 0, scale[2]]
    #                                    ])
    #     ell_end_points = (rot @ ell_end_points.T).T
    #     ell_end_points += trans
    #     ell_end_points = ell_end_points.numpy()
    #
    #     ell_color = part_colors[i]
    #     ell_center = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    #     ell_center.paint_uniform_color(ell_color)
    #     ell_center.translate(trans.numpy())
    #     geo.append(ell_center)
    #
    #     ell_lines = [[0, 1], [2, 3], [4, 5]]
    #     ell_lines_color = [ell_color for _ in range(len(ell_lines))]
    #     line_set = o3d.geometry.LineSet()
    #     line_set.points = o3d.utility.Vector3dVector(ell_end_points)
    #     line_set.lines = o3d.utility.Vector2iVector(ell_lines)
    #     line_set.colors = o3d.utility.Vector3dVector(ell_lines_color)
    #     geo.append(line_set)

    # Uniformly sample points in the 3D space and make inference
    N_samples = config.N_samples
    space_size = config.scene_bound
    multiplier = 2
    xs = torch.linspace(-space_size, space_size, steps=N_samples * multiplier)
    ys = torch.linspace(-space_size, space_size, steps=N_samples * multiplier)
    zs = torch.linspace(-space_size, space_size, steps=N_samples * multiplier)
    xyz = torch.cartesian_prod(xs, ys, zs)
    xyz = xyz.reshape(-1, N_samples, 3).cuda()
    results = inference_pts_occ(nerflet, embeddings, xyz, config.chunk)
    # if config.predict_density:
    #     static_density = results
    #     # TODO: this delta shouldn't be using cam's z_vals because we are in the local nerf coordinate?
    #     deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples_-1)
    #     delta_inf = 1e2 * torch.ones_like(deltas[:, :1])  # (N_rays, 1) the last delta is infinity
    #     deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)
    #     deltas = deltas.unsqueeze(-1).expand(-1, -1, nerflet.M)
    #     noise = torch.randn_like(static_density, device=deltas.device)
    #     results = 1 - torch.exp(-deltas * torch.relu(static_density + noise))

    xyz = xyz.reshape(-1, 3).cpu()
    results = results.reshape(-1, config.num_parts)

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

    if args.view_in_3d:
        o3d.visualization.draw_geometries(geo)
    else:
        # sample_view = dataset[0]
        # c2w = sample_view['c2w']

        vis = o3d.visualization.Visualizer()
        # vis.create_window(width=w * config.img_downscale, height=h * config.img_downscale, visible=True)
        vis.create_window()
        for geo_ in geo:
            vis.add_geometry(geo_)

        R = geo[0].get_rotation_matrix_from_xyz((np.pi / 50, 0, 0))
        for i in tqdm(range(100)):
            # ctr = vis.get_view_control()
            # # cam_params = ctr.convert_to_pinhole_camera_parameters()
            # # cam_params.extrinsic = np.concatenate((c2w.numpy(), np.array([[0, 0, 0, 1]])), axis=0)
            # # K_ = dataset.K
            # # K_[0, 2] -= 0.5
            # # K_[1, 2] -= 0.5
            # # cam_params.intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, dataset.K)
            # # ctr.convert_from_pinhole_camera_parameters(cam_params)
            # ctr.rotate(10 * i, 10 * i)

            for geo_ in geo:
                geo_.rotate(R, center=(0, 0, 0))
                vis.update_geometry(geo_)
            vis.poll_events()
            vis.update_renderer()

            image = vis.capture_screen_float_buffer(do_render=False)
            image = (np.asarray(image) * 255).astype(np.uint8)

            out_img_path = output_dir / f"{i}.png"
            # plt.imsave(out_img_path, image)
            writer.append_data(image)
        vis.destroy_window()

    writer.close()

def get_wandb_point_scene(config, dataset, nerflet, embeddings):
    # Prepare colors for each part
    np.random.seed(19)
    part_colors = np.random.randint(low=0, high=256, size=(config.num_parts, 3)).astype(np.float32)

    # Collect a bunch of cameras
    cams = []
    for i in range(0, len(dataset), 2):
        sample = dataset[i]
        if config.dataset_name == 'sitcom3D':
            w, h = sample['img_wh']
        elif config.dataset_name == 'blender':
            w, h = config.img_wh
        elif config.dataset_name in ['replica', '3dfront', 'kitti360']:
            w, h = dataset.img_wh

        # Get ray directions (in world coordinate)
        rays = sample['rays'].reshape(h, w, -1)
        corner_ray0 = rays[0, 0]
        corner_ray1 = rays[0, w - 1]
        corner_ray2 = rays[h - 1, 0]
        corner_ray3 = rays[h - 1, w - 1]
        cam = {'rays_o': corner_ray0[:3].numpy(),
               'rays_d': [corner_ray0[3:6].numpy(),
                          corner_ray1[3:6].numpy(),
                          corner_ray2[3:6].numpy(),
                          corner_ray3[3:6].numpy()
                          ]}
        cams.append(cam)

    # Render cameras
    vectors = []
    for cam in cams:
        rays_o = cam['rays_o']
        for rays_d in cam['rays_d']:
            vec_start = rays_o.tolist()
            vec_end = (rays_o + rays_d).tolist()
            vec = {"start": vec_start, "end": vec_end}
            vectors.append(vec)

    # Uniformly sample points in the 3D space and make inference
    N_samples = config.N_samples
    space_size = config.scene_bound
    multiplier = 2
    xs = torch.linspace(-space_size, space_size, steps=N_samples * multiplier)
    ys = torch.linspace(-space_size, space_size, steps=N_samples * multiplier)
    zs = torch.linspace(-space_size, space_size, steps=N_samples * multiplier)
    xyz = torch.cartesian_prod(xs, ys, zs)
    xyz = xyz.reshape(-1, N_samples, 3).cuda()
    results = inference_pts_occ(nerflet, embeddings, xyz, config.chunk)

    xyz = xyz.reshape(-1, 3).cpu()
    results = results.reshape(-1, config.num_parts)

    occ_threshold = 0.5
    pt_max_occ, pt_association = results.max(dim=-1)
    pt_occupied_mask = pt_max_occ > occ_threshold
    pt_to_show = xyz[pt_occupied_mask]
    pt_to_show_association = pt_association[pt_occupied_mask]

    all_pts_to_add = []
    for idx in range(config.num_parts):
        pt_part_mask = pt_to_show_association == idx
        pt_part = pt_to_show[pt_part_mask]
        if len(pt_part) == 0:
            continue
        part_color = part_colors[idx]
        colors = np.tile(part_color, (len(pt_part), 1))
        pt_to_add = np.concatenate((pt_part.numpy(), colors), axis=1)
        all_pts_to_add.append(pt_to_add)

    all_pts_to_add = np.concatenate(all_pts_to_add, axis=0)
    res = wandb.Object3D({
        "type": "lidar/beta",
        "points": all_pts_to_add,
        "vectors": np.array(vectors),
    })
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results/cam_ellipsoids')
    parser.add_argument('--use_ckpt', type=str)
    parser.add_argument('--split', type=str, default='test_train')
    parser.add_argument('--view_in_3d', action='store_true', default=False)
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
        kwargs = {}
        kwargs.update({'environment_dir': config.environment_dir,
                      'near_far_version': config.near_far_version})
        # kwargs['img_downscale'] = config.img_downscale
        kwargs['val_num'] = 5
        kwargs['use_cache'] = config.use_cache
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
                                     img_downscale=config.img_downscale, split=args.split)
    elif config.dataset_name == 'kitti360':
        dataset = Kitti360Dataset(root_dir=config.environment_dir, split=args.split,
                                  img_downscale=config.img_downscale,
                                  near=config.near, far=config.far, scene_bound=config.scene_bound)
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
    nerflet = Nerflet(D=config.num_hidden_layers, W=config.dim_hidden_layers, skips=config.skip_layers,
                      N_emb_xyz=config.N_emb_xyz, N_emb_dir=config.N_emb_dir,
                      encode_a=config.encode_a, encode_t=config.encode_t, predict_label=config.predict_label,
                      num_classes=config.num_classes, M=config.num_parts,
                      disable_ellipsoid=config.disable_ellipsoid,
                      scale_min=config.scale_min, scale_max=config.scale_max,
                      use_spread_out_bias=config.use_spread_out_bias, bbox=bbox,
                      label_only=config.label_only, disable_tf=config.disable_tf,
                      sharpness=config.sharpness, predict_density=config.predict_density).cuda()
    load_ckpt(nerflet, ckpt_path, model_name='nerflet')
    models = {'nerflet': nerflet}

    # visualize_nerflets(config, dataset, nerflet, embeddings, output_dir)
    get_wandb_point_scene(config, dataset, nerflet, embeddings)