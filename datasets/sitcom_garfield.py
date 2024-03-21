import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pickle
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm
from pathlib import Path
import json

from .ray_utils import *
from .colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary

import cv2
import kornia
import copy

# from sitcoms3D.utils.io import load_from_json
from transformers import pipeline


def width_height_from_intr(K):
    cx, cy = K[0, 2], K[1, 2]
    W = int(2 * cx)
    H = int(2 * cy)
    return W, H


def near_far_from_points(xyz_world_h, w2c):
    """
    """
    xyz_cam = (xyz_world_h @ w2c.T)[:, :3]  # xyz in the ith cam coordinate
    xyz_cam = xyz_cam[xyz_cam[:, 2] > 0]  # filter out points that lie behind the cam
    near = np.percentile(xyz_cam[:, 2], 0.1)
    far = np.percentile(xyz_cam[:, 2], 99.9)
    return near, far


def torch_ray_intersect_aabb(rays_o, rays_d, aabb):
    """
    Args:
        rays_o (torch.tensor): (batch_size, 3)
        rays_d (torch.tensor): (batch_size, 3)
        aabb (torch.tensor): (2, 3)
            This is [min point (x,y,z), max point (x,y,z)]
    """

    # avoid divide by zero
    dir_fraction = 1.0 / (rays_d + 1e-6)

    # x
    t1 = (aabb[0, 0] - rays_o[:, 0:1]) * dir_fraction[:, 0:1]
    t2 = (aabb[1, 0] - rays_o[:, 0:1]) * dir_fraction[:, 0:1]
    # y
    t3 = (aabb[0, 1] - rays_o[:, 1:2]) * dir_fraction[:, 1:2]
    t4 = (aabb[1, 1] - rays_o[:, 1:2]) * dir_fraction[:, 1:2]
    # z
    t5 = (aabb[0, 2] - rays_o[:, 2:3]) * dir_fraction[:, 2:3]
    t6 = (aabb[1, 2] - rays_o[:, 2:3]) * dir_fraction[:, 2:3]

    nears = torch.max(torch.cat([
        torch.minimum(t1, t2),
        torch.minimum(t3, t4),
        torch.minimum(t5, t6)], dim=1),
        dim=1).values
    fars = torch.min(torch.cat([
        torch.maximum(t1, t2),
        torch.maximum(t3, t4),
        torch.maximum(t5, t6)], dim=1),
        dim=1).values

    # TODO(ethan): handle two cases
    # fars < 0: means the ray is behind the camera
    # nears > fars: means no intersection
    # currently going to assert the valid cases
    # assert torch.all(fars > 0)
    # assert torch.all(fars > nears)
    # if not torch.all(fars > 0) or not torch.all(fars > nears):
    #     print("OUT OF BOUNDS!\n\n")

    mask = (fars > nears).float() * (fars > 0).float()
    # nears, fars = nears[mask], fars[mask]
    # set nears to be zero
    nears[nears < 0.0] = 0.0

    nears = nears.unsqueeze(-1)
    fars = fars.unsqueeze(-1)
    mask = mask.unsqueeze(-1)
    return nears, fars, mask


class RenderDataset(Dataset):
    def __init__(self, *args, **kwargs):
        """
        """
        super().__init__()
        pass

    def get_bbox_pointcloudT(self):
        """Get the bounding box and the point cloud transformation.
        Here we want to bound the scene with a tight bbox.
        The bbox_params.json file comes from three.js editor online.
        """
        filename = os.path.join(self.environment_dir, 'threejs.json')
        assert os.path.exists(filename)

        with open(filename, 'r') as f:
            data = json.load(f)

        # point cloud transformation
        # pointcloudT = np.array(data['object']['children'][0]['children'][0]["matrix"]).reshape(4, 4).T
        pointcloudT = np.array(data['object']['children'][0]["matrix"]).reshape(4, 4).T
        assert pointcloudT[3, 3] == 1.0

        # bbox transformation
        bbox_T = np.array(data['object']['children'][1]["matrix"]).reshape(4, 4).T
        w, h, d = data["geometries"][1]["width"], data["geometries"][1]["height"], data["geometries"][1]["depth"]
        temp = np.array([w, h, d]) / 2.0
        bbox = np.array([-temp, temp])
        bbox = np.concatenate([bbox, np.ones_like(bbox[:, 0:1])], axis=1)
        bbox = (bbox_T @ bbox.T).T[:, 0:3]

        return bbox, pointcloudT

    def get_img(self, id_, img_downscale=None):
        img = Image.open(os.path.join(self.environment_dir, 'images',
                                      self.id_to_image_path[id_])).convert('RGB')
        img_w, img_h = img.size
        if not img_downscale:
            img_downscale = self.img_downscale
        if img_downscale > 1:
            img_w = img_w // img_downscale
            img_h = img_h // img_downscale
            img = img.resize((img_w, img_h), Image.LANCZOS)
        img = np.array(img)
        return img

    def get_pose(self, id_, homogeneous=False):
        pose = copy.deepcopy(self.poses_dict[id_])
        if homogeneous:
            pose = np.concatenate([pose, np.zeros_like(pose[:1])], -2)
            pose[3, 3] = 1
        return pose

    def get_HW(self, id_):
        K = copy.deepcopy(self.Ks[id_])
        H, W = round(K[1, 2] * 2.0), round(K[0, 2] * 2.0)
        return H, W

    def get_K(self, id_):
        K = copy.deepcopy(self.Ks[id_])
        return K

    def get_human_mask(self, id_):
        # human mask
        # TODO(ethan): confirm that this works!
        panoptic = np.array(Image.open(os.path.join(self.environment_dir, 'segmentations', 'thing',
                                                '%s.png' % self.id_to_image_path[id_][:-4])))
        mask = np.zeros_like(panoptic)
        mask[panoptic == 1] = 1 # 1 is the person class
        K = self.get_K(id_)
        img_w, img_h = width_height_from_intr(K)
        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        return mask

    def remap_panoptic_labels(self):
        json_path = Path('data/sparse_reconstruction_and_nerf_data/HIMYM-red_apartment/panoptic_classes.json')
        new_cls_list = []
        with open(json_path) as f:
            cls_meta = json.load(f)
            new_cls_list.extend(cls_meta['stuff'])
            new_cls_list.extend(cls_meta['thing'])
        new_cls_list.remove('thing')
        new_cls_list.remove('stuff')
        redundant_cls_list = []
        for cls in new_cls_list:
            if 'floor-' in cls or 'window-' in cls or 'wall-' in cls:
                redundant_cls_list.append(cls)
        for redudant_cls in redundant_cls_list:
            new_cls_list.remove(redudant_cls)
        # print(f"Found {len(new_cls_list)} panoptic classes")

        # Remapping original stuff and thing labels to new class list
        stuff_cls_list = cls_meta['stuff']
        thing_cls_list = cls_meta['thing']
        stuff_cls_mapping = dict()
        thing_cls_mapping = dict()
        for i, cls in enumerate(stuff_cls_list):
            if cls == 'thing':
                continue
            if 'floor' in cls:
                cls = 'floor'
            if 'window' in cls:
                cls = 'window'
            if 'wall' in cls:
                cls = 'wall'
            tar_idx = new_cls_list.index(cls)
            stuff_cls_mapping[i] = tar_idx
        for i, cls in enumerate(thing_cls_list):
            if cls == 'stuff':
                continue
            tar_idx = new_cls_list.index(cls)
            thing_cls_mapping[i] = tar_idx
        return stuff_cls_mapping, thing_cls_mapping

    def get_panoptic_labels(self, id_):
        # thing: 81, stuff: 54
        thing = np.array(Image.open(os.path.join(self.environment_dir, 'segmentations', 'thing',
                                                '%s.png' % self.id_to_image_path[id_][:-4]))).astype(int)
        stuff = np.array(Image.open(os.path.join(self.environment_dir, 'segmentations', 'stuff',
                                                '%s.png' % self.id_to_image_path[id_][:-4]))).astype(int)
        stuff_cls_mapping, thing_cls_mapping = self.remap_panoptic_labels()

        for i in range(thing.shape[0]):
            for j in range(thing.shape[1]):
                if thing[i, j] == 0:
                    continue
                thing[i, j] = thing_cls_mapping[thing[i, j]]

        for i in range(stuff.shape[0]):
            for j in range(stuff.shape[1]):
                if stuff[i, j] == 0:
                    continue
                stuff[i, j] = stuff_cls_mapping[stuff[i, j]]

        panoptic = thing
        obj_mask = np.zeros_like(panoptic)
        for i in range(panoptic.shape[0]):
            for j in range(panoptic.shape[1]):
                if panoptic[i, j] == 0:
                    panoptic[i, j] = stuff[i, j]
                else:
                    obj_mask[i, j] = 1

        K = self.get_K(id_)
        img_w, img_h = width_height_from_intr(K)
        panoptic = cv2.resize(panoptic, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        obj_mask = cv2.resize(obj_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        return panoptic, obj_mask

    def get_image_paths(self):
        """
        """

        imgs = []
        imdata = read_images_binary(os.path.join(self.environment_dir, 'colmap', 'images.bin'))
        image_paths = set([v.name for v in imdata.values()])
        filter_image_paths = set(self.image_filenames)
        if len(filter_image_paths) > 0:
            image_paths = list(image_paths.intersection(filter_image_paths))
            assert len(image_paths) == len(filter_image_paths)

        for img_path in image_paths:
            temp = os.path.join(self.environment_dir, 'images', img_path)
            assert os.path.exists(temp)
        image_paths = list(image_paths)
        image_paths = sorted(image_paths)
        if self.num_limit != -1:
            image_paths = image_paths[:self.num_limit]

        return image_paths

    def get_fl(self, id_):
        """Returns the focal length.
        """
        K = self.get_K(id_)
        assert K[0, 0] == K[1, 1]
        return float(K[0, 0])

    def get_fov(self, id_):
        fl = self.get_fl(id_)
        H, W = self.get_HW(id_)
        fov = 2 * np.arctan(float(H) / (2.0 * fl))
        fov *= 180.0 / np.pi
        return fov

    def get_aa(self, id_):
        """Return the angle axis (aa) representation for the c2w.
        """
        c2w = self.get_pose(id_)
        rotation_matrix = torch.from_numpy(c2w[:3, :3]).reshape(1, 3, 3).clone()
        aa = kornia.rotation_matrix_to_angle_axis(rotation_matrix)[0].numpy()
        return aa

    def get_viewdir(self, id_):
        """
        """
        c2w = self.get_pose(id_)
        viewdir = np.array((0, 0, -1)) @ c2w[:3, :3].T
        return viewdir

    def get_xyz(self, id_):
        c2w = self.get_pose(id_)
        xyz = c2w[:3, 3]
        return xyz


def create_pixel_mask_array(masks: torch.Tensor):
    """
    Create per-pixel data structure for grouping supervision.
    pixel_mask_array[x, y] = [m1, m2, ...] means that pixel (x, y) belongs to masks m1, m2, ...
    where Area(m1) < Area(m2) < ... (sorted by area).
    """
    max_masks = masks.sum(dim=0).max().item()
    image_shape = masks.shape[1:]
    pixel_mask_array = torch.full(
        (max_masks, image_shape[0], image_shape[1]), -1, dtype=torch.int
    ).to(masks.device)

    for m, mask in enumerate(masks):
        mask_clone = mask.clone()
        for i in range(max_masks):
            free = pixel_mask_array[i] == -1
            masked_area = mask_clone == 1
            right_index = free & masked_area
            if len(pixel_mask_array[i][right_index]) != 0:
                pixel_mask_array[i][right_index] = m
            mask_clone[right_index] = 0
    pixel_mask_array = pixel_mask_array.permute(1, 2, 0)

    return pixel_mask_array


class SitcomSAMDataset(RenderDataset):
    def __init__(self, environment_dir, split='train', img_downscale=1,
                 val_num=1, use_cache=False, near_far_version="cam", read_points=True, num_limit=-1, near=None,
                 bs=2048, num_rays_per_img=256):
        """
        img_downscale: how much scale to downsample the training images.
                       The original image sizes are around 500~100, so value of 1 or 2
                       are recommended.
                       ATTENTION! Value of 1 will consume large CPU memory,
                       about 40G for brandenburg gate.
        val_num: number of val images (used for multigpu, validate same image for all gpus)
        use_cache: during data preparation, use precomputed rays (useful to accelerate
                   data loading, especially for multigpu!)
        """
        super().__init__()

        self.environment_dir = environment_dir
        self.image_filenames = os.listdir(os.path.join(self.environment_dir, 'images'))

        self.read_points = read_points

        # make nerf folder if it doesn't exist
        self.cache_dir = os.path.join(self.environment_dir, f'cache/{near_far_version}/')

        self.split = split
        assert img_downscale >= 1, 'image can only be downsampled, please set img_downscale>=1!'
        self.img_downscale = img_downscale
#        if split == 'val': # image downscale=1 will cause OOM in val mode
#            self.img_downscale = max(2, self.img_downscale)
        self.val_num = max(1, val_num)  # at least 1
        self.use_cache = use_cache
        self.near_far_version = near_far_version
        self.near_overwrite = near
        self.define_transforms()
        self.sam_model = None
        self.num_limit = num_limit
        # print(f"Using near_far_version: {self.near_far_version}")

        self.bs = bs
        self.num_rays_per_img = num_rays_per_img
        assert self.bs % self.num_rays_per_img == 0, "batch_size cannot be divided by num_rays_per_img!"
        self.read_meta()
        self.white_back = False

    def get_nears_fars_from_rays_or_cam(self, rays_o, rays_d, c2w=None):
        """
        """
        if self.near_far_version == "box":
            nears, fars, ray_mask = torch_ray_intersect_aabb(rays_o, rays_d, aabb=self.bbox)
        elif self.near_far_version == "cam":
            assert not isinstance(c2w, type(None))
            xyz_world_h = np.concatenate([self.xyz_world, np.ones((len(self.xyz_world), 1))], -1)
            c2w_h = np.concatenate([c2w.cpu().numpy(), np.zeros((1, 4))], -2)
            c2w_h[3, 3] = 1
            c2w_h[:, 1:3] *= -1
            w2c = np.linalg.inv(c2w_h)
            near, far = near_far_from_points(xyz_world_h, w2c)
            nears = near * torch.ones_like(rays_o[:, 0:1])
            fars = far * torch.ones_like(rays_o[:, 0:1])
            ray_mask = torch.ones_like(fars)
        else:
            raise NotImplementedError("")
        if self.near_overwrite is not None:
            nears[:] = self.near_overwrite
        return nears, fars, ray_mask

    def read_meta(self):

        # Step 1. load image paths
        # Attention! The 'id' column in the tsv is BROKEN, don't use it!!!!
        # Instead, read the id from images.bin using image file name!
        if self.use_cache:
            with open(os.path.join(self.cache_dir, 'img_ids.pkl'), 'rb') as f:
                self.img_ids = pickle.load(f)
            with open(os.path.join(self.cache_dir, 'id_to_image_path.pkl'), 'rb') as f:
                self.id_to_image_path = pickle.load(f)
        else:
            image_paths = self.get_image_paths()
            train_split = int(len(image_paths) * 0.8)
            if self.split == 'train' or self.split == 'test_train':
                image_paths = image_paths[:train_split]
            elif self.split == 'val':
                image_paths = image_paths[train_split:]
            self.img_paths = image_paths
            self.img_paths = [Path(img_path) for img_path in self.img_paths]

            imdata = read_images_binary(os.path.join(self.environment_dir, 'colmap', 'images.bin'))
            img_path_to_id = {}
            for v in imdata.values():
                img_path_to_id[v.name] = v.id
            self.img_ids = []
            self.id_to_image_path = {}  # {id: filename}
            for image_path in image_paths:
                id_ = img_path_to_id[image_path]
                self.id_to_image_path[id_] = image_path
                self.img_ids += [id_]

        self.image_path_to_id = {}
        for id_, image_path in self.id_to_image_path.items():
            self.image_path_to_id[image_path] = id_

        self.id_to_idx = {}
        for idx, id_ in enumerate(self.img_ids):
            self.id_to_idx[id_] = idx

        # Step 2: read and rescale camera intrinsics
        if self.use_cache:
            raise NotImplementedError
        else:
            self.Ks = {}  # {id: K}
            camdata = read_cameras_binary(os.path.join(self.environment_dir, 'colmap', 'cameras.bin'))
            for id_ in self.img_ids:
                K = np.zeros((3, 3), dtype=np.float32)
                cam = camdata[id_]
                assert len(cam.params) == 3
                img_w, img_h = int(cam.params[1] * 2), int(cam.params[2] * 2)
                img_w_, img_h_ = img_w // self.img_downscale, img_h // self.img_downscale
                self.img_hw = (img_h_, img_w_)
                K[0, 0] = cam.params[0] * img_w_ / img_w  # fx
                K[1, 1] = cam.params[0] * img_h_ / img_h  # fy
                K[0, 2] = cam.params[1] * img_w_ / img_w  # cx
                K[1, 2] = cam.params[2] * img_h_ / img_h  # cy
                K[2, 2] = 1
                assert K[0, 0] == K[1, 1], "maybe check if img_downscale is too small?"
                self.Ks[id_] = K

        # Step 3: read c2w poses (of the images in tsv file only) and correct the order
        if self.use_cache:
            self.poses = np.load(os.path.join(self.cache_dir, 'poses.npy'))
        else:
            w2c_mats = []
            bottom = np.array([0, 0, 0, 1.]).reshape(1, 4)
            for id_ in self.img_ids:
                im = imdata[id_]
                R = im.qvec2rotmat()
                t = im.tvec.reshape(3, 1)
                w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
            w2c_mats = np.stack(w2c_mats, 0)  # (N_images, 4, 4)
            self.poses = np.linalg.inv(w2c_mats)[:, :3]  # (N_images, 3, 4)
            # Original poses has rotation in form "right down front", change to "right up back"
            self.poses[..., 1:3] *= -1

        # Step 4: correct scale
        if self.use_cache:
            # self.xyz_world = np.load(os.path.join(self.cache_dir, 'xyz_world.npy'))
            # self.rgb_world = np.load(os.path.join(self.cache_dir, 'rgb_world.npy'))
            self.scale_factor = float(np.load(os.path.join(self.cache_dir, 'scale_factor.npy')))
            self.bbox = np.load(os.path.join(self.cache_dir, 'bbox.npy'))
            self.pointcloudT = np.load(os.path.join(self.cache_dir, 'pointcloudT.npy'))
        else:
            if self.read_points:
                pts3d = read_points3d_binary(os.path.join(self.environment_dir, 'colmap', 'points3D.bin'))
                self.xyz_world = np.array([pts3d[p_id].xyz for p_id in pts3d])
                self.rgb_world = np.array([pts3d[p_id].rgb for p_id in pts3d])
                xyz_world_h = np.concatenate([self.xyz_world, np.ones((len(self.xyz_world), 1))], -1)

            if self.near_far_version == "box":
                self.bbox, self.pointcloudT = self.get_bbox_pointcloudT()
                self.poses = np.concatenate([self.poses, np.zeros_like(self.poses[:, 0:1, :])], axis=1)
                self.poses[:, 3, 3] = 1.0  # (N_images, 4, 4)
                self.poses = self.pointcloudT @ self.poses
                self.poses = self.poses[:, :3]
                if self.read_points:
                    self.xyz_world = (xyz_world_h @ self.pointcloudT.T)[:, :3]
                temp = self.bbox[0] - self.bbox[1]
                max_far = np.sqrt(np.dot(temp.T, temp))
            elif self.near_far_version == "cam":
                camera_translations = self.poses[:, :3, 3]
                all_points = np.concatenate([self.xyz_world, camera_translations], axis=0)
                vol_point_min = np.min(all_points, axis=0)  # volume point minimum
                vol_point_max = np.max(all_points, axis=0)  # volume point maximum
                self.bbox = np.array([vol_point_min, vol_point_max])
                # Compute near and far bounds for each image individually
                nears, fars = {}, {}  # {id_: distance}
                for i, id_ in enumerate(self.img_ids):
                    xyz_cam_i = (xyz_world_h @ w2c_mats[i].T)[:, :3]  # xyz in the ith cam coordinate
                    xyz_cam_i = xyz_cam_i[xyz_cam_i[:, 2] > 0]  # filter out points that lie behind the cam
                    nears[id_] = np.percentile(xyz_cam_i[:, 2], 0.1)
                    fars[id_] = np.percentile(xyz_cam_i[:, 2], 99.9)
                max_far = np.fromiter(fars.values(), np.float32).max()

            print(f"max_far: {max_far}")
            self.scale_factor = max_far / 5  # so that the max far is scaled to 5   #TODO: figure out what this scale_factor is doing
            self.poses[..., 3] /= self.scale_factor
            self.xyz_world /= self.scale_factor
            self.bbox /= self.scale_factor
        self.poses_dict = {id_: self.poses[i] for i, id_ in enumerate(self.img_ids)}

        # Get SAM masks
        sam_out_dir = Path(self.environment_dir) / 'sam_masks'
        sam_out_dir.mkdir(exist_ok=True)
        sam_out_path = "train.pkl" if self.split in {'train', 'test_train'} else 'val.pkl'
        sam_out_path = sam_out_dir / sam_out_path
        self.sam_dict = {}
        if sam_out_path.exists():
            with open(sam_out_path, 'rb') as handle:
                self.sam_dict.update(pickle.load(handle))
        else:
            print("Generating SAM masks......")
            if self.sam_model is None:
                self.sam_model = pipeline("mask-generation", model="facebook/sam-vit-huge", device=torch.device('cuda'))
            for id_ in tqdm(self.img_ids):
                img = self.get_img(id_)
                img = Image.fromarray(img)
                masks = self.sam_model(img, points_per_side=32, pred_iou_thresh=0.90, stability_score_thresh=0.90)
                masks = masks['masks']
                masks = sorted(masks, key=lambda x: x.sum())
                self.sam_dict[id_] = masks
            with open(sam_out_path, 'wb') as handle:
                pickle.dump(self.sam_dict, handle)


        self.all_img_ids = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_valid_ray_masks = []  # rays that are inside the estimated bbox. (almost 1 for every ray)
        self.all_valid_pixel_indices = []
        self.all_labels = []
        self.all_sam_masks = []

        for id_ in tqdm(self.img_ids):
            self.all_img_ids.append(id_)
            c2w = torch.FloatTensor(self.get_pose(id_))
            # Get image
            img = self.get_img(id_)
            img_h, img_w, _ = img.shape
            img = self.transform(img).permute(1, 2, 0)  # (h, w, 3)
            self.all_rgbs.append(img)

            # Get labels predicted by detectron
            label, _ = self.get_panoptic_labels(id_)
            label = label.astype(float)
            label = self.transform(label).permute(1, 2, 0).to(torch.long)
            self.all_labels.append(label)

            # Get rays
            directions = get_ray_directions(img_h, img_w, self.get_K(id_))
            rays_o, rays_d = get_rays(directions, c2w)  # (h*w, 3)
            rays_t = self.id_to_idx[id_] * torch.ones(len(rays_o), 1)
            nears, fars, inbbox_ray_mask = self.get_nears_fars_from_rays_or_cam(rays_o, rays_d, c2w=c2w)
            rays = torch.cat([rays_o, rays_d,
                              nears,
                              fars,
                              rays_t],
                             1) # (h*w, 9)
            rays = rays.reshape(img_h, img_w, -1)
            self.all_rays += [rays]

            # Get valid masks
            inbbox_ray_mask = inbbox_ray_mask.reshape(img_h, img_w, 1).to(torch.bool)
            human_mask = self.get_human_mask(id_).astype(float)
            human_mask = self.transform(human_mask).permute(1, 2, 0).to(torch.bool)
            valid_ray_mask = ~human_mask & inbbox_ray_mask
            self.all_valid_ray_masks += [valid_ray_mask]
            valid_pixel_indices = valid_ray_mask.squeeze(-1).nonzero()
            self.all_valid_pixel_indices.append(valid_pixel_indices)

            # Get SAM masks
            sam_masks = self.sam_dict[id_]
            sam_masks = torch.tensor(np.stack(sam_masks, axis=0))
            sam_masks = create_pixel_mask_array(sam_masks).long()
            self.all_sam_masks.append(sam_masks)

        self.all_rays = torch.stack(self.all_rays, 0)  # (N_images, h, w, 9)
        self.all_rgbs = torch.stack(self.all_rgbs, 0)  # (N_images, h, w, 3)
        self.all_valid_ray_masks = torch.stack(self.all_valid_ray_masks, 0)
        self.all_labels = torch.stack(self.all_labels, 0)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            N, h, w = self.all_rays.shape[:-1]
            return N * h * w // self.bs
        if self.split == 'test_train':
            return len(self.img_ids)
        if self.split == 'val':
            return len(self.img_ids)

    def __getitem__(self, idx):
        if self.split == 'train':  # use data in the buffers
            num_images = self.bs // self.num_rays_per_img
            img_indices = torch.randint(low=0, high=len(self.img_ids), size=(num_images,))
            pixel_indices = []
            for img_idx in img_indices:
                valid_pixel_indices = self.all_valid_pixel_indices[img_idx]
                rand_indices = torch.randint(low=0, high=len(valid_pixel_indices), size=(self.num_rays_per_img,))
                cur_pixel_indices = valid_pixel_indices[rand_indices]
                temp = torch.ones(self.num_rays_per_img, 1) * img_idx
                cur_pixel_indices = torch.cat((temp.long(), cur_pixel_indices), dim=1)
                pixel_indices.append(cur_pixel_indices)
            pixel_indices = torch.cat(pixel_indices, dim=0)
            ids = pixel_indices[:, 0]
            xs = pixel_indices[:, 1]
            ys = pixel_indices[:, 2]

            # pixel_indices = (torch.rand((self.bs, 3)) * torch.tensor([len(self.img_ids), *self.img_hw])).long()
            # pixel_indices[:, 0] = img_indices.repeat_interleave(self.num_rays_per_img)
            sample = {'rays': self.all_rays[ids, xs, ys, :8],
                      'ts': self.all_rays[ids, xs, ys, 8].long(),
                      'rgbs': self.all_rgbs[ids, xs, ys],
                      'labels': self.all_labels[ids, xs, ys],
                      'ray_mask': self.all_valid_ray_masks[ids, xs, ys]}
        elif self.split in ['val', 'test_train']:
            sample = {}
            sample['rays'] = self.all_rays[idx, :, :, :8].reshape(-1, 8)
            sample['ts'] = self.all_rays[idx, :, :, 8].reshape(-1, 1)
            sample['rgbs'] = self.all_rgbs[idx].reshape(-1, 3)
            sample['labels'] = self.all_labels[idx].reshape(-1, 1)
            sample['ray_mask'] = self.all_valid_ray_masks[idx].reshape(-1, 1)
            sample['img_wh'] = torch.LongTensor(self.img_hw)
        else:
            raise NotImplementedError(f"Split {self.split} of dataset is not implemented")

        return sample