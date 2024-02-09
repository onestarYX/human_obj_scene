import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image, ImageDraw
from torchvision import transforms as T
from pathlib import Path
from tqdm import tqdm

from .ray_utils import *
from .kitti_labels import labels as labels_dict
import random

class Kitti360Dataset(Dataset):
    def __init__(self, root_dir, split='train', img_downscale=1, near=0.5, far=12, scene_bound=1):
        self.root_dir = Path(root_dir)
        frame_start = int(self.root_dir.stem.split('_')[-2])
        frame_end = int(self.root_dir.stem.split('_')[-1])
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.split = split

        self.img_downscale = img_downscale
        assert 1408 % img_downscale == 0 and 376 % img_downscale == 0, "Invalid img_downscale!"
        self.img_wh = (1408 // img_downscale, 376 // img_downscale)
        self.near = near
        self.far = far
        self.scene_bound = scene_bound
        self.define_transform()
        self.read_meta()
        self.white_back = True

    def define_transform(self):
        self.transform = T.ToTensor()

    def read_cam_intrinsics(self):
        with open(self.cam_meta_file) as f:
            lines = f.read().splitlines()
        K = None
        for line in lines:
            if line.startswith('P_rect_00'):
                line = line.split(' ')
                K = [float(x) for x in line[1:]]
                K = np.reshape(K, [3, 4])
                K = K[:, :3]
                break
        return K

    def read_cam_poses(self):
        with open(self.cam_pose_file) as f:
            lines = f.read().splitlines()
        cam_poses = {}
        print("Reading cam poses")
        for line in tqdm(lines):
            line = line.split(' ')
            frame_idx = int(line[0])
            pose = [float(x) for x in line[1:-1]]
            pose = np.reshape(pose, [4, 4])
            cam_poses[frame_idx] = pose
        return cam_poses

    def normalize_cam_poses(self):
        x_min = 1e10
        y_min = 1e10
        z_min = 1e10
        x_max = -1e10
        y_max = -1e10
        z_max = -1e10
        for i in range(self.frame_start, self.frame_end + 1, 1):
            pose = self.cam_poses[i]
            x_min = min(pose[0, 3], x_min)
            x_max = max(pose[0, 3], x_max)
            y_min = min(pose[1, 3], y_min)
            y_max = max(pose[1, 3], y_max)
            z_min = min(pose[2, 3], z_min)
            z_max = max(pose[2, 3], z_max)

        center = np.array([(x_min + x_max) / 2,
                           (y_min + y_max) / 2,
                           (z_min + z_max) / 2])
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        xyz_range = np.array([x_range, y_range, z_range])
        self.xyz_range = xyz_range
        print(f"Cam range: {xyz_range}")
        for i in range(self.frame_start, self.frame_end + 1):
            self.cam_poses[i][:3, 3] -= center

    def rescale_scene(self):
        max_range = np.max(self.xyz_range)
        scale_factor = max_range / self.scene_bound
        for i in range(self.frame_start, self.frame_end):
            self.cam_poses[i][:3, 3] /= scale_factor
        self.xyz_range /= scale_factor
        print(f"After rescaling, cam range: {self.xyz_range}")

    def remap_label(self, label_map):
        old_id_max = label_map.max()
        for i in range(-1, old_id_max+1):
            temp_mask = label_map == i
            new_id = self.labels_remapping[i]
            label_map[temp_mask] = new_id
        return label_map

    def read_meta(self):
        w, h = self.img_wh
        self.img_dir = self.root_dir / 'rgb'
        self.img_paths = []
        for file in self.img_dir.iterdir():
            self.img_paths.append(file)
        random.seed(19)
        random.shuffle(self.img_paths)
        num_train_imgs = int(len(self.img_paths) * 0.8)
        if self.split == 'train':
            self.img_paths = self.img_paths[:num_train_imgs]
        elif self.split == 'test_train':
            self.img_paths = self.img_paths[:num_train_imgs]

        elif self.split == 'val':
            self.img_paths = self.img_paths[num_train_imgs:]
        else:
            raise NotImplementedError
        # self.img_paths.sort(key=lambda x: x.name)

        self.labels_dir = self.root_dir / 'semantic'
        self.labels_remapping = {}
        for entry in labels_dict:
            new_label_id = entry.trainId
            if new_label_id == 255 or new_label_id == -1:
                new_label_id = 19
            self.labels_remapping[entry.id] = new_label_id

        self.cam_pose_file = self.root_dir / 'cam0_to_world.txt'
        self.cam_meta_file = self.root_dir / 'perspective.txt'
        self.K = self.read_cam_intrinsics()
        self.K[0, 0] /= self.img_downscale
        self.K[1, 1] /= self.img_downscale
        self.K[0, 2] /= self.img_downscale
        self.K[1, 2] /= self.img_downscale
        self.cam_poses = self.read_cam_poses()
        self.normalize_cam_poses()
        self.rescale_scene()

        # bounds, common for all scenes
        self.bounds = np.array([self.near, self.far])
        
        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(h, w, self.K) # (h, w, 3)
            
        if self.split == 'train': # create buffer of all rays and rgb data
            self.all_rays = []
            self.all_rgbs = []
            self.all_masks = []
            self.all_labels = []

            for t, img_path in enumerate(self.img_paths):
                frame_idx = int(img_path.stem)
                c2w = torch.FloatTensor(self.cam_poses[frame_idx][:3])

                img = Image.open(img_path)
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img)
                img = img.view(3, -1).permute(1, 0)
                self.all_rgbs += [img]
                ray_mask = torch.ones(img.shape[0], dtype=torch.int)
                self.all_masks.append(ray_mask)

                rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)
                rays_t = t * torch.ones(len(rays_o), 1)

                self.all_rays += [torch.cat([rays_o, rays_d,
                                             self.near*torch.ones_like(rays_o[:, :1]),
                                             self.far*torch.ones_like(rays_o[:, :1]),
                                             rays_t],
                                             1)] # (h*w, 9)

                label_path = self.labels_dir / img_path.name
                label = Image.open(label_path)
                label = label.resize(self.img_wh, Image.LANCZOS)
                label = torch.tensor(np.array(label), dtype=torch.long).view(-1)
                label = self.remap_label(label)
                self.all_labels.append(label)


            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_masks = torch.cat(self.all_masks, 0)
            self.all_labels = torch.cat(self.all_labels, 0)
            # Dummy obj_mask
            self.all_obj_masks = torch.ones_like(self.all_masks, dtype=torch.bool)

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'test_train':
            return len(self.img_paths)
        if self.split == 'val':
            return len(self.img_paths)

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            sample = {'rays': self.all_rays[idx, :8],
                      'ts': self.all_rays[idx, 8].long(),
                      'rgbs': self.all_rgbs[idx],
                      'labels': self.all_labels[idx],
                      'ray_mask': self.all_masks[idx],
                      'obj_mask': self.all_obj_masks[idx]}

        else: # create data for each image separately
            img_path = self.img_paths[idx]
            frame_idx = int(img_path.stem)
            c2w = torch.FloatTensor(self.cam_poses[frame_idx][:3])
            t = 0 # transient embedding index, 0 for val and test (no perturbation)

            img = Image.open(img_path)
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)
            img = img.view(3, -1).permute(1, 0)
            ray_mask = torch.ones(img.shape[0], dtype=torch.int)
            obj_mask = torch.ones(img.shape[0], dtype=torch.bool)

            rays_o, rays_d = get_rays(self.directions, c2w)
            rays_t = t * torch.ones(len(rays_o), 1)

            rays = torch.cat([rays_o, rays_d, 
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1]),
                              rays_t],
                              1) # (H*W, 9)

            label_path = self.labels_dir / img_path.name
            label = Image.open(label_path)
            label = label.resize(self.img_wh, Image.LANCZOS)
            label = torch.tensor(np.array(label), dtype=torch.long).view(-1)
            label = self.remap_label(label)

            sample = {'rays': rays,
                      'ts': t * torch.ones(len(rays), dtype=torch.long),
                      'rgbs': img,
                      'c2w': c2w,
                      'labels': label,
                      'ray_mask': ray_mask,
                      'obj_mask': obj_mask
                      }

        return sample