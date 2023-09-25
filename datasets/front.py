import torch
from torch.utils.data import Dataset
import math
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
from pathlib import Path
import argparse
import random
import shutil
import json
from datasets.ray_utils import *


class ThreeDFrontDataset(Dataset):
    def __init__(self, root_dir, split='train', img_downscale=1, test_imgs=[], train_ratio=0.8):
        self.root_dir = Path(root_dir)
        self.split = split
        self.train_ratio = train_ratio
        self.img_wh = (480, 480)
        self.img_downscale = img_downscale
        self.define_transforms()
        self.test_imgs = test_imgs
        self.white_back = True
        self.read_meta()

    def read_meta(self):
        self.img_dir = self.root_dir / 'images'
        self.cam_dir = self.root_dir / 'cameras'
        self.mask_dir = self.root_dir / 'masks'
        self.img_list = []
        for img_path in self.img_dir.iterdir():
            self.img_list.append(img_path)
        # self.img_list = sorted(self.img_list, key=lambda p: p.stem)
        num_train_imgs = int(self.train_ratio * len(self.img_list))
        if self.split == 'train' or self.split == 'test_train':
            self.img_list = self.img_list[:num_train_imgs]
        else:
            self.img_list = self.img_list[num_train_imgs:]

        self.c2w_list = []
        self.K_list = []
        self.cam_path_sorted = []
        for cam_path in self.cam_dir.iterdir():
            self.cam_path_sorted.append(cam_path)
        self.cam_path_sorted = sorted(self.cam_path_sorted, key=lambda p: p.stem)
        for cam_path in self.cam_path_sorted:
            cam_dict = np.load(cam_path)
            R = cam_dict['R_cam_to_world']
            t = cam_dict['translation']
            c2w = np.concatenate([R, t[..., np.newaxis]], axis=1)
            self.c2w_list.append(c2w)
            self.K_list.append(cam_dict['K'])
        self.c2w_list = np.stack(self.c2w_list, axis=0)
        self.K = self.K_list[0]

        w, h = self.img_wh
        # bounds, common for all scenes
        self.near = 2.0
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])
        
        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, self.K) # (h, w, 3)

        if self.split == 'train': # create buffer of all rays and rgb data
            self.all_rays = []
            self.all_rgbs = []
            self.all_masks = []
            self.all_labels = []
            for img_path in self.img_list:
                t = int(img_path.stem)
                pose = self.c2w_list[t]
                c2w = torch.FloatTensor(pose)

                img = Image.open(img_path)
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (3, h, w)
                img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB
                self.all_rgbs += [img]

                # masks
                mask_path = self.mask_dir / img_path.name
                ray_mask = Image.open(mask_path)
                ray_mask = ray_mask.resize(self.img_wh, Image.LANCZOS)
                ray_mask = self.transform(ray_mask)
                ray_mask = ray_mask.view(3, -1).permute(1, 0)[:, 0]
                self.all_masks.append(ray_mask)
                
                rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)
                rays_t = t * torch.ones(len(rays_o), 1)

                self.all_rays += [torch.cat([rays_o, rays_d,
                                             self.near*torch.ones_like(rays_o[:, :1]),
                                             self.far*torch.ones_like(rays_o[:, :1]),
                                             rays_t],
                                             1)] # (h*w, 8)

            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_masks = torch.cat(self.all_masks, 0).to(torch.int)
            # self.all_labels = torch.cat(self.all_labels, 0)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'val':
            return len(self.img_list)
        if self.split == 'test_train':
            return len(self.img_list)

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            sample = {'rays': self.all_rays[idx, :8],
                      'ts': self.all_rays[idx, 8].long(),
                      'rgbs': self.all_rgbs[idx],
                      'labels': 0,
                      'ray_mask': self.all_masks[idx]}

        else: # create data for each image separately
            img_path = self.img_list[idx]
            img_idx = int(img_path.stem)
            pose = self.c2w_list[img_idx]
            c2w = torch.FloatTensor(pose)

            img = Image.open(img_path)
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (3, h, w)
            img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB
            rays_o, rays_d = get_rays(self.directions, c2w)
            rays = torch.cat([rays_o, rays_d, 
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1])],
                              1) # (H*W, 8)

            mask_path = self.mask_dir / img_path.name
            ray_mask = Image.open(mask_path)
            ray_mask = ray_mask.resize(self.img_wh, Image.LANCZOS)
            ray_mask = self.transform(ray_mask)
            ray_mask = ray_mask.view(3, -1).permute(1, 0)[:, 0]

            t = 0
            sample = {'rays': rays,
                      'ts': t * torch.ones(len(rays), dtype=torch.long),
                      'rgbs': img,
                      'c2w': c2w,
                      'labels': 0,
                      'ray_mask': ray_mask
                      }

        return sample
