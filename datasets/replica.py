import torch
from torch.utils.data import Dataset
import math
import numpy as np
import os
from PIL import Image, ImageDraw
from torchvision import transforms as T
from pathlib import Path
import argparse
import random
import shutil

from datasets.ray_utils import *


class ReplicaDataset(Dataset):
    def __init__(self, root_dir, split='train', img_downscale=1, test_imgs=[]):
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_wh = (640, 480)
        self.img_downscale = img_downscale
        self.define_transforms()
        self.test_imgs = test_imgs
        self.read_meta()
        self.white_back = True

    def read_meta(self):
        if self.split == 'test_train':
            split_dir = 'train'
        else:
            split_dir = self.split
        self.data_dir = self.root_dir / split_dir
        self.img_dir = self.data_dir / 'rgb'
        self.img_list = []
        for img_path in self.img_dir.iterdir():
            self.img_list.append(img_path)

        self.c2w_list = np.load(self.root_dir / 'cam.npy')

        w, h = self.img_wh
        self.hfov = 90
        self.focal = w / 2.0 / math.tan(math.radians(self.hfov / 2.0))
        self.K = np.eye(3)
        self.K[0, 0] = self.K[1, 1] = self.focal
        self.K[0, 2] = w/2
        self.K[1, 2] = h/2

        # bounds, common for all scenes
        self.near = 2.0
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])
        
        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions_replica(h, w, self.K, convention='opencv') # (h, w, 3)
            
        if self.split == 'train': # create buffer of all rays and rgb data
            self.all_rays = []
            self.all_rgbs = []
            self.all_masks = []
            for img_path in self.img_list:
                t = int(img_path.stem.split('_')[-1])
                pose = self.c2w_list[t, :3, :4]
                c2w = torch.FloatTensor(pose)

                img = Image.open(img_path)
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (3, h, w)
                ray_mask = torch.ones(img.shape[1:], dtype=torch.int).flatten()  # (H*W) valid color area
                self.all_masks.append(ray_mask)
                img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB
                self.all_rgbs += [img]
                
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

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'val':
            return len(self.img_list) # only validate 8 images (to support <=8 gpus)
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
            idx = int(img_path.stem.split('_')[-1])
            c2w = torch.FloatTensor(self.c2w_list)[idx, :3, :4]

            img = Image.open(img_path)
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img) # (3, H, W)
            ray_mask = torch.ones(img.shape[1:], dtype=torch.int).flatten() # (H*W) valid color area
            img = img.view(3, -1).permute(1, 0) # (H*W, 3) RGB
            rays_o, rays_d = get_rays(self.directions, c2w)

            rays = torch.cat([rays_o, rays_d, 
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1])],
                              1) # (H*W, 8)

            t = 0
            sample = {'rays': rays,
                      'ts': t * torch.ones(len(rays), dtype=torch.long),
                      'rgbs': img,
                      'c2w': c2w,
                      'labels': 0,
                      'ray_mask': ray_mask.to(torch.int)
                      }

        return sample


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/replica/office_0/Sequence_1')
    args = parser.parse_args()

    valid_split = 0.15
    test_split = 0.15
    data_dir = Path(args.data_dir)
    rgb_dir = data_dir / 'rgb'
    depth_dir = data_dir / 'depth'
    label_dir = data_dir / 'semantic_class'
    camera_path = data_dir / 'traj_w_c.txt'

    output_dir = data_dir.parent / 'nerflet'
    train_out_dir = output_dir / 'train'
    valid_out_dir = output_dir / 'val'
    test_out_dir = output_dir / 'test'
    camera_out_path = output_dir / 'cam.npy'

    img_list = []
    for img_file in rgb_dir.iterdir():
        img_list.append(img_file)
    random.shuffle(img_list)

    valid_count = int(valid_split * len(img_list))
    test_count = int(test_split * len(img_list))
    valid_imgs = img_list[:valid_count]
    test_imgs = img_list[valid_count:valid_count + test_count]
    train_imgs = img_list[valid_count + test_count:]

    camera_list = []
    with open(camera_path, 'r') as f:
        for line in f:
            cam = line.split(' ')
            cam = [float(x) for x in cam]
            cam = np.array(cam).reshape(4, 4)
            camera_list.append(cam)
    camera_list = np.stack(camera_list, axis=0)
    np.save(camera_out_path, camera_list)

    train_rgb_dir = train_out_dir / 'rgb'; train_rgb_dir.mkdir(parents=True, exist_ok=True)
    train_depth_dir = train_out_dir / 'depth'; train_depth_dir.mkdir(parents=True, exist_ok=True)
    train_label_dir = train_out_dir / 'label'; train_label_dir.mkdir(parents=True, exist_ok=True)
    for img in train_imgs:
        shutil.copyfile(img, train_rgb_dir / img.name)

        idx = int(img.stem.split('_')[-1])
        src_depth_file = depth_dir / f"depth_{idx}.png"
        src_label_file = label_dir / f"semantic_class_{idx}.png"
        shutil.copyfile(src_depth_file, train_depth_dir / src_depth_file.name)
        shutil.copyfile(src_label_file, train_label_dir / src_label_file.name)

    valid_rgb_dir = valid_out_dir / 'rgb'; valid_rgb_dir.mkdir(parents=True, exist_ok=True)
    valid_depth_dir = valid_out_dir / 'depth'; valid_depth_dir.mkdir(parents=True, exist_ok=True)
    valid_label_dir = valid_out_dir / 'label'; valid_label_dir.mkdir(parents=True, exist_ok=True)
    for img in valid_imgs:
        shutil.copyfile(img, valid_rgb_dir / img.name)

        idx = int(img.stem.split('_')[-1])
        src_depth_file = depth_dir / f"depth_{idx}.png"
        src_label_file = label_dir / f"semantic_class_{idx}.png"
        shutil.copyfile(src_depth_file, valid_depth_dir / src_depth_file.name)
        shutil.copyfile(src_label_file, valid_label_dir / src_label_file.name)

    test_rgb_dir = test_out_dir / 'rgb'; test_rgb_dir.mkdir(parents=True, exist_ok=True)
    test_depth_dir = test_out_dir / 'depth'; test_depth_dir.mkdir(parents=True, exist_ok=True)
    test_label_dir = test_out_dir / 'label'; test_label_dir.mkdir(parents=True, exist_ok=True)
    for img in test_imgs:
        shutil.copyfile(img, test_rgb_dir / img.name)

        idx = int(img.stem.split('_')[-1])
        src_depth_file = depth_dir / f"depth_{idx}.png"
        src_label_file = label_dir / f"semantic_class_{idx}.png"
        shutil.copyfile(src_depth_file, test_depth_dir / src_depth_file.name)
        shutil.copyfile(src_label_file, test_label_dir / src_label_file.name)
