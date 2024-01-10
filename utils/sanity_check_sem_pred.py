import os
import torch
from collections import defaultdict
from tqdm import tqdm

from datasets.sitcom3D import Sitcom3DDataset
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str)
    parser.add_argument('--split', type=str, default='test_train')
    parser.add_argument('--index', type=int)
    parser.add_argument('--check_pixel', nargs=2, type=int)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)

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

    sample = dataset[args.index]
    results_dir = exp_dir / 'results/eval'
    results_files = []
    for file in results_dir.iterdir():
        results_files.append(file)

    results_files.sort(key=lambda x:x.name)

    print("here")
