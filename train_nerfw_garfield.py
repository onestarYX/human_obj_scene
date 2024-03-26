import os
from opt import get_opts
import torch
from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime
import numpy as np
from typing import Optional
# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import json
from pathlib import Path
import pickle

from transformers import pipeline
from datasets.sitcom_garfield import SitcomSAMDataset
from datasets.kitti360 import Kitti360Dataset
from datasets.ray_utils import get_ray_directions, get_rays, get_rays_batch_version
from PIL import Image

# models
from models.nerf import (
    PosEmbedding,
    NeRFWG
)
from models.garfield import GarfieldPredictor
from models.rendering_garfield import render_rays

# optimizer, scheduler, visualization
from utils import (
    get_parameters,
    get_optimizer,
    get_scheduler,
    get_learning_rate
)

# losses
from losses import loss_dict
from metrics import psnr

from utils import load_ckpt

import wandb
from eval_nerfw_garfield import render_to_path, batched_inference
from train_nerfletw import count_parameters


class NeRFSystem(LightningModule):
    def __init__(self, hparams, eval_only=False):
        super().__init__()
        # self.hparams = hparams
        self.save_hyperparameters(hparams)

        loss_weights = {
            'c_l': hparams.w_color_l,
            'f_l': hparams.w_color_l,
            'b_l': hparams.w_beta_l,
            's_l': hparams.w_transient_reg,
            'garfield_l': hparams.w_garfield
        }
        self.loss = loss_dict['nerfw_garfield'](coef=loss_weights)

        self.models_to_train = []
        self.embedding_xyz = PosEmbedding(hparams.N_emb_xyz - 1, hparams.N_emb_xyz)
        self.embedding_dir = PosEmbedding(hparams.N_emb_dir - 1, hparams.N_emb_dir)
        self.embeddings = {'xyz': self.embedding_xyz,
                           'dir': self.embedding_dir}

        if hparams.encode_a:
            self.embedding_a = torch.nn.Embedding(hparams.N_vocab, hparams.N_a)
            self.embeddings['a'] = self.embedding_a
            self.models_to_train += [self.embedding_a]
        if hparams.encode_t:
            self.embedding_t = torch.nn.Embedding(hparams.N_vocab, hparams.N_tau)
            self.embeddings['t'] = self.embedding_t
            self.models_to_train += [self.embedding_t]

        self.nerf_coarse = NeRFWG('coarse',
                                  D=hparams.num_hidden_layers,
                                  W=hparams.dim_hidden_layers,
                                  skips=hparams.skip_layers,
                                  in_channels_xyz=6 * hparams.N_emb_xyz + 3,
                                  in_channels_dir=6 * hparams.N_emb_dir + 3,
                                  encode_appearance=False,
                                  use_view_dirs=hparams.use_view_dirs)
        self.models = {'coarse': self.nerf_coarse}
        if hparams.N_importance > 0:
            if hparams.fine_coarse_same:
                self.nerf_fine = self.nerf_coarse
            else:
                self.nerf_fine = NeRFWG('fine',
                                        D=hparams.num_hidden_layers,
                                        W=hparams.dim_hidden_layers,
                                        skips=hparams.skip_layers,
                                        in_channels_xyz=6 * hparams.N_emb_xyz + 3,
                                        in_channels_dir=6 * hparams.N_emb_dir + 3,
                                        encode_appearance=hparams.encode_a,
                                        in_channels_a=hparams.N_a,
                                        encode_transient=hparams.encode_t,
                                        in_channels_t=hparams.N_tau,
                                        beta_min=hparams.beta_min,
                                        use_view_dirs=hparams.use_view_dirs)
            self.models['fine'] = self.nerf_fine

        if self.hparams.overwrite_nerfw_ckpt is not None:
            self.load_from_ckpt_path(self.hparams.overwrite_nerfw_ckpt)

        self.garfield_predictor = GarfieldPredictor()
        self.models['garfield_predictor'] = self.garfield_predictor

        self.models_to_train += [self.models]
        self.eval_only = eval_only
        self.init_datasets()
        self.sam_dict = None

        self.near_min = 0.1
        self.appearance_id = None
        self.height = None

        num_params = 0
        for item in self.models_to_train:
            if isinstance(item, list):
                for model in item:
                    num_model_params = count_parameters(model)
                    print(f"# of params in some unnamed model----{num_model_params}")
                    num_params += num_model_params
            elif isinstance(item, dict):
                for model_name, model in item.items():
                    num_model_params = count_parameters(model)
                    print(f"# of params in {model_name}----{num_model_params}")
                    num_params += num_model_params
            else:
                num_model_params = count_parameters(item)
                print(f"# of params in some unnamed model----{num_model_params}")
                num_params += num_model_params
        print(f"Number of parameters in total: {num_params}")

    def load_from_ckpt_path(self, ckpt_path):
        """TODO(ethan): move this elsewhere
        """
        load_ckpt(self.embedding_a, ckpt_path, model_name="embedding_a")
        load_ckpt(self.embedding_t, ckpt_path, model_name="embedding_t")
        load_ckpt(self.nerf_coarse, ckpt_path, model_name="nerf_coarse")
        load_ckpt(self.nerf_fine, ckpt_path, model_name="nerf_fine")

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def init_datasets(self):
        if self.hparams.dataset_name == 'sitcom3D':
            dataset = SitcomSAMDataset
            kwargs = {'environment_dir': self.hparams.environment_dir,
                      'near_far_version': self.hparams.near_far_version,
                      'num_limit': self.hparams.num_limit,
                      'bs': self.hparams.batch_size}
            # Far comes from estiamtion of scene point cloud. And we didn't overwrite near here !!!!!
            self.train_dataset = dataset(split='train', img_downscale=self.hparams.img_downscale, **kwargs)
            self.val_dataset = dataset(split='val', img_downscale=self.hparams.img_downscale_val, **kwargs)
            self.test_dataset = dataset(split='test_train', img_downscale=self.hparams.img_downscale_val, **kwargs)
        elif self.hparams.dataset_name == 'kitti360':
            self.train_dataset = Kitti360Dataset(root_dir=self.hparams.environment_dir, split='train',
                                                 img_downscale=self.hparams.img_downscale,
                                                 near=self.hparams.near, far=self.hparams.far,
                                                 scene_bound=self.hparams.scene_bound)
            self.val_dataset = Kitti360Dataset(root_dir=self.hparams.environment_dir, split='val',
                                               img_downscale=self.hparams.img_downscale_val,
                                               near=self.hparams.near, far=self.hparams.far,
                                               scene_bound=self.hparams.scene_bound)
            self.test_dataset = Kitti360Dataset(root_dir=self.hparams.environment_dir, split='test_train',
                                                img_downscale=self.hparams.img_downscale,
                                                near=self.hparams.near, far=self.hparams.far,
                                                scene_bound=self.hparams.scene_bound)


    def setup(self, stage):
        pass
        # TODO(ethan): handle optimizer parameters here

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models_to_train)
        # self.optimizer.add_param_group({"params": get_parameters(self.models_mm_to_train), "lr": self.hparams.pose_lr})
        scheduler = get_scheduler(self.hparams, self.optimizer)
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=1,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1,  # validate one image (H*W rays) at a time
                          pin_memory=True)

    def is_learning_pose(self):
        return (self.hparams.learn_f or self.hparams.learn_r or self.hparams.learn_t)

    def check_sam_results_existed(self):
        data_dir = Path(self.hparams.environment_dir)
        sam_file = data_dir / 'sam_masks' / 'sam_results.pkl'
        if sam_file.exists():
            return True
        else:
            return False

    def create_pixel_mask_array(self, masks: torch.Tensor):
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
        pixel_mask_array = pixel_mask_array.permute(1, 2, 0).long()

        return pixel_mask_array

    @torch.no_grad()
    def calculate_3d_grouping(self):
        # Get SAM masks
        sam_out_dir = Path(self.hparams.environment_dir) / 'sam_masks'
        sam_out_dir.mkdir(exist_ok=True)
        sam_out_path = 'sam_results.pkl'
        sam_out_path = sam_out_dir / sam_out_path
        self.sam_dict = {}
        print("Generating SAM masks......")
        sam_model = pipeline("mask-generation", model="facebook/sam-vit-huge", device=torch.device('cuda'))

        all_sam_data = {}
        all_pixel_level_keys = []
        all_scales = []
        all_mask_cdf = []
        for i in tqdm(range(len(self.test_dataset))):
            sample = self.test_dataset[i]
            img_id = sample['id'].item()

            img_w, img_h = sample['img_wh']
            img = sample['rgbs'].view(img_h, img_w, -1)
            img = (img * 255).to(torch.uint8).cpu().numpy()
            img = Image.fromarray(img)
            sam_masks_raw = sam_model(img, points_per_side=32, pred_iou_thresh=0.90, stability_score_thresh=0.90)
            sam_masks_raw = sam_masks_raw['masks']
            sam_masks_raw = sorted(sam_masks_raw, key=lambda x: x.sum())
            sam_masks_raw = torch.tensor(np.stack(sam_masks_raw, axis=0)).cuda()    # (N_masks, h, w)


            rays = sample['rays'].cuda()
            ts = sample['ts'].cuda().squeeze()
            results = batched_inference(models=self.models, embeddings=self.embeddings,
                                        rays=rays, ts=ts,
                                        do_grouping=False, N_samples=self.hparams.N_samples,
                                        N_importance=self.hparams.N_importance,
                                        use_disp=self.hparams.use_disp, chunk=self.hparams.chunk,
                                        white_back=self.test_dataset.white_back)

            depth = results['depth_fine_static'].unsqueeze(-1).cuda()
            rays_o = rays[:, 0:3]
            rays_d = rays[:, 3:6]
            rays_d = torch.nn.functional.normalize(rays_d, dim=-1)
            depth_points = rays_o + rays_d * depth

            valid_ray_mask = sample['ray_mask'].squeeze().cuda()
            sam_masks = []
            scales = []
            for i in range(len(sam_masks_raw)):
                cur_mask = sam_masks_raw[i].flatten()
                cur_mask = cur_mask & valid_ray_mask
                if cur_mask.sum() <= 50:
                    continue
                cur_points = depth_points[cur_mask]
                extent = (cur_points.std(dim=0) * 2).norm()
                sam_masks.append(cur_mask)
                scales.append(extent)

            sam_masks = torch.stack(sam_masks, dim=0).view(-1, img_h, img_w)
            pixel_level_keys = self.create_pixel_mask_array(sam_masks)
            scales = torch.tensor(scales).unsqueeze(-1)

            # Calculate group sampling CDF, to bias sampling towards smaller groups
            # Be careful to not include -1s in the CDF (padding, or unlabeled pixels)
            # Inversely proportional to log of mask size.
            mask_inds, counts = torch.unique(pixel_level_keys, return_counts=True)
            counts[0] = 0  # don't include -1
            mask_sorted = torch.argsort(counts)
            mask_inds, counts = mask_inds[mask_sorted], counts[mask_sorted]
            probs = counts / counts.sum()  # [-1, 0, ...]
            mask_probs = torch.gather(probs, 0, pixel_level_keys.reshape(-1) + 1).view(
                pixel_level_keys.shape
            )
            mask_log_probs = torch.log(mask_probs)
            never_masked = mask_log_probs.isinf()
            mask_log_probs[never_masked] = 0.0
            mask_log_probs = mask_log_probs / (
                    mask_log_probs.sum(dim=-1, keepdim=True) + 1e-6
            )
            mask_cdf = torch.cumsum(mask_log_probs, dim=-1)
            mask_cdf[never_masked] = 1.0

            all_pixel_level_keys.append(pixel_level_keys.cpu())
            all_scales.append(scales.cpu())
            all_mask_cdf.append(mask_cdf.cpu())

        all_sam_data = {
            'pixel_level_keys': all_pixel_level_keys,
            'scales': all_scales,
            'group_cdf': all_mask_cdf
        }

        with open(sam_out_path, 'wb') as f:
            pickle.dump(all_sam_data, f)

        self.sam_dict = all_sam_data
        del sam_model
        torch.cuda.empty_cache()

    @torch.no_grad()
    def sample_masks_and_scales(self, indices):
        indices = indices.detach().cpu()    # (B, 3)
        npximg = self.hparams.num_rays_per_img
        img_ind = indices[:, 0]
        x_ind = indices[:, 1]
        y_ind = indices[:, 2]

        mask_id = torch.zeros((indices.shape[0],))
        scale = torch.zeros((indices.shape[0],))

        random_vec_sampling = (torch.rand((1,)) * torch.ones((npximg,))).view(-1, 1)
        random_vec_densify = (torch.rand((1,)) * torch.ones((npximg,))).view(-1, 1)

        pixel_level_keys = self.sam_dict['pixel_level_keys']    # (N_imgs, h, w, max_masks)
        scales = self.sam_dict['scales']   # (N_imgs, num_masks, 1)
        group_cdf = self.sam_dict['group_cdf']  # (N_imgs, h, w, max_masks)
        for i in range(0, indices.shape[0], npximg):
            img_idx = img_ind[i]

            # Use `random_vec` to choose a group for each pixel.
            per_pixel_index = pixel_level_keys[img_idx][
                x_ind[i: i + npximg], y_ind[i: i + npximg]
            ]
            random_index = torch.sum(random_vec_sampling.view(-1, 1)
                                     > group_cdf[img_idx][x_ind[i: i + npximg], y_ind[i: i + npximg]],
                                     dim=-1)

            # `per_pixel_index` encodes the list of groups that each pixel belongs to.
            # If there's only one group, then `per_pixel_index` is a 1D tensor
            # -- this will mess up the future `gather` operations.
            if per_pixel_index.shape[-1] == 1:
                per_pixel_mask = per_pixel_index.squeeze()
            else:
                per_pixel_mask = torch.gather(
                    per_pixel_index, 1, random_index.unsqueeze(-1)
                ).squeeze()
                per_pixel_mask_ = torch.gather(
                    per_pixel_index,
                    1,
                    torch.max(random_index.unsqueeze(-1) - 1, torch.Tensor([0]).int()),
                ).squeeze()

            mask_id[i: i + npximg] = per_pixel_mask

            # interval scale supervision
            curr_scale = scales[img_idx][per_pixel_mask]
            curr_scale[random_index == 0] = (
                    scales[img_idx][per_pixel_mask][random_index == 0]
                    * random_vec_densify[random_index == 0]
            )
            for j in range(1, group_cdf[img_idx].shape[-1]):
                if (random_index == j).sum() == 0:
                    continue
                curr_scale[random_index == j] = (
                        scales[img_idx][per_pixel_mask_][random_index == j]
                        + (
                                scales[img_idx][per_pixel_mask][random_index == j]
                                - scales[img_idx][per_pixel_mask_][random_index == j]
                        )
                        * random_vec_densify[random_index == j]
                )
            scale[i: i + npximg] = curr_scale.squeeze().to(self.device)

        return mask_id, scale


    def forward(self, rays, ts, pixel_indices, do_grouping, test_time):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            inputs = {'rays': rays[i:i + self.hparams.chunk],
                      'ts': ts[i:i + self.hparams.chunk]}
            rendered_ray_chunks = \
                render_rays(models=self.models,
                            embeddings=self.embeddings,
                            inputs=inputs,
                            N_samples=self.hparams.N_samples,
                            use_disp=self.hparams.use_disp,
                            perturb=self.hparams.perturb if not test_time else 0,
                            noise_std=self.hparams.noise_std,
                            N_importance=self.hparams.N_importance,
                            chunk=self.hparams.chunk,  # chunk size is effective in val mode
                            white_back=self.train_dataset.white_back,
                            test_time=test_time)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)

        if do_grouping:
            assert pixel_indices is not None
            sam_masks, scales = self.sample_masks_and_scales(indices=pixel_indices)
            sam_masks = sam_masks.to(rays.device)
            scales = scales.to(rays.device)
            results['sam_masks'] = sam_masks
            results['scales'] = scales
            results['num_rays_per_img'] = self.hparams.num_rays_per_img

        return results


    def preprocess_batch(self, batch):
        for k in batch.keys():
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].squeeze(0)

    def training_step(self, batch, batch_nb):
        self.preprocess_batch(batch)
        rays = batch['rays']
        rgbs = batch['rgbs']
        ts = batch['ts']

        if self.current_epoch == self.hparams.epochs_begin_grouping:
            if self.sam_dict is None:
                if self.check_sam_results_existed():
                    sam_file_path = Path(self.hparams.environment_dir) / 'sam_masks' / 'sam_results.pkl'
                    with open(sam_file_path, 'rb') as f:
                        sam_dict = pickle.load(f)
                        self.sam_dict = sam_dict
                else:
                    self.calculate_3d_grouping()

            if self.garfield_predictor.quantile_transformer is None:
                scales_stats = self.sam_dict['scales']
                scales_stats = torch.cat(scales_stats, dim=0)   # (all_num_masks, 1)
                self.garfield_predictor.get_quantile_func(scales_stats)

        do_grouping = False
        if self.current_epoch >= self.hparams.epochs_begin_grouping:
            do_grouping = True
        results = self.forward(rays, ts, do_grouping=do_grouping, pixel_indices=batch['pixel_indices'], test_time=False)
        loss_d = self.loss(results, batch, self.models, do_grouping=do_grouping)
        loss = sum(l for l in loss_d.values())

        with torch.no_grad():
            typ = 'fine' if 'rgb_fine' in results else 'coarse'
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)

        self.log('lr', get_learning_rate(self.optimizer))
        self.log('train/loss', loss)
        for k, v in loss_d.items():
            self.log(f'train/{k}', v, prog_bar=True)
        self.log('train/psnr', psnr_, prog_bar=True)

        # wandb log
        dict_to_log = {}
        dict_to_log['lr'] = get_learning_rate(self.optimizer)
        dict_to_log['epoch'] = self.current_epoch
        dict_to_log['train/loss'] = loss
        for k, v in loss_d.items():
            dict_to_log[f"train/{k}"] = v
        dict_to_log['train/psnr'] = psnr_
        wandb.log(dict_to_log)

        return loss

    def validation_step(self, batch, batch_nb):
        self.preprocess_batch(batch)
        rays = batch['rays']
        ray_mask = batch['ray_mask']
        rgbs = batch['rgbs']
        ts = batch['ts'].squeeze()

        results = self.forward(rays, ts, do_grouping=False, pixel_indices=None, test_time=True)
        loss_d = self.loss(results, batch, self.models, do_grouping=False)
        loss = sum(l for l in loss_d.values())

        # Render metrics
        dict_to_log = {'val/loss': loss}
        self.log('val/loss', loss, prog_bar=True)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
        dict_to_log['val/psnr'] = psnr_
        self.log('val/psnr', psnr_, prog_bar=True)
        wandb.log(dict_to_log)

        do_grouping = False
        if self.current_epoch >= self.hparams.epochs_begin_grouping:
            do_grouping = True
        # Render sample images from training set
        sample_img_idx = self.test_dataset.img_paths[batch_nb].stem
        render_img_name = f"s={self.global_step:06d}_i={batch_nb:03d}_{sample_img_idx}"
        print(f"Rendering sample image {render_img_name} from the training set...")
        render_dir = os.path.join(self.hparams.render_dir, 'train')
        os.makedirs(render_dir, exist_ok=True)
        render_path = os.path.join(render_dir, f"{render_img_name}.png")
        np.random.seed(19)
        _, res_img = render_to_path(path=render_path, dataset=self.test_dataset,
                                    idx=batch_nb, models=self.models, embeddings=self.embeddings,
                                    config=self.hparams, do_grouping=do_grouping)
        wd_img = wandb.Image(res_img, caption=f"{render_img_name}")
        wandb.log({f"train_rendering/Renderings_id={batch_nb}": wd_img})

        # Render sample images from validation set
        sample_img_idx = self.val_dataset.img_paths[batch_nb].stem
        render_img_name = f"s={self.global_step:06d}_i={batch_nb:03d}_{sample_img_idx}"
        print(f"Rendering sample image {render_img_name} from the validation set...")
        render_dir = os.path.join(self.hparams.render_dir, 'val')
        os.makedirs(render_dir, exist_ok=True)
        render_path = os.path.join(render_dir, f"{render_img_name}.png")
        _, res_img = render_to_path(path=render_path, dataset=self.val_dataset,
                                    idx=batch_nb, models=self.models, embeddings=self.embeddings,
                                    config=self.hparams, do_grouping=do_grouping)
        wd_img = wandb.Image(res_img, caption=f"{render_img_name}")
        wandb.log({f"val_rendering/Renderings_id={batch_nb}": wd_img})

        return dict_to_log


def main(hparams):
    save_dir = os.path.join(hparams.environment_dir, "runs")

    if hparams.resume_name:
        exp_name = hparams.resume_name
        group_name = exp_name.split('_')[:-1]
        group_name = '_'.join(group_name) + '_cont'
    else:
        exp_name = hparams.exp_name

        group_name = exp_name
        time_string = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
        exp_name += '_' + time_string

    # following pytorch lightning convention here

    logger = TensorBoardLogger(save_dir=save_dir,
                                name=exp_name)
    version = 0
    if not isinstance(version, int):
        version = 0

    if hparams.resume_name:
        assert hparams.ckpt_path is not None

    # following pytorch lightning convention here
    dir_path = os.path.join(save_dir, exp_name, f"version_{version}")
    os.makedirs(dir_path, exist_ok=True)
    config = vars(hparams)
    config_save_path = os.path.join(dir_path, 'config.json')
    json_obj = json.dumps(config, indent=2)
    with open(config_save_path, 'w') as f:
        f.write(json_obj)

    run = wandb.init(
        # Set the project where this run will be logged
        project="my_nerfw",
        name=exp_name,
        # Track hyperparameters and run metadata
        config=config,
        group=group_name
    )

    hparams.render_dir = f"{dir_path}/render_logs"
    os.makedirs(hparams.render_dir, exist_ok=True)
    system = NeRFSystem(hparams)

    checkpoint_filepath = os.path.join(f'{dir_path}/ckpts')
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_filepath,
                                          monitor='val/psnr',
                                          mode='max',
                                          save_top_k=5,
                                          save_last=True)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      callbacks=[checkpoint_callback],
                      logger=logger,
                      log_every_n_steps=hparams.bar_update_freq,
                      num_nodes=hparams.num_gpus,
                      accelerator='cuda',
                      num_sanity_val_steps=0,
                      val_check_interval=hparams.val_freq if hparams.val_freq < 1 else int(hparams.val_freq),
                      limit_val_batches=5,
                      benchmark=True,
                      accumulate_grad_batches=hparams.accumulate_grad_batches
                      #   profiler="simple" if hparams.num_gpus == 1 else None
                      )

    trainer.fit(system)


if __name__ == '__main__':
    hparams = get_opts()
    main(hparams)
