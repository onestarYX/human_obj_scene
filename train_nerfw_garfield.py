import os
from opt import get_opts
import torch
from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime
import cv2
import numpy as np
from typing import Optional
# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TestTubeLogger
from einops import repeat
import json

from datasets.sitcom_garfield import SitcomSAMDataset
from datasets.kitti360 import Kitti360Dataset
from datasets.ray_utils import get_ray_directions, get_rays, get_rays_batch_version

# models
from models.nerf import (
    PosEmbedding,
    NeRFWG,
    GarfieldPredictor
)
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
from eval_nerfw_garfield import render_to_path
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
            's_l': hparams.w_transient_reg
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

            self.garfield_predictor = GarfieldPredictor(D=2, W=hparams.dim_hidden_layers)
            self.models['garfield_predictor'] = self.garfield_predictor
            self.models['fine'] = self.nerf_fine
        self.models_to_train += [self.models]
        self.models_mm_to_train = []
        self.eval_only = eval_only
        self.init_datasets()

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
        load_ckpt(self.learn_f, ckpt_path, model_name="learn_f")
        load_ckpt(self.learn_p, ckpt_path, model_name="learn_p")

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def forward(self, rays, ts, do_grouping, test_time):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = \
                render_rays(models=self.models,
                            embeddings=self.embeddings,
                            rays=rays[i:i + self.hparams.chunk],
                            ts=ts[i:i + self.hparams.chunk],
                            do_grouping=do_grouping,
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
        return results

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
        self.optimizer.add_param_group({"params": get_parameters(self.models_mm_to_train), "lr": self.hparams.pose_lr})
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

    def preprocess_batch(self, batch):
        for k in batch.keys():
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].squeeze(0)

    def training_step(self, batch, batch_nb):
        self.preprocess_batch(batch)
        rays = batch['rays']
        ray_mask = batch['ray_mask']
        rgbs = batch['rgbs']
        ts = batch['ts']

        # if self.current_epoch == self.hparams.epochs_begin_grouping:
        #     if

        if self.current_epoch >= self.hparams.epochs_begin_grouping:
            results = self.forward(rays, ts, do_grouping=True, test_time=False)
        else:
            results = self.forward(rays, ts, do_grouping=False, test_time=False)
        loss_d = self.loss(results, rgbs, ray_mask)
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

        if self.current_epoch >= self.hparams.epochs_begin_grouping:
            do_grouping = True
        else:
            do_grouping = False
        results = self.forward(rays, ts, do_grouping=do_grouping, test_time=True)
        loss_d = self.loss(results, rgbs, ray_mask)
        loss = sum(l for l in loss_d.values())

        # Render metrics
        dict_to_log = {'val/loss': loss}
        self.log('val/loss', loss, prog_bar=True)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
        dict_to_log['val/psnr'] = psnr_
        self.log('val/psnr', psnr_, prog_bar=True)
        wandb.log(dict_to_log)

        # Render sample images from training set
        sample_img_idx = self.test_dataset.img_paths[batch_nb].stem
        render_img_name = f"s={self.global_step:06d}_i={batch_nb:03d}_{sample_img_idx}"
        print(f"Rendering sample image {render_img_name} from the training set...")
        render_dir = os.path.join(self.hparams.render_dir, 'train')
        os.makedirs(render_dir, exist_ok=True)
        render_path = os.path.join(render_dir, f"{render_img_name}.png")
        np.random.seed(19)
        label_colors = np.random.rand(self.hparams.num_classes, 3)
        _, res_img = render_to_path(path=render_path, dataset=self.test_dataset,
                                    idx=batch_nb, models=self.models, embeddings=self.embeddings,
                                    config=self.hparams, label_colors=label_colors, do_grouping=do_grouping)
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
                                    config=self.hparams, label_colors=label_colors, do_grouping=do_grouping)
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

    logger = TestTubeLogger(save_dir=save_dir,
                            name=exp_name,
                            debug=False,
                            create_git_tag=False,
                            log_graph=False)
    version = logger.experiment.version
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
                      resume_from_checkpoint=hparams.ckpt_path,
                      logger=logger,
                      weights_summary=None,
                      progress_bar_refresh_rate=hparams.bar_update_freq,
                      gpus=hparams.num_gpus,
                      accelerator='ddp' if hparams.num_gpus > 1 else None,
                      num_sanity_val_steps=1,
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
