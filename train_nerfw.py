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

from datasets.sitcom3D import Sitcom3DDataset
from datasets.kitti360 import Kitti360Dataset
from datasets.ray_utils import get_ray_directions, get_rays, get_rays_batch_version

# models
from models.nerf import (
    PosEmbedding,
    NeRF
)
from models.rendering import render_rays
from models.pose import LearnFocal, LearnPose

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
from eval_nerfw import render_to_path
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
            'cce_coarse': hparams.w_label_cce,
            'cce_fine': hparams.w_label_cce
        }
        self.loss = loss_dict['nerfw'](coef=loss_weights, predict_label=hparams.predict_label)

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

        self.nerf_coarse = NeRF('coarse',
                                D=hparams.num_hidden_layers,
                                W=hparams.dim_hidden_layers,
                                skips=hparams.skip_layers,
                                in_channels_xyz=6 * hparams.N_emb_xyz + 3,
                                in_channels_dir=6 * hparams.N_emb_dir + 3,
                                encode_appearance=False,
                                predict_label=self.hparams.predict_label,
                                num_classes=self.hparams.num_classes,
                                use_view_dirs=hparams.use_view_dirs)
        self.models = {'coarse': self.nerf_coarse}
        if hparams.N_importance > 0:
            if hparams.fine_coarse_same:
                self.nerf_fine = self.nerf_coarse
            else:
                self.nerf_fine = NeRF('fine',
                                      D=hparams.num_hidden_layers,
                                      W=hparams.dim_hidden_layers,
                                      skips=hparams.skip_layers,
                                      in_channels_xyz=6 * hparams.N_emb_xyz + 3,
                                      in_channels_dir=6 * hparams.N_emb_dir + 3,
                                      encode_appearance=hparams.encode_a,
                                      in_channels_a=hparams.N_a,
                                      encode_transient=hparams.encode_t,
                                      in_channels_t=hparams.N_tau,
                                      predict_label=self.hparams.predict_label,
                                      num_classes=self.hparams.num_classes,
                                      beta_min=hparams.beta_min,
                                      use_view_dirs=hparams.use_view_dirs)
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

    @torch.no_grad()
    def forward_rays(self,
                     rays_o,
                     rays_d,
                     a=None,
                     t=None,
                     id_=None,
                     N_samples=128,
                     N_importance=128,
                     near_min=0.1):
        """
        rays_o: (H, W, 3)
        rays_d: (H, W, 3)
        a: ()
        t: ()

        near_min is helpful to avoid any artifacts.
        however, probably don't need this for test time if during training near_min
        other than 0 is used
        """

        if isinstance(a, type(None)):
            if id_ is None:
                id_ = self.train_dataset.img_ids[0]
                a = torch.zeros_like(self.embedding_a(torch.tensor(id_).to(rays_o.device)))
            else:
                a = self.embedding_a(torch.tensor(id_).to(rays_o.device))
        if isinstance(t, type(None)):
            if id_ is None:
                id_ = self.train_dataset.img_ids[0]
                t = torch.zeros_like(self.embedding_t(torch.tensor(id_).to(rays_o.device)))
            else:
                t = self.embedding_t(torch.tensor(id_).to(rays_o.device))

        H, W, _ = rays_o.shape
        assert rays_o.shape == rays_d.shape
        rays_o_flat = rays_o.view(-1, 3)
        rays_d_flat = rays_d.view(-1, 3)
        nears, fars, ray_mask = self.train_dataset.get_nears_fars_from_rays_or_cam(rays_o_flat, rays_d_flat, c2w=None)
        nears[nears < near_min] = near_min

        rays = torch.cat([rays_o_flat,
                          rays_d_flat,
                          nears,
                          fars],
                         1)

        B = rays.shape[0]
        results = defaultdict(list)
        for i in tqdm(range(0, B, self.hparams.chunk)):
            kwargs = {}
            kwargs["a_embedded"] = repeat(a, 'c -> n c', n=len(rays[i:i + self.hparams.chunk]))
            kwargs["t_embedded"] = repeat(t, 'c -> n c', n=len(rays[i:i + self.hparams.chunk]))
            rendered_ray_chunks = \
                render_rays(models=self.models,
                            embeddings=self.embeddings,
                            rays=rays[i:i + self.hparams.chunk],
                            ts=None,  # ts[i:i + self.hparams.chunk],
                            N_samples=N_samples,
                            use_disp=False,
                            perturb=self.hparams.perturb,
                            noise_std=False,
                            N_importance=N_importance,
                            chunk=self.hparams.chunk,  # chunk size is effective in val mode
                            white_back=True,
                            test_time=True,
                            **kwargs)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v.cpu()]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        results['img_wh'] = [W, H]
        return results

    @torch.no_grad()
    def forward_pose_K_a_t(self, c2w, K,
                           a=None,
                           t=None,
                           id_=None,
                           #    N_samples=128,
                           #    N_importance=128,
                           N_samples=128,
                           #    N_importance=64,
                           N_importance=128,
                           near_min=0.1):
        """
        pose: (3, 4)
        K: (3, 3)
        a: ()
        t: ()

        near_min is helpful to avoid any artifacts.
        however, probably don't need this for test time if during training near_min
        other than 0 is used
        """

        if isinstance(a, type(None)):
            if id_ is None:
                id_ = self.train_dataset.img_ids[0]
                a = torch.zeros_like(self.embedding_a(torch.tensor(id_).to(c2w.device)))
            else:
                a = self.embedding_a(torch.tensor(id_).to(c2w.device))
        if isinstance(t, type(None)):
            if id_ is None:
                id_ = self.train_dataset.img_ids[0]
                t = torch.zeros_like(self.embedding_t(torch.tensor(id_).to(c2w.device)))
            else:
                t = self.embedding_t(torch.tensor(id_).to(c2w.device))

        assert c2w.shape == (3, 4)

        H, W = round(K[1, 2].item() * 2.0), round(K[0, 2].item() * 2.0)  # using "round" bc of floating precision

        directions = get_ray_directions(H, W, K)
        rays_o, rays_d = get_rays(directions, c2w)
        nears, fars, ray_mask = self.train_dataset.get_nears_fars_from_rays_or_cam(rays_o, rays_d, c2w=c2w)

        # print((nears < near_min).float().sum())
        nears[nears < near_min] = near_min

        # nears += 0.05 # TODO(ethan): decide to keep this or not
        rays = torch.cat([rays_o, rays_d,
                          nears,
                          fars],
                         1)

        B = rays.shape[0]
        results = defaultdict(list)
        for i in tqdm(range(0, B, self.hparams.chunk)):
            kwargs = {}
            kwargs["a_embedded"] = repeat(a, 'c -> n c', n=len(rays[i:i + self.hparams.chunk]))
            kwargs["t_embedded"] = repeat(t, 'c -> n c', n=len(rays[i:i + self.hparams.chunk]))
            rendered_ray_chunks = \
                render_rays(models=self.models,
                            embeddings=self.embeddings,
                            rays=rays[i:i + self.hparams.chunk],
                            ts=None,  # ts[i:i + self.hparams.chunk],
                            N_samples=N_samples,
                            use_disp=False,
                            perturb=self.hparams.perturb,
                            noise_std=False,
                            N_importance=N_importance,
                            chunk=self.hparams.chunk,  # chunk size is effective in val mode
                            white_back=True,
                            test_time=True,
                            **kwargs)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v.cpu()]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        results['img_wh'] = [W, H]
        return results

    def forward(self, rays, ts, version=None):
        """Do batched inference on rays using chunk."""
        assert version is not None
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i + self.hparams.chunk],
                            ts[i:i + self.hparams.chunk],
                            self.hparams.predict_label,
                            self.hparams.num_classes,
                            self.hparams.N_samples,
                            self.hparams.use_disp,
                            self.hparams.perturb if version == "train" else 0,
                            self.hparams.noise_std,
                            self.hparams.N_importance,
                            self.hparams.chunk,  # chunk size is effective in val mode
                            self.train_dataset.white_back,
                            validation_version=version == "val")

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def init_datasets(self):
        if self.hparams.dataset_name == 'sitcom3D':
            dataset = Sitcom3DDataset
            kwargs = {'environment_dir': self.hparams.environment_dir,
                      'near_far_version': self.hparams.near_far_version}
            # kwargs['img_downscale'] = self.hparams.img_downscale
            kwargs['val_num'] = self.hparams.num_gpus
            kwargs['use_cache'] = self.hparams.use_cache
            kwargs['num_limit'] = self.hparams.num_limit
            # !!!!! There is no manual near and far restriction to the dataset. We didn't overwrite near here !!!!!
            self.train_dataset = dataset(split='train' if not self.eval_only else 'val',
                                         img_downscale=self.hparams.img_downscale, **kwargs)
            self.val_dataset = dataset(split='val', img_downscale=self.hparams.img_downscale_val, **kwargs)
            self.test_dataset = dataset(split='test_train', img_downscale=self.hparams.img_downscale, **kwargs)
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
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1,  # validate one image (H*W rays) at a time
                          pin_memory=True)

    def is_learning_pose(self):
        return (self.hparams.learn_f or self.hparams.learn_r or self.hparams.learn_t)

    def rays_from_batch(self, batch):
        if self.is_learning_pose():
            raise NotImplementedError
        else:
            rays = batch['rays'].squeeze()
            ray_mask = batch['ray_mask'].squeeze()
        return rays, ray_mask

    def training_step(self, batch, batch_nb):
        rays, ray_mask = self.rays_from_batch(batch)
        rgbs = batch['rgbs']
        if 'ts2' in batch:
            ts = batch['ts2']
        else:
            ts = batch['ts']
        labels = batch['labels'].to(torch.long).squeeze()
        results = self.forward(rays, ts, version="train")
        loss_d = self.loss(results, rgbs, labels, ray_mask)
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
        rays, ray_mask = self.rays_from_batch(batch)
        rgbs = batch['rgbs']
        if 'ts2' in batch:
            ts = batch['ts2']
        else:
            ts = batch['ts']
        labels = batch['labels'].to(torch.long).squeeze()
        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        ts = ts.squeeze()  # (H*W)
        results = self.forward(rays, ts, version="val")
        loss_d = self.loss(results, rgbs, labels, ray_mask)
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
                                    config=self.hparams, label_colors=label_colors)
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
                                    config=self.hparams, label_colors=label_colors)
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
        if hparams.predict_label:
            exp_name += '_label'

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
