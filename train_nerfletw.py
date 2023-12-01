import os
from opt import get_opts
import torch
from collections import defaultdict
from torch.utils.data import DataLoader
import datetime
# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TestTubeLogger

from datasets.sitcom3D import Sitcom3DDataset
from datasets.blender import BlenderDataset
from datasets.replica import ReplicaDataset
from datasets.front import ThreeDFrontDataset
from datasets.ray_utils import get_rays_batch_version

# models
from models.nerflet import Nerflet
from models.rendering_nerflet import render_rays

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
import json

import wandb
from eval_nerfletw import render_to_path
import numpy as np

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class NerfletWSystem(LightningModule):
    def __init__(self, hparams, eval_only=False):
        super().__init__()
        # self.hparams = hparams
        self.save_hyperparameters(hparams)

        use_mask_loss = True
        if 'use_mask_loss' in hparams:
            use_mask_loss = hparams.use_mask_loss
        self.loss = loss_dict['nerfletw'](weight_coverage_loss=hparams.weight_coverage_loss,
                                          use_mask_loss=hparams.use_mask_loss, label_only=hparams.label_only)

        self.models_to_train = {}
        self.embeddings = {}
        if hparams.encode_a:
            self.embedding_a = torch.nn.Embedding(hparams.N_vocab, hparams.N_a)
            self.embeddings['a'] = self.embedding_a
            self.models_to_train['embedding_a'] = self.embedding_a
        if hparams.encode_t:
            self.embedding_t = torch.nn.Embedding(hparams.N_vocab, hparams.N_tau)
            self.embeddings['t'] = self.embedding_t
            self.models_to_train['embedding_t'] = self.embedding_t


        self.models_mm_to_train = []
        self.eval_only = eval_only
        self.scene_bbox = None
        self.init_datasets()
        self.nerflet = Nerflet(D=hparams.num_hidden_layers, W=hparams.dim_hidden_layers, skips=hparams.skip_layers,
                               N_emb_xyz=hparams.N_emb_xyz, N_emb_dir=hparams.N_emb_dir,
                               encode_a=hparams.encode_a, encode_t=hparams.encode_t,
                               predict_label=hparams.predict_label, num_classes=hparams.num_classes,
                               M=hparams.num_parts, disable_ellipsoid=hparams.disable_ellipsoid,
                               scale_min=hparams.scale_min, scale_max=hparams.scale_max,
                               use_spread_out_bias=hparams.use_spread_out_bias, bbox=self.scene_bbox,
                               label_only=hparams.label_only, disable_tf=hparams.disable_tf,
                               sharpness=hparams.sharpness, predict_density=hparams.predict_density)
        self.models = {'nerflet': self.nerflet}
        self.models_to_train.update(self.models)

        self.near_min = 0.1
        self.appearance_id = None
        self.height = None

        num_params = 0
        for model_name, model in self.models_to_train.items():
            num_model_params = count_parameters(model)
            print(f"# of params in {model_name}----{num_model_params}")
            num_params += num_model_params
        print(f"Number of parameters in total: {num_params}")

    def init_datasets(self):
        if self.hparams.dataset_name == 'sitcom3D':
            dataset = Sitcom3DDataset
            kwargs = {'environment_dir': self.hparams.environment_dir,
                      'near_far_version': self.hparams.near_far_version}
            # kwargs['img_downscale'] = self.hparams.img_downscale
            kwargs['val_num'] = self.hparams.num_gpus
            kwargs['use_cache'] = self.hparams.use_cache
            kwargs['num_limit'] = self.hparams.num_limit
            self.train_dataset = dataset(split='train',
                                         img_downscale=self.hparams.img_downscale,
                                         near=self.hparams.near, **kwargs)
            self.val_dataset = dataset(split='val', img_downscale=self.hparams.img_downscale_val,
                                       near=self.hparams.near, **kwargs)
            self.test_dataset = dataset(split='test_train', img_downscale=self.hparams.img_downscale,
                                        near=self.hparams.near, **kwargs)
            self.scene_bbox = self.train_dataset.bbox
        elif self.hparams.dataset_name == 'blender':
            self.train_dataset = BlenderDataset(root_dir=self.hparams.environment_dir,
                                                img_wh=self.hparams.img_wh, split='train')
            self.val_dataset = BlenderDataset(root_dir=self.hparams.environment_dir,
                                              img_wh=self.hparams.img_wh, split='val')
        elif self.hparams.dataset_name == 'replica':
            self.train_dataset = ReplicaDataset(root_dir=self.hparams.environment_dir,
                                                img_downscale=self.hparams.img_downscale, split='train',
                                                things_only=self.hparams.things_only)
            self.val_dataset = ReplicaDataset(root_dir=self.hparams.environment_dir,
                                              img_downscale=self.hparams.img_downscale, split='val',
                                              things_only=self.hparams.things_only)
        elif self.hparams.dataset_name == '3dfront':
            self.train_dataset = ThreeDFrontDataset(root_dir=self.hparams.environment_dir,
                                                    img_downscale=self.hparams.img_downscale, split='train',
                                                    near=self.hparams.near, far=self.hparams.far)
            self.val_dataset = ThreeDFrontDataset(root_dir=self.hparams.environment_dir,
                                                  img_downscale=self.hparams.img_downscale, split='val',
                                                  near=self.hparams.near, far=self.hparams.far)

    def load_from_ckpt_path(self, ckpt_path):
        """TODO(ethan): move this elsewhere
        """
        load_ckpt(self.embedding_a, ckpt_path, model_name="embedding_a")
        load_ckpt(self.embedding_t, ckpt_path, model_name="embedding_t")
        load_ckpt(self.nerflet, ckpt_path, model_name="nerflet")
        load_ckpt(self.learn_f, ckpt_path, model_name="learn_f")
        load_ckpt(self.learn_p, ckpt_path, model_name="learn_p")

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

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
                            self.hparams.N_importance,
                            self.hparams.chunk,  # chunk size is effective in val mode
                            self.train_dataset.white_back,
                            predict_density=self.hparams.predict_density,
                            test_time=version == "val")

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models_to_train)
        self.optimizer.add_param_group({"params": get_parameters(self.models_mm_to_train), "lr": self.hparams.pose_lr})
        scheduler = get_scheduler(self.hparams, self.optimizer)
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=8,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=8,
                          batch_size=1,  # validate one image (H*W rays) at a time
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          shuffle=False,
                          num_workers=8,
                          batch_size=1,
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
        print(self.global_step)
        rays, ray_mask = self.rays_from_batch(batch)
        if 'ts2' in batch:
            ts = batch['ts2']
        else:
            ts = batch['ts']
        rgbs, gt_labels = batch['rgbs'], batch['labels']
        results = self.forward(rays, ts, version="train")
        loss_d = self.loss(results, rgbs, gt_labels, ray_mask,
                           self.hparams.encode_t, self.hparams.predict_label, self.hparams.loss_pos_ray_ratio)
        loss = sum(l for l in loss_d.values())

        with torch.no_grad():
            # typ = 'fine' if 'rgb_fine' in results else 'coarse'
            if self.hparams.encode_t:
                psnr_ = psnr(results['combined_rgb_map'], rgbs)
            else:
                psnr_ = psnr(results['static_rgb_map'], rgbs)

        self.log('lr', get_learning_rate(self.optimizer))
        self.log('train/loss', loss)
        for k, v in loss_d.items():
            self.log(f'train/{k}', v, prog_bar=True)
        self.log('train/psnr', psnr_, prog_bar=True)

        # wandb log
        dict_to_log = {}
        dict_to_log['lr'] = get_learning_rate(self.optimizer)
        dict_to_log['train/loss'] = loss
        for k, v in loss_d.items():
            dict_to_log[f"train/{k}"] = v
        dict_to_log['train/psnr'] = psnr_
        wandb.log(dict_to_log)

        return loss

    def validation_step(self, batch, batch_nb):
        rays, ray_mask = self.rays_from_batch(batch)
        if 'ts2' in batch:
            ts = batch['ts2']
        else:
            ts = batch['ts']
        rgbs, gt_labels = batch['rgbs'], batch['labels']
        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        ts = ts.squeeze()  # (H*W)
        gt_labels = gt_labels.squeeze()
        results = self.forward(rays, ts, version="val")
        loss_d = self.loss(results, rgbs, gt_labels, ray_mask,
                           self.hparams.encode_t, self.hparams.predict_label, self.hparams.loss_pos_ray_ratio)
        loss = sum(l for l in loss_d.values())
        self.log('val/loss', loss, prog_bar=True)
        dict_to_log = {'val/loss': loss}
        for k, v in loss_d.items():
            dict_to_log[f"val/{k}"] = v

        if self.hparams.encode_t:
            psnr_ = psnr(results['combined_rgb_map'], rgbs)
        else:
            psnr_ = psnr(results['static_rgb_map'], rgbs)
        dict_to_log['val/psnr'] = psnr_
        self.log('val/psnr', psnr_, prog_bar=True)
        wandb.log(dict_to_log)

        print("Rendering some sample images from the training set...")
        render_img_name = f"s={self.global_step:06d}_i={batch_nb:03d}"
        render_path = os.path.join(self.hparams.render_dir, f"{render_img_name}.png")
        np.random.seed(19)
        label_colors = np.random.rand(self.hparams.num_classes, 3)
        part_colors = np.random.rand(self.hparams.num_parts, 3)
        _, res_img = render_to_path(path=render_path, dataset=self.test_dataset,
                                    idx=batch_nb, models=self.models, embeddings=self.embeddings,
                                    config=self.hparams, label_colors=label_colors, part_colors=part_colors)
        wd_img = wandb.Image(res_img, caption=f"{render_img_name}")
        wandb.log({f"Renderings_id={batch_nb}": wd_img})

        return dict_to_log


def main(hparams):
    save_dir = os.path.join(hparams.environment_dir, "runs")

    if hparams.resume_name:
        exp_name = hparams.resume_name
    else:
        exp_name = hparams.exp_name
        exp_name = f"{exp_name}_np={hparams.num_parts}"
        if hparams.predict_label:
            exp_name += '_label'

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

    dir_path = os.path.join(save_dir, exp_name, f"version_{version}")
    os.makedirs(dir_path, exist_ok=True)
    config = vars(hparams)
    config_save_path = os.path.join(dir_path, 'config.json')
    json_obj = json.dumps(config, indent=2)
    with open(config_save_path, 'w') as f:
        f.write(json_obj)

    run = wandb.init(
        # Set the project where this run will be logged
        project="my_nerfletsW",
        name=exp_name,
        # Track hyperparameters and run metadata
        config=config,
        group="total"
    )


    hparams.render_dir = f"{dir_path}/render_logs"
    os.makedirs(hparams.render_dir, exist_ok=True)
    system = NerfletWSystem(hparams)

    checkpoint_filepath = f'{dir_path}/ckpts'
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_filepath,
                                          monitor='val/psnr',
                                          mode='max',
                                          save_top_k=-1,
                                          every_n_epochs=1)

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
                      )

    trainer.fit(system)


if __name__ == '__main__':
    hparams = get_opts()
    main(hparams)
