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
from datasets.ray_utils import get_rays_batch_version

# models
from models.nerflet import Nerflet
from models.rendering_nerflet import render_rays
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
import json

class NerfletWSystem(LightningModule):
    def __init__(self, hparams, eval_only=False):
        super().__init__()
        # self.hparams = hparams
        self.save_hyperparameters(hparams)

        self.loss = loss_dict['nerfletw'](coef=1)

        self.models_to_train = []

        self.embeddings = {}
        if hparams.encode_a:
            self.embedding_a = torch.nn.Embedding(hparams.N_vocab, hparams.N_a)
            self.embeddings['a'] = self.embedding_a
            self.models_to_train += [self.embedding_a]
        if hparams.encode_t:
            self.embedding_t = torch.nn.Embedding(hparams.N_vocab, hparams.N_tau)
            self.embeddings['t'] = self.embedding_t
            self.models_to_train += [self.embedding_t]

        self.nerflet = Nerflet(N_emb_xyz=hparams.N_emb_xyz, N_emb_dir=hparams.N_emb_dir,
                               encode_a=hparams.encode_a, encode_t=hparams.encode_t,
                               predict_label=hparams.predict_label,
                               num_classes=hparams.num_classes,
                               M=hparams.num_parts, disable_ellipsoid=hparams.disable_ellipsoid)
        self.models = {'nerflet': self.nerflet}
        self.models_to_train += [self.models]
        self.models_mm_to_train = []
        self.eval_only = eval_only
        self.init_datasets()

        self.near_min = 0.1
        self.appearance_id = None
        self.height = None

    def init_datasets(self):
        if self.hparams.dataset_name == 'sitcom3D':
            dataset = Sitcom3DDataset
            kwargs = {'environment_dir': self.hparams.environment_dir,
                      'near_far_version': self.hparams.near_far_version}
            # kwargs['img_downscale'] = self.hparams.img_downscale
            kwargs['val_num'] = self.hparams.num_gpus
            kwargs['use_cache'] = self.hparams.use_cache
            kwargs['num_limit'] = self.hparams.num_limit
            self.train_dataset = dataset(split='train' if not self.eval_only else 'val',
                                         img_downscale=self.hparams.img_downscale, **kwargs)
            self.val_dataset = dataset(split='val', img_downscale=self.hparams.img_downscale_val, **kwargs)

            # if self.is_learning_pose():
            # NOTE(ethan): self.train_dataset.poses is all the poses, even those in the val dataset
            train_poses = torch.FloatTensor(self.train_dataset.poses)  # (N, 3, 4)
            train_poses = torch.cat([train_poses, torch.zeros_like(train_poses[:, 0:1, :])], dim=1)
            train_poses[:, 3, 3] = 1.0
            self.learn_f = LearnFocal(len(train_poses), self.hparams.learn_f).cuda()
            self.learn_p = LearnPose(len(train_poses), self.hparams.learn_r, self.hparams.learn_t,
                                     init_c2w=train_poses).cuda()
            self.models_mm = {}
            self.models_mm["learn_f"] = self.learn_f
            self.models_mm["learn_p"] = self.learn_p
            self.models_mm_to_train += [self.learn_f]
            self.models_mm_to_train += [self.learn_p]
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
            directions = batch['directions'].squeeze()
            ts2 = batch['ts2'].squeeze()
            f = self.models_mm["learn_f"]()[ts2]
            cameras, c2w_delta = self.models_mm["learn_p"]()
            c2w = cameras[ts2]
            directions_new = directions / f
            rays_o, rays_d = get_rays_batch_version(directions_new, c2w)
            # NOTE(ethan): nerfmm won't work when near_far_version == "cam" due to ray near/far computation
            nears, fars, ray_mask = self.train_dataset.get_nears_fars_from_rays_or_cam(rays_o, rays_d, c2w=None)
            rays_t = batch['ts'].squeeze().unsqueeze(-1)  # hacky way to get [N, 1]
            rays = torch.cat([rays_o, rays_d, nears, fars, rays_t], 1)
            ray_mask = ray_mask.squeeze()
        else:
            rays = batch['rays'].squeeze()
            ray_mask = batch['ray_mask'].squeeze()
        return rays, ray_mask

    def training_step(self, batch, batch_nb):
        rays, ray_mask = self.rays_from_batch(batch)
        rgbs, ts, gt_labels = batch['rgbs'], batch['ts'], batch['labels']
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

        return loss

    def validation_step(self, batch, batch_nb):
        rays, ray_mask = self.rays_from_batch(batch)
        rgbs, ts, gt_labels = batch['rgbs'], batch['ts'], batch['labels']
        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        ts = ts.squeeze()  # (H*W)
        gt_labels = gt_labels.squeeze()
        results = self.forward(rays, ts, version="val")
        loss_d = self.loss(results, rgbs, gt_labels, ray_mask,
                           self.hparams.encode_t, self.hparams.predict_label, self.hparams.loss_pos_ray_ratio)
        loss = sum(l for l in loss_d.values())
        log = {'val_loss': loss}
        # typ = 'fine' if 'rgb_fine' in results else 'coarse'

        if batch_nb == 0:
            if self.hparams.dataset_name == 'sitcom3D':
                WH = batch['img_wh']
                W, H = WH[0, 0].item(), WH[0, 1].item()
            else:
                W, H = self.hparams.img_wh
            vis_data = {}
            vis_data.update(results)
            vis_data["rgbs"] = rgbs
            vis_data["img_wh"] = [W, H]
            # image = get_image_summary_from_vis_data(vis_data)
            # self.logger.experiment.add_image('val/GT_pred_depth', image, self.global_step)

        if self.hparams.encode_t:
            psnr_ = psnr(results['combined_rgb_map'], rgbs)
        else:
            psnr_ = psnr(results['static_rgb_map'], rgbs)
        log['val_psnr'] = psnr_

        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()

        self.log('val/loss', mean_loss)
        self.log('val/psnr', mean_psnr, prog_bar=True)



def main(hparams):
    save_dir = os.path.join(hparams.environment_dir, "runs")

    if hparams.resume_name:
        timedatestring = hparams.resume_name
    else:
        timedatestring = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")

    # print(hparams.exp_name)

    # following pytorch lightning convention here
    logger = TestTubeLogger(save_dir=save_dir,
                            name=timedatestring,
                            debug=False,
                            create_git_tag=False,
                            log_graph=False)
    version = logger.experiment.version
    if not isinstance(version, int):
        version = 0

    if hparams.resume_name:
        assert hparams.ckpt_path is not None

    # following pytorch lightning convention here
    dir_path = os.path.join(save_dir, timedatestring, f"version_{version}")
    config = vars(hparams)
    config_save_path = os.path.join(dir_path, 'config.json')
    json_obj = json.dumps(config, indent=2)
    with open(config_save_path, 'w') as f:
        f.write(json_obj)

    system = NerfletWSystem(hparams)

    checkpoint_filepath = os.path.join(f'{dir_path}/ckpts')
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_filepath,
                                          monitor='val/psnr',
                                          mode='max',
                                          save_top_k=-1)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      callbacks=[checkpoint_callback],
                      resume_from_checkpoint=hparams.ckpt_path,
                      logger=logger,
                      weights_summary=None,
                      progress_bar_refresh_rate=hparams.refresh_every,
                      gpus=hparams.num_gpus,
                      accelerator='ddp' if hparams.num_gpus > 1 else None,
                      num_sanity_val_steps=1,
                      val_check_interval=int(4000),  # run val every int(X) batches
                      benchmark=True,
                      accumulate_grad_batches=hparams.accumulate_grad_batches
                      )

    trainer.fit(system)


if __name__ == '__main__':
    hparams = get_opts()
    main(hparams)
