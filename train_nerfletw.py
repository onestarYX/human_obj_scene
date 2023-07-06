import os
from opt import get_opts
import torch
from collections import defaultdict
from torch.utils.data import DataLoader
import torch.nn as nn
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

from datasets.sitcom3D import Sitcom3DDataset
from datasets.ray_utils import get_ray_directions, get_rays, get_rays_batch_version

# models
from models.nerf import (
    PosEmbedding,
    NeRF,
    Nerflet
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


class NerfletWSystem(nn.Module):
    def __init__(self, hparams, eval_only=False):
        super().__init__()
        # self.hparams = hparams
        self.save_hyperparameters(hparams)

        self.loss = loss_dict['nerfw'](coef=1)

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

        self.nerflet = Nerflet(in_channels_xyz=6 * hparams.N_emb_xyz + 3,
                            in_channels_dir=6 * hparams.N_emb_dir + 3,
                            predict_label=self.hparams.predict_label,
                            num_classes=self.hparams.num_classes)
        self.models = {'coarse': self.nerflet}
        if hparams.N_importance > 0:
            self.nerf_fine = NeRF('fine',
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
                      val_check_interval=int(1000),  # run val every int(X) batches
                      benchmark=True,
                      #   profiler="simple" if hparams.num_gpus == 1 else None
                      )

    trainer.fit(system)


if __name__ == '__main__':
    hparams = get_opts()
    main(hparams)
