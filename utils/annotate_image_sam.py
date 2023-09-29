import torch, detectron2
import os, json, cv2, random
import argparse
from pathlib import Path

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='data/3D_FRONT/MasterBedroom-2074/images')
    parser.add_argument('--output_dir', type=str, default='data/3D_FRONT/MasterBedroom-2074/masks')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("new_baselines/mask_rcnn_regnetx_4gf_dds_FPN_100ep_LSJ.py"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("new_baselines/mask_rcnn_regnetx_4gf_dds_FPN_100ep_LSJ.py")
    predictor = DefaultPredictor(cfg)

    for img_path in input_dir.iterdir():
        im = cv2.imread(str(img_path))
        out = predictor(im)
        vis = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out_vis = vis.draw_instance_predictions(out["instances"].to("cpu"))
        cv2.imshow('Output', out_vis.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()