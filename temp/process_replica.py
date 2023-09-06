from pathlib import Path
import argparse
import imageio
import numpy as np
from PIL import Image
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    splits = ['train', 'val', 'test']

    label_info_path = data_dir / 'info_semantic.json'
    with open(label_info_path, 'r') as f:
        label_info = json.load(f)
    stuff_classes = []
    classes = label_info['classes']
    stuff_keywords = ['ceiling', 'floor', 'stair', 'wall', 'window']
    for class_info in classes:
        class_id = class_info['id']
        for keyword in stuff_keywords:
            if keyword in class_info['name']:
                stuff_classes.append(class_id)
                break

    for split in splits:
        split_dir = data_dir / split
        img_dir = split_dir / 'rgb'
        label_dir = split_dir / 'label'
        img_out_dir = split_dir / 'rgb_objects'; img_out_dir.mkdir(exist_ok=True)
        label_out_dir = split_dir / 'label_objects'; label_out_dir.mkdir(exist_ok=True)

        for img_path in img_dir.iterdir():
            img = imageio.imread(img_path)
            img = np.array(img)

            t = int(img_path.stem.split('_')[-1])
            label_map_path = label_dir / f"semantic_class_{t}.png"
            label_map = imageio.imread(label_map_path)
            label_map = np.array(label_map)

            for stuff_class in stuff_classes:
                stuff_mask = label_map == stuff_class
                label_map[stuff_mask] = 0
                img[stuff_mask] = 255

            img_out_path = img_out_dir / img_path.name
            imageio.imwrite(img_out_path, img)
            label_out_path = label_out_dir / label_map_path.name
            imageio.imwrite(label_out_path, label_map)


