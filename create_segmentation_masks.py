import os
import argparse
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

DRIVABLE_COLOR_TO_CLASS = {
    (np.uint8(0), np.uint8(0), np.uint8(0)): 0,
    (np.uint8(219), np.uint8(94), np.uint8(86)): 1,
    (np.uint8(86), np.uint8(211), np.uint8(219)): 2,
}

LANE_COLOR_TO_CLASS = {(np.uint8(255), np.uint8(255), np.uint8(255)): 0}

TASK_TO_COLOR_DICT = {"drivable": DRIVABLE_COLOR_TO_CLASS, "lane": LANE_COLOR_TO_CLASS}


def write_class_masks(args):
    for img_path in tqdm(os.listdir(args.colormaps_dir)):
        mask = np.array(
            Image.open(os.path.join(args.colormaps_dir, img_path)).convert("RGB")
        )
        h, w = mask.shape[:2]
        if args.task == "lane":
            class_mask = np.ones((h, w), dtype=np.uint8)
        else:
            class_mask = np.zeros((h, w), dtype=np.uint8)

        for rgb, class_idx in TASK_TO_COLOR_DICT[args.task].items():
            class_mask[np.all(mask == rgb, axis=-1)] = class_idx

        cv2.imwrite(os.path.join(args.output_dir, img_path), class_mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--colormaps_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    write_class_masks(args)
