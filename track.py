import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import random
import numpy as np
import torch
import cv2
import argparse
import time
import json

seed = 1337
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

cv2.setNumThreads(8)
cv2.ocl.setUseOpenCL(False)

import sys, os.path as osp
from tools.utils import load_cfg, sub_exp_ini, build_detector, build_reid, setup_logger
from tracker.track_engine import image_demo

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-object tracker demo"
    )
    parser.add_argument(
        "-c", "--config",
        default="setting.yaml",
        help="YAML SETTING FILEs"
    )
    return parser.parse_args()


def main(cfg, output_dir, d_path, det_path, detector, reid, logger):
    print(det_path)
    vis_folder, img_sizes = sub_exp_ini(cfg, output_dir, d_path, logger)
    logger.info(f"Results save to: {vis_folder}")
    image_demo(detector, reid, vis_folder, img_sizes, cfg, d_path, det_path, logger)


if __name__ == "__main__":

    args = parse_args()
    cfg = load_cfg(args.config)
    dirs = open(cfg.DIRS_TXT)

    timestamp_s = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    output_dir = osp.join('Results', timestamp_s)
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger(output_dir)
    logger.info(f"Logs save to: {output_dir}")
    logger.info("Configuration Settings:\n" + json.dumps(vars(cfg), indent=4))
    
    detector = build_detector(cfg, logger)
    reid = build_reid(cfg, logger)

    # print("reid type:", type(reid))
    # print("reid attributes:", dir(reid))
    # print(f"Debug {reid.model}")

    for d_path in dirs.readlines():
        d_path = d_path.strip()
        if cfg.public_tracking:
            det_path = d_path.replace("img1", "det/det.txt")
        else:
            det_path = None

        main(cfg, output_dir, d_path, det_path, detector, reid, logger)

