import os
import os.path as osp
import yaml

from types import SimpleNamespace
from yolox.exp import get_exp
import torch
from yolox.detector import Predictor
from reid.reid_extractor import FeatureExtractor
import logging
from PIL import Image

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def load_cfg(cfg_path='setting.yaml'):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    cfg['TRACKER'] = str(cfg['TRACKER'])
    cfg['DIRS_TXT'] = str(cfg['DIRS_TXT'])
    cfg['save_result'] = cfg['save_result']
    cfg['save_image'] = cfg['save_image']
    cfg['exp_file'] = str(cfg['exp_file'])
    cfg['fps'] = int(cfg['fps'])
    cfg['device'] = str(cfg['device'])

    cfg['reid_backbone']  = str(cfg['reid_backbone'])
    cfg['reid_ckpt'] = str(cfg['reid_ckpt'])
    cfg['det_conf'] = float(cfg['det_conf'])
    cfg['det_nms'] = float(cfg['det_nms'])
    cfg['det_aspect_ratio'] = float(cfg['det_aspect_ratio'])
    cfg['det_min_area'] = int(cfg['det_min_area'])

    cfg['track_high_th'] = float(cfg['track_high_th'])
    cfg['track_low_th'] = float(cfg['track_low_th'])
    cfg['track_new_th'] = float(cfg['track_new_th'])
    cfg['track_buffer'] = int(cfg['track_buffer'])
    cfg['track_match_th'] = float(cfg['track_match_th'])
    cfg['track_p_th'] = float(cfg['track_p_th'])
    cfg['track_vc_th'] = float(cfg['track_vc_th'])
    cfg['track_vf_th'] = float(cfg['track_vf_th'])
    cfg['track_b_th'] = float(cfg['track_b_th'])

    cfg['with_reid'] = cfg['with_reid']
    cfg['EIoU'] = cfg['EIoU']

    cfg['vp_dga'] = cfg['vp_dga']
    cfg['ris'] = cfg['ris']
    
    cfg['num_threads'] = int(cfg.get('num_threads', 4))
    cfg['num_workers'] = int(cfg.get('num_workers', 4))

    cfg = SimpleNamespace(**cfg)

    return cfg


def setup_logger(record_dir):
    log_file = os.path.join(record_dir, 'eval.log')
    logger = None
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def sub_exp_ini(cfg, output_dir, d_path, logger):
    d_name = os.path.basename(os.path.dirname(d_path))
    vis_folder = osp.join(output_dir, d_name)
    if cfg.save_image:
        os.makedirs(vis_folder, exist_ok=True)

    # Check if images in the folder have the same size
    logger.info(f"Checking image sizes in {d_path}...")
    sizes = set()

    for filename in os.listdir(d_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
            img_path = os.path.join(d_path, filename)
            try:
                with Image.open(img_path) as img:
                    sizes.add(img.size)
            except Exception as e:
                logger.warning(f"Error loading {filename}: {e}")

    if len(sizes) == 1:
        img_size = sizes.pop()
        logger.info(f"All images have the same size: {img_size}")
        return vis_folder, (img_size[1], img_size[0])
    
    else:
        logger.warning(f"Images have different sizes: {sizes}")
        return None, None


def build_detector(cfg, logger):
    exp = get_exp(cfg.exp_file, cfg.det_name)
    if cfg.det_conf is not None:
        exp.test_conf = cfg.det_conf
    if cfg.det_nms is not None:
        exp.nmsthre = cfg.det_nms

    model = exp.get_model().to(cfg.device)
    model.eval()
    ckpt = torch.load(cfg.det_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    if cfg.fp16:
        model = model.half()  # to FP16

    detector = Predictor(
        logger,
        cfg,
        model, 
        exp, 
        cfg.device, 
        cfg.fp16)

    return detector


def build_reid(cfg, logger):
    extractor = FeatureExtractor(
        logger = logger,
        model_name=cfg.reid_backbone,
        weight_path=cfg.reid_ckpt,
        device=cfg.device
    )
    return extractor


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names
