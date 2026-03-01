import torch
import cv2
from yolox.data.data_augment import preproc
from yolox.utils import get_model_info, postprocess
import os.path as osp

class Predictor(object):
    def __init__(
        self,
        logger,
        cfg,
        model,
        exp,
        device=torch.device("cuda"),
        fp16=False
    ):
        self.model = model
        self.logger = logger
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        info, flops, params = get_model_info(model, exp.test_size)
        logger.info("================BIULD YOLO MODEL===============")   
        logger.info(f"MODEL CKPT: {cfg.det_ckpt}")
        logger.info('- params: {:,}'.format(params))
        logger.info('- flops: {:,}'.format(flops))
        logger.info("===============================================")

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            # timer.tic()
            outputs = self.model(img)
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
        return outputs, img_info