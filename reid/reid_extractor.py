# From Torchreid REPO
from __future__ import absolute_import
import numpy as np
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image

from reid.tools import (
    build_model, check_isfile, load_pretrained_weights
)


class FeatureExtractor(object):

    def __init__(
        self,
        logger,
        model_name='',
        weight_path='',
        image_size=(384, 384),
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
        pixel_norm=True,
        device='cuda',
        verbose=True
    ):
        
        logger = logger
        logger.info("================BIULD ReID MODEL===============")   
        logger.info(f"MODEL NAME: {model_name}")
        logger.info(f"MODEL CKPT: {weight_path}")

        image_size = (384,384)

        # Build model
        model = build_model(
            model_name,
            num_classes=1,
            pretrained=not (weight_path and check_isfile(weight_path)),
            use_gpu=device.startswith('cuda')
        )

        model.eval()

        if weight_path and check_isfile(weight_path):
            load_pretrained_weights(logger, model, weight_path)

        # Build transform functions (Eval: only allow Resize & ToTensor)
        transforms = []
        transforms += [T.Resize(image_size)]
        transforms += [T.ToTensor()]
        if pixel_norm:
            transforms += [T.Normalize(mean=pixel_mean, std=pixel_std)]
        preprocess = T.Compose(transforms)

        to_pil = T.ToPILImage()

        device = torch.device(device)
        model.to(device)

        # Class attributes
        self.model = model
        self.preprocess = preprocess
        self.to_pil = to_pil
        self.device = device
        logger.info("===============================================")

    def __call__(self, input):
        if isinstance(input, list):
            images = []

            for element in input:
                if isinstance(element, str):
                    image = Image.open(element).convert('RGB')

                elif isinstance(element, np.ndarray): 
                    if (element.shape[0]!=0) & (element.shape[1]!=0) & (element.shape[2]==3):
                        image = self.to_pil(element)
                    else: 
                        element = np.zeros([1,1,3], dtype='uint8')
                        element.astype('uint8')
                        image = self.to_pil(element)

                else:
                    raise TypeError(
                        'Type of each element must belong to [str | numpy.ndarray]'
                    )

                image = self.preprocess(image)
                images.append(image)

            images = torch.stack(images, dim=0)
            images = images.to(self.device)

        elif isinstance(input, str):
            image = Image.open(input).convert('RGB')
            image = self.preprocess(image)
            images = image.unsqueeze(0).to(self.device)

        elif isinstance(input, np.ndarray):
            image = self.to_pil(input)
            image = self.preprocess(image)
            images = image.unsqueeze(0).to(self.device)

        elif isinstance(input, torch.Tensor):
            if input.dim() == 3:
                input = input.unsqueeze(0)
            images = input.to(self.device)

        else:
            raise NotImplementedError

        with torch.no_grad():
            features = self.model(images)
            features = F.normalize(features, dim=-1)

        return features
