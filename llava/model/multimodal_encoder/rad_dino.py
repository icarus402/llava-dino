"""WIP"""

import json
import os
import torch

from transformers import AutoImageProcessor, AutoModel
import torchvision.transforms as transforms


LLAVARAD_HF_REPO = "microsoft/llava-rad"

class Processor:
    def __init__(self, image_size=518) -> None:
        self.img_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5307, 0.5307, 0.5307], 
                               std=[0.2583, 0.2583, 0.2583])
        ])

    def preprocess(self, image, return_tensors="pt"):
        if return_tensors != "pt":
            raise NotImplementedError
        return {"pixel_values": [self.transform(image)]}
    
class RadDinoTower(torch.nn.Module):
    def __init__(self, vision_tower, args, local_model_path=None, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.local_model_path = local_model_path

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = vision_tower
        
    def load_model(self):
        self.vision_tower = AutoModel.from_pretrained(self.local_model_path , local_files_only=True)
        self.image_processor = Processor()
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.vision_tower.parameters()).device

    @property
    def config(self):
        raise NotImplementedError

    @property
    def hidden_size(self):
        return self.vision_tower.hidden_size

    @property
    def num_patches(self):
        return self.vision_tower.num_patches

    