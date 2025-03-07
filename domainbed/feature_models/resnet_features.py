import torch
import torch.nn.functional as F
from torchvision import models, transforms
from ezcolorlog import root_logger as logger
import timm
from .base_encoder import BaseVisionTower

import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import os
from torchvision import transforms

class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape=224):
        super(ResNet, self).__init__()
        # if hparams['resnet18']:
        #     self.network = torchvision.models.resnet18(pretrained=True)
        #     self.n_outputs = 512
        # else:


        self.network = torchvision.models.resnet50(pretrained=True)
        # self.network = timm.create_model("resnet50.ram_in1k", pretrained=True)
        self.n_outputs = 2048

        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = 3
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        self.freeze_bn()
        # self.hparams = hparams
        # self.dropout = nn.Dropout(hparams['resnet_dropout'])

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.network(x)

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class ResNetVisionTower(BaseVisionTower):
    def __init__(self, vision_tower_name, config, delay_load=False):
        super(ResNetVisionTower, self).__init__(vision_tower_name, config, delay_load)

        self._config = config  # Use _config to avoid conflicts

        # Extract the base model name and interpolation size from the vision tower name
        self.vision_tower_name = vision_tower_name
        self._interp_size = config.get("interp_size", None)

        if not self.delay_load:
            self.load_model()
        elif self.unfreeze_mm_vision_tower:
            self.load_model()
        else:
            self.cfg_only = None  # ResNet doesn't have a separate config to load

        self.preprocess = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # Standard ImageNet mean
                    std=[0.229, 0.224, 0.225],  # Standard ImageNet std
                ),
            ]
        )

    def load_model(self, device_map=None):
        
        self.model = ResNet()


    def _feature_select(self, features):
        # For ResNet, features are usually taken from the final pooling layer
        return features

    def interpolate(self, features):
        if self._interp_size is None:
            return features

        # Perform interpolation if necessary
        features = F.interpolate(
            features.unsqueeze(1),  # Add channel dimension
            size=(self._interp_size, self._interp_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)  # Remove channel dimension
        return features

    def _forward(self, images):
        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            images = images.to(device="cuda")

            # Forward pass through ResNet50 up to the second-to-last layer
            x = images
            x = self.model(x)
            x = self._feature_select(x)

        return x

    def get_raw_features(self, images):
        return self._forward(images)


class ResNetFeatures:
    def __init__(self, config):
        self._config = config
        self._init_feature_encoder()

    def _init_feature_encoder(self):
        # Initialize ResNetVisionTower with parameters from config
        vision_tower_name = self._config.get("model_name", "resnet50")
        self.model = ResNetVisionTower(
            vision_tower_name, self._config, delay_load=False
        ).to(self._config.get("device", "cpu"))
        # self.preprocess = self.model.image_processor

    # def get_features(self, batch_images):
    #     # Preprocess images
    #     preprocessed_images = torch.stack([self.preprocess(img) for img in batch_images])
    #     # Forward pass through the vision tower to get features
    #     image_features = self.model(preprocessed_images)
    #     return image_features

    def get_raw_features(self, batch_images, prompt_embeds=None, time_step=None):
        # Forward pass through the vision tower to get features
        image_features = self.model(batch_images)
        return image_features
