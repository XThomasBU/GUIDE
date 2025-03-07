from .clip_features import CLIPFeatures
from .dino_features import DINOFeatures
from .diffusion_features import DiffusionFeatures, DiffusionFeaturesDiT
from .mae_features import MAEFeatures
from .resnet_features import ResNetFeatures
import torch


class FeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.feature_model_dict = {
            "clip": CLIPFeatures,
            "dino": DINOFeatures,
            "diffusion": DiffusionFeatures,
            "dit": DiffusionFeaturesDiT,
            "mae": MAEFeatures,
            "resnet50": ResNetFeatures
        }
        self.feature_model = self.feature_model_dict[config["feature_model"].lower()](
            config
        )
        try:
            self.preprocess = self.feature_model.preprocess
        except:
            self.preprocess = None
        self.device = torch.device("cpu")  # Placeholder; will be set by accelerator

    def get_features(self, batch_images):
        return self.feature_model.get_features(batch_images)

    def get_raw_features(self, batch_images, prompt_embeds=None, time_step=None):
        return self.feature_model.get_raw_features(batch_images, prompt_embeds=None, time_step=time_step)
