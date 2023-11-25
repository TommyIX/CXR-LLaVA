import os
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
import torch
import open_clip
from .clip_encoder import CLIPVisionTower
import torch.nn as nn
import torch.nn.functional as F


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)

    if ("BiomedCLIP" in vision_tower) or is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')