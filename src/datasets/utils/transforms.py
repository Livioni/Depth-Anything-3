# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DUST3R default transforms
# --------------------------------------------------------
import torchvision.transforms as tvf
from src.utils.image import ImgNorm

# define the standard image transforms
ColorJitter = tvf.Compose([
    tvf.ColorJitter(0.5, 0.5, 0.5, 0.1), 
    tvf.ToTensor(),
    tvf.Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
])

