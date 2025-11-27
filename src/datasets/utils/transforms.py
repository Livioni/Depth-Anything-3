# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DUST3R default transforms
# --------------------------------------------------------
from src.utils.image import ImgNorm
from src.datasets.utils.augmentation import get_image_augmentation
# define the standard image transforms

ColorJitter = get_image_augmentation()

