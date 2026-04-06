# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Plotting utils
"""

import contextlib
import math
import os
from copy import copy
from pathlib import Path
from urllib.error import URLError

import cv2
import numpy as np
import pandas as pd
# import seaborn as sn
import torch
from PIL import Image, ImageDraw, ImageFont

from sas_yolov7_seg.utils import TryExcept, threaded
from sas_yolov7_seg.utils.general import (CONFIG_DIR, FONT, LOGGER, check_font, check_requirements, clip_coords, increment_path,
                           is_ascii, xywh2xyxy, xyxy2xywh)
from utils.metrics import fitness

# Settings
RANK = int(os.getenv('RANK', -1))


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


