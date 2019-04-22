from pathlib import Path
import cv2
import numpy as np
import torch

from utils.utils import letterbox_image

import matplotlib.pyplot as plt


class LoadImages():
    def __init__(self, path, img_size=(416, 416)):
        path = Path(path)
        if path.is_dir():
            image_format = [".jpg", ".jpeg", ".png"]
            path = path.resolve()
            self.files = [file for file in path.glob("*") if file.suffix in image_format]
        elif path.is_file():
            self.files = [path]

        self.num_images = len(self.files)
        self.width, self.height = img_size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.num_images:
            raise StopIteration

        img_path = str(self.files[self.count])

        # Padded resize
        img0 = cv2.imread(img_path)
        img = letterbox_image(img0, (self.width, self.height))

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1) / 255.0
        # img = np.ascontiguousarray(img, dtype=np.float32)

        return img_path, img, img0

    def __len__(self):
        return self.num_images
