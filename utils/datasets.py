import os
import cv2
import numpy as np
import torch

from utils.utils import letterbox_image

import matplotlib.pyplot as plt


class LoadImages():
    def __init__(self, path, img_size=(416, 416)):
        if os.path.isdir(path):
            image_format = [".jpg", ".jpeg", ".png"]
            path = os.path.realpath(path)
            self.files = [os.path.join(path, file) for file in os.listdir(path) if os.path.splitext(file)[
                1].lower() in image_format]
        elif os.path.isfile(path):
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

        img_path = self.files[self.count]

        # Padded resize
        img0 = cv2.imread(img_path)
        img = letterbox_image(img0, (self.width, self.height))

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)

        return img_path, img / 255.0

    def __len__(self):
        return self.num_images
