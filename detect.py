from __future__ import division
import time
import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import cv2

import os
import argparse
from pathlib import Path
import pickle as pkl
import random

from darknet import Darknet
from utils.utils import *
from utils.datasets import *


import matplotlib.pyplot as plt


def arg_parse():

    parser = argparse.ArgumentParser()

    parser.add_argument("--images", dest="images",
                        help="Image / Directory containg images to perform detection upon",
                        default="imgs", type=str)
    parser.add_argument("--det", dest="det",
                        help="Image / Directory to store detections to",
                        default="det", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence",
                        help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh",
                        help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help="Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile",
                        default="weights/yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help="Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)

    return parser.parse_args()


def detect(cfg,
           weights,
           images,
           output="output",  # output folder
           img_size=416,
           conf_thres=0.3,
           nms_thres=0.45,
           save_images=True,
           webcam=False):
    device = select_device()

    if not os.path.exists(output):
        os.makedirs(output)

    model = Darknet(cfg, img_size)
    model.load_weights(weights)

    model.to(device).eval()

    if webcam:
        pass
    else:
        dataloader = LoadImages(images, img_size=(img_size, img_size))

    # Get classes and colors
    classes = load_classes("data/coco.names")
    num_classes = len(classes)
    colors = np.random.randint(0, 255, (num_classes, 3)).tolist()

    for i, (path, img, img0) in enumerate(dataloader):
        if webcam:
            pass
        else:
            print("{0:d}/{1:d} {2:s}:".format(i+1,
                                              len(dataloader), path), end=" ")
        # detect
        start = time.time()
        img = torch.from_numpy(img).float().unsqueeze(0).to(device)
        pred = model(img)
        # move boxes < conf_threshold
        pred = pred[pred[:, :, 4] > conf_thres]

        end = time.time()

        if len(pred) > 0:
            # NMS
            detections = non_max_suppression(
                pred.unsqueeze(0), conf_thres, nms_thres)[0]

            unique_classes = detections[:, -1].cpu().unique()
            for c in unique_classes:
                n = (detections[:, -1].cpu() == c).sum()
                print("{} {}".format(n, classes[int(c)]), end=", ")

            # Rescale boxes from img_size to the original image size
            detections = rescale_coords(img_size, detections, img0.shape)
            # Draw bboxes and labels
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                label = "{0:s} {1:.2f}".format(classes[int(cls_pred)], conf)
                img0 = draw_bbox(img0, [x1, y1, x2, y2],
                                 label=label, color=colors[int(cls_pred)])

        if save_images:
            save_path = str(Path(os.path.realpath(output)) / Path(path).name)
            cv2.imwrite(save_path, img0)
        if webcam:
            pass

        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="cfg/yolov3.cfg",
                        help="cfg file path")
    parser.add_argument('--weights', type=str,
                        default='weights/yolov3.weights', help='path to weights file')
    parser.add_argument('--images', type=str,
                        default='example', help='path to images')
    parser.add_argument('--img-size', type=int, default=32 *
                        13, help='size of each image dimension')
    parser.add_argument('--conf-thres', type=float,
                        default=0.50, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.45,
                        help='iou threshold for non-maximum suppression')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(
            opt.cfg,
            opt.weights,
            opt.images,
            img_size=opt.img_size,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres
        )