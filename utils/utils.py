from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2
import re


def parse_cfg(cfg_path):
    """
    Parses the yolov3 layer configuration file

    Arguments:
        cfg_path {str} -- cfg file path

    Returns a list of blocks. Each blocks describes a block in the neural network
    to be built. Block is represented as a dictionary in the list
    """

    blocks = []
    with open(cfg_path, "r") as f:
        # get rid of comments and \n
        lines = (line.strip() for line in f if not line.startswith("#"))
        # get rid of whitespaces
        lines = (re.sub("\s+", "", line) for line in lines if len(line) > 0)

        for line in lines:
            if line.startswith("["):  # start of a new block
                blocks.append({})
                blocks[-1]["type"] = line[1:-1]
            else:
                key, value = line.split("=")
                blocks[-1][key] = value
        return blocks


def select_device(force_cpu=False):
    if force_cpu:
        cuda = False
        device = torch.device("cpu")
    else:
        cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if cuda else "cpu")

    print("Using {} {}".format(device.type,
                               torch.cuda.get_device_properties(0) if cuda else ""))
    return device


def non_max_suppression(prediction, conf_thres, nms_thres):
    """
    Remove detections with lower object confidence score than 'conf_thres' and
    performs Non-maximum Suppression to further filter detections.

    Arguments:
        prediction {[type]} -- [description]
        conf_thres {[type]} -- [description]
        nms_thres {[type]} -- [description]

    Returns:
        detections with shape: (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # Convert (center_x, center_y, w, h) to
    # top-left corner (x1,y1) and bottom-right corner (x2,y2)
    box_corner = prediction.new(prediction.shape)
    # top-left corner
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    # bottom-right corner
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        conf_mask = (pred[:, 4] >= conf_thres).squeeze()
        pred = pred[conf_mask]
        # if none are remaining --> process next image
        if pred.size(0) == 0:
            continue

        class_prob, class_pred = torch.max(pred[:, 5:], 1, keepdim=True)

        detections = torch.cat(
            (pred[:, :5], class_prob.float(), class_pred.float()), 1)
        # loop over all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda(prediction.device)

        for c in unique_labels:
            # Get the detections with class c
            detections_class = detections[detections[:, -1] == c]
            # Sort the detections such that the entry with the maximum objectness
            # confidence is at the top
            _, conf_sort_index = torch.sort(
                detections_class[:, 4] * detections_class[:, 5], descending=True)
            detections_class = detections_class[conf_sort_index]

            # Non-maximum suppresion
            det_max = []
            while detections_class.size(0):
                det_max.append(detections_class[:1])
                if len(detections_class) == 1:
                    break
                ious = bbox_iou(det_max[-1], detections_class[1:])
                detections_class = detections_class[1:][ious < nms_thres]

            det_max = torch.cat(det_max)
            # Add max detections to outputs
            output[i] = det_max if output[i] is None else torch.cat(
                (output[i], det_max))

    return output


#=============================


def bbox_iou(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Get the coordinates of the intersection rectangle
    # left-top coordinates of intersection area
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    # right-bottom coordinates of intersection area
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * \
        torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)

    # Union area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def load_classes(namesfile):
    with open(namesfile, "r") as fp:
        names = [line.strip() for line in fp]
        return names


def letterbox_image(image, inp_dim):
    """Resize image with unchanged aspect ratio

    Arguments:
        img {ndarray} -- [description]
        inp_dim {tuple} -- [description]

    Returns:
        canvas {torch.Tensor}
    """
    shape = image.shape[0:2]  # [height, width]
    w, h = inp_dim
    scale = float(h) / max(shape)  # scale factor = old / new
    new_shape = (round(shape[1] * scale), round(shape[0] * scale))
    dh = (h - new_shape[1]) / 2  # height padding
    dw = (h - new_shape[0]) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    resized_image = cv2.resize(image, new_shape, interpolation=cv2.INTER_CUBIC)
    img = cv2.copyMakeBorder(resized_image, top, bottom, left,
                             right, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    return img


def rescale_coords(img_size, detections, img0_shape):
    scale = img_size / max(img0_shape)  # scale = old / new
    padx = (img_size - img0_shape[1] * scale) / 2
    pady = (img_size - img0_shape[0] * scale) / 2
    detections[:, [0, 2]] -= padx
    detections[:, [1, 3]] -= pady
    detections[:, :4] /= scale
    detections[:, :4] = torch.clamp(detections[:, :4], 0)
    return detections


def draw_bbox(img, coords, label=None, color=None, lw=None):
    # line_width = lw or round(0.002 * max(img.shape[0:2])) + 1
    color = color or np.random.randint(0, 255, 3).tolist()
    ltp = (int(coords[0]), int(coords[1]))  # left-top point
    rbp = (int(coords[2]), int(coords[3]))  # right-bottom point
    cv2.rectangle(img, ltp, rbp, color, 1)

    if label:
        t_size = cv2.getTextSize(label, 0, 1, 1)[0]
        # rbp = ltp[0] + t_size[0] + 3, ltp[1] - t_size[1] - 4
        # cv2.rectangle(img, ltp, rbp, color, -1)  # filled
        cv2.putText(img, label, (ltp[0], ltp[1]-4),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, color, thickness=1, lineType=cv2.LINE_AA)

    return img
