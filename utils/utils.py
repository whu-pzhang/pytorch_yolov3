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


def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)

    return tensor_res


def bbox_iou(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
        torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    # Union area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def write_results(prediction, confidence, num_classes, nms_conf=0.4):
    """[summary]

    Arguments:
        prediction {[type]} -- [description]
        confidence {[type]} -- [description]
        num_classes {int} -- number of classes

    Keyword Arguments:
        nms_conf {float} -- NMS IoU threshold (default: {0.4})
    """
    # set the values which below a threshold to zero
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction *= conf_mask

    box_corner = prediction.new(prediction.shape)
    # top-left corner
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    # bottom-right corner
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    batch_size = prediction.size(0)
    write = False

    for ind in range(batch_size):
        image_pred = prediction[ind]

        max_conf, max_conf_score = torch.max(image_pred[:, 5:5+num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:, :5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)

        non_zero_idx = torch.nonzero(image_pred[:, 4])
        try:
            image_pred_ = image_pred[non_zero_idx, :].view(-1, 7)
        except:
            continue

        if image_pred_.shape[0] == 0:
            continue

        # Get the various classes detected in the image
        img_classes = unique(image_pred_[:, -1])

        for cls in img_classes:
            # perform NMS

            # Get the detections with one particalar class
            cls_mask = image_pred_ * \
                (image_pred_[:, -1] == cls).float().unsqueeze(1)

            class_mask_idx = torch.nonzero(cls_mask[:, -2]).squeeze()
            image_pred_class = image_pred_[class_mask_idx].view(-1, 7)

            # Sort the detections such that the entry with the maximum objectness
            # confidence is at the top
            conf_sort_index = torch.sort(
                image_pred_class[:, 4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)  # Number of detections

            for i in range(idx):
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(
                        0), image_pred_class[i+1:])
                except ValueError:
                    break
                except IndexError:
                    break

                # Zero out all the detections that have IoU > threshold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask

                # Remove the non-zero entries
                non_zero_idx = torch.nonzero(image_pred_class[:, 4]).squeeze()
                image_pred_class = image_pred_class[non_zero_idx].view(-1, 7)

            batch_ind = image_pred_class.new(
                image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class

            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))

    try:
        return output
    except:
        return 0


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
    img_h, img_w = image.shape[0], image.shape[1]
    w, h = inp_dim
    scale = min(w/img_w, h/img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    resized_image = cv2.resize(
        img, (new_h, new_w), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)
    canvas[(h-new_h)//2:(h-new_h)//2+new_h,
           (w-new_w) // 2:(w-new_w)//2+new_w, :] = resized_image

    return canvas


def prep_image(img, inp_dim):
    """Prepare image for inputting

    Arguments:
        img {ndarray} -- [description]
        inp_dim {torch.Tensor} -- [description]
    """
    img = cv2.resize(img, (inp_dim, inp_dim))
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img
