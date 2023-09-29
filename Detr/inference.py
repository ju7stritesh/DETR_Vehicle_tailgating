import math

from PIL import Image
import requests
import matplotlib.pyplot as plt
# %config InlineBackend.figure_format = 'retina'
#
# import ipywidgets as widgets
# from IPython.display import display, clear_output

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False);
import time
import cv2
import os
import numpy as np
import json

# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox).cpu()
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

# def plot_results(pil_img, prob, boxes):
#     plt.figure(figsize=(16,10))
#     plt.imshow(pil_img)
#     ax = plt.gca()
#     colors = COLORS * 100
#     for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
#         ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
#                                    fill=False, color=c, linewidth=3))
#         cl = p.argmax()
#         text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
#         ax.text(xmin, ymin, text, fontsize=15,
#                 bbox=dict(facecolor='yellow', alpha=0.5))
    # plt.axis('off')
    # plt.show()

model = torch.hub.load('facebookresearch/detr', 'detr_resnet101_dc5' , pretrained=True)
model.eval()
model = model.cuda()

path = 'Images/EyeQ_FN/'

images_dict = {}

def np_to_pil(np_image):
    pil_image = Image.fromarray(np.uint8(np_image))
    return pil_image

def predict(im):
    im = np_to_pil(im)
    cv_image = cv2.cvtColor(np.array(im), cv2.COLOR_BGR2RGB)
    img = transform(im).unsqueeze(0)
    img = img.cuda()
    # propagate through the model
    with torch.no_grad():
        outputs = model(img)
    # keep only predictions with 0.9+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.9

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    colors = COLORS * 100
    img = cv_image
    boxes = []
    scores = []
    labels = []
    for p, (xmin, ymin, xmax, ymax), c in zip(probas[keep], bboxes_scaled.tolist(), colors):
            cl = p.argmax()
            if CLASSES[cl] == 'car' or CLASSES[cl] == 'bus' or CLASSES[cl] == 'truck':
                text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
                score = float(f'{p[cl]:0.2f}')*100
                x1 = int(xmin)
                y1 = int(ymin)
                x2 = int(xmax)
                y2 = int(ymax)
                boxes.append([y1,x1,y2,x2])
                labels.append(CLASSES[cl])
                scores.append(int(score))
    return boxes, labels, scores, img

