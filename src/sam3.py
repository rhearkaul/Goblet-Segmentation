import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from itertools import product
from skimage import io, color
from skimage.measure import label, regionprops, find_contours
from sklearn.metrics import jaccard_score, f1_score, accuracy_score, precision_score, recall_score
import segment_anything

from segment_anything import (
    sam_model_registry,
    SamPredictor,
    build_sam_vit_b,
    build_sam_vit_h,
    build_sam_vit_l,
    build_sam
)

from segment_anything.utils.transforms import ResizeLongestSide

device = "cuda" if torch.cuda.is_available() else "cpu"

### Copied from Meta ###
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

### OURS ###
def object_metrics(ground, mask):
  return [
      f1_score(ground, mask, average="micro"),
      jaccard_score(ground, mask, average="micro"),
  ]

def process(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = np.sum([cv2.contourArea(contour) for contour in contours])
    counts = len(contours)

    return counts, area, np.array(img == 255, dtype = np.uint8)

def metrics(mask, ground):
    accuracy = accuracy_score(ground.ravel(), mask.ravel())
    precision = precision_score(ground, mask, average="micro")
    dice = f1_score(ground, mask, average="micro")
    iou = jaccard_score(ground, mask, average="micro")
    recall = recall_score(ground, mask, average="micro")

    return accuracy, precision, recall, dice, iou

def get_mask(binaries):
  final_mask = binaries[0]

  for image in binaries[1:]:
    final_mask = np.bitwise_or(final_mask, image)

  return final_mask


def mask_to_border(mask):
    h, w = mask.shape
    border = np.zeros((h, w))

    contours = find_contours(mask, 128)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 255

    return border

def mask_to_bbox(mask):
    bboxes = []

    mask_im = mask_to_border(mask)
    lbl = label(mask_im)
    props = regionprops(lbl)
    for prop in props:
        x1 = prop.bbox[1]
        y1 = prop.bbox[0]

        x2 = prop.bbox[3]
        y2 = prop.bbox[2]

        # width = x2-x1
        # height = y2-y1
        width = x2
        height = y2

        bboxes.append([x1, x2, y1, y2])

    return bboxes

def loadAnnotations(filename='annotations.npz'):
    data = np.load(filename)
    input_points = data['points']
    input_boxes = data['boxes']
    print("Loaded points:", input_points)
    print("Loaded boxes:", input_boxes)
    return input_points, input_boxes




points, original_box = loadAnnotations(filename='annotations.npz')
input_pts = points.tolist()
box1_tensor = torch.tensor(original_box)
points_tensor = torch.tensor(input_pts)
input_pts_tensor = torch.tensor(input_pts, device='cuda').unsqueeze(1)


image_path = "C:/Users/steve/Downloads/data/Data Aug (011924)/raw/test222.jpg"
path_to_weights = "C:/Users/steve/PycharmProjects/Goblet-Segmentation/src/sam_vit_h_4b8939.pth"
image = cv2.imread(image_path)
if image is not None:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
else:
    print(f"Warning: Image at {image_path} could not be loaded.")
    image = None

# Build the SAM model, ensure that we use the correct builder (h vs b)
weights = "C:/Users/steve/PycharmProjects/Goblet-Segmentation/src/sam_vit_h_4b8939.pth"

sam = build_sam_vit_h(weights)
sam.to(device=device)
predictor = SamPredictor(sam)

# Set the image
predictor.set_image(image)


input_pts_tensor = torch.tensor(input_pts, device=predictor.device).unsqueeze(1)

transformed_pts = predictor.transform.apply_coords_torch(
    input_pts_tensor, image.shape[:2]
)

masks, scores, logits = predictor.predict_torch(
    point_coords=transformed_pts,
    point_labels=None,
    multimask_output=True,
)

print("hold")