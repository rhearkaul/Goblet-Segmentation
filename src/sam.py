import csv

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())
import sys

def loadAnnotations(filename='annotations.npz'):
    data = np.load(filename, allow_pickle=True)
    input_points = data['points']
    input_boxes = data['boxes']
    image_path = str(data['image_path'])  # Convert to string explicitly

    print("Loaded points:", input_points)
    print("Loaded boxes:", input_boxes)
    print("Image path:", image_path)

    image = cv2.imread(image_path)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        print(f"Warning: Image at {image_path} could not be loaded.")
        image = None

    return input_points, input_boxes, image



points, original_box,image = loadAnnotations(filename='annotations.npz')
original_box = original_box[0]
box1_tensor = torch.tensor(original_box)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

plt.figure(figsize=(10, 10))
plt.imshow(image)
show_box(original_box, plt.gca())
plt.axis('off')
plt.show()


import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)


predictor = SamPredictor(sam)
predictor.set_image(image)

print(original_box)

input_boxes = box1_tensor

transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
masks, _, _ = predictor.predict_torch(
    point_coords=None,
    point_labels=None,
    boxes=transformed_boxes,
    multimask_output=False,
)

plt.figure(figsize=(10, 10))
plt.imshow(image)
for mask in masks:
    show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
plt.axis('off')
plt.show()
