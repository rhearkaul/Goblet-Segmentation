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

def load_data_from_csv(csv_file_path, device='cpu'):
    with open(csv_file_path, 'r', newline='') as csvfile:
        with open(csv_file_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            boxes = []
            for row in reader:
                if row['Type'] == 'Box':
                    # Convert string to int and parse coordinates
                    x1, y1 = int(row['X1']), int(row['Y1'])  # Bottom-left
                    x2, y2 = int(row['X2']), int(row['Y2'])  # Bottom-right
                    x3, y3 = int(row['X3']), int(row['Y3'])  # Top-right
                    x4, y4 = int(row['X4']), int(row['Y4'])  # Top-left

                    # Calculate width and height
                    width = x2 - x1
                    height = y1 - y3

                    # The top-left corner is (x4, y4)
                    top_left_x = x4
                    top_left_y = y4

                    width = abs(width)
                    height = abs(height)

                    boxes.append((top_left_x, top_left_y, width, height))

    original_box = boxes[1]
    boxes_tensor = torch.tensor(boxes, device=device)

    image_path = csv_file_path.replace('.csv', '.jpg')
    image = cv2.imread(image_path)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        print(f"Warning: Image at {image_path} could not be loaded.")
        image = None

    return boxes_tensor, image, original_box

csv_file_path = 'C:/Users/steve/PycharmProjects/Goblet-Segmentation/src/csv/1.csv'
device = 'cpu'
boxes1_tensor, image1,original_box = load_data_from_csv(csv_file_path, device=device)

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

image = image1

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

input_boxes = boxes1_tensor

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
for box in input_boxes:
    show_box(box.cpu().numpy(), plt.gca())
plt.axis('off')
plt.show()
