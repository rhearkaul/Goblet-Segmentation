from src.sam.sam import SAModel
from src.sam.sam import SAModelType

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

import segment_anything
from segment_anything import (
    sam_model_registry,
    SamPredictor,
    build_sam_vit_b,
    build_sam_vit_h,
    build_sam_vit_l,
    build_sam
)

sam2 = build_sam_vit_h(path_to_weights)
sam2.to(device='cuda')
predictor = SamPredictor(sam2)

# Set the image
predictor.set_image(image)


input_pts_tensor = torch.tensor(input_pts, device=predictor.device).unsqueeze(1)


transformed_pts = predictor.transform.apply_coords_torch(
    input_pts_tensor, image.shape[:2]
)


sam = SAModel()
sam.load_weights(model_type=SAModelType.SAM_VIT_H, path_to_weights=path_to_weights) # use sam.load_weights() if you have downloaded weights
sam.set_image(image)
masks, iou_scores = sam.predict(points=transformed_pts) # labels & bbox is optional

print("hold")