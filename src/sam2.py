from src.sam.sam import SAModel, SAModelType
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
from datetime import datetime


def load_annotations(filename='annotations.npz'):
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

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def save_masks_and_ious(masks, iou_scores):
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    output_dir = f'output_masks/{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    # Save masks as images
    for i, mask in enumerate(masks):
        plt.imsave(f'{output_dir}/mask_{i}.png', mask.cpu().numpy(), cmap='gray')

    # Save IOUs as a CSV
    with open(f'{output_dir}/ious.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Mask Index', 'IOU Score'])
        for i, iou in enumerate(iou_scores):
            writer.writerow([i, iou.item()])

def main(path_to_weights, annotations_filename='annotations.npz'):
    points, original_box,image = load_annotations(filename=annotations_filename)
    input_pts = points.tolist()

    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        print(f"Warning: Image could not be loaded.")
        return

    sam = SAModel()
    sam.load_weights(model_type=SAModelType.SAM_VIT_H, path_to_weights=path_to_weights)

    input_pts_tensor = torch.tensor(input_pts, device=sam.model.device).unsqueeze(1)
    transformed_pts = sam.model.transform.apply_coords_torch(input_pts_tensor, image.shape[:2])
    input_lbls = [1 for _ in range(len(input_pts))]
    input_lbls_tensor = torch.tensor(input_lbls, device=sam.model.device).unsqueeze(1)

    sam.set_image(image)
    masks, iou_scores = sam.predict(points=transformed_pts, labels=input_lbls_tensor)

    save_masks_and_ious(masks, iou_scores)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    path_to_weights = "sam_vit_h_4b8939.pth"
    main(path_to_weights)
