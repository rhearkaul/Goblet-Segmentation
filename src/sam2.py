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

def save_masks_and_ious(masks, iou_scores, image, output_dir):
    # Get the existing mask files in the output directory
    existing_mask_files = [f for f in os.listdir(output_dir) if f.startswith("mask_")]

    # Find the maximum mask index from the existing files
    if existing_mask_files:
        max_index = max([int(f.split("_")[1].split(".")[0]) for f in existing_mask_files])
        next_index = max_index + 1
    else:
        next_index = 0

    # Save masks as images with unique names
    for i, mask in enumerate(masks):
        mask_file = f"mask_{next_index + i}.png"
        plt.imsave(os.path.join(output_dir, mask_file), mask.cpu().numpy(), cmap='gray')

    # Save IOUs as a CSV
    with open(os.path.join(output_dir, 'ious.csv'), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if next_index == 0:
            writer.writerow(['Mask Index', 'IOU Score'])
        for i, iou in enumerate(iou_scores):
            writer.writerow([next_index + i, iou.item()])

    # Save the predicted image
    plt.imsave(os.path.join(output_dir, "predicted_image.png"), image)

def sam_main(path_to_weights, annotations_filename='annotations.npz', image_folder=""):
    points, boxes, image = load_annotations(filename=annotations_filename)
    input_pts = points.tolist()
    boxes = boxes.tolist()

    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        print(f"Warning: Image could not be loaded.")
        return

    sam = SAModel()
    sam.load_weights(model_type=SAModelType.SAM_VIT_H, path_to_weights=path_to_weights)
    sam.set_image(image)

    if input_pts:
        input_pts_tensor = torch.tensor(input_pts, device=sam.model.device).unsqueeze(1)
        transformed_pts = sam.model.transform.apply_coords_torch(input_pts_tensor, image.shape[:2])
        input_lbls = [1 for _ in range(len(input_pts))]
        input_lbls_tensor = torch.tensor(input_lbls, device=sam.model.device).unsqueeze(1)
        masks, iou_scores = sam.predict(points=transformed_pts, labels=input_lbls_tensor)
    else:
        masks, iou_scores = [], []

    if boxes:
        input_boxes = torch.tensor(boxes, device=sam.model.device).unsqueeze(1)
        transformed_boxes = sam.model.transform.apply_boxes_torch(input_boxes, image.shape[:2])
        masks2, iou_scores2 = sam.predict(bboxes=transformed_boxes)
    else:
        masks2, iou_scores2 = [], []

    final_masks = masks + masks2
    final_iou_scores = iou_scores + iou_scores2

    if final_masks:
        output_dir = image_folder
        save_masks_and_ious(final_masks, final_iou_scores, image, output_dir)

        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        for mask in final_masks:
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        plt.axis('off')
        plt.show()

        return output_dir
    else:
        print("No points or boxes provided, or model did not predict any masks.")



if __name__ == '__main__':
    path_to_weights = "sam_vit_h_4b8939.pth"
    sam_main(path_to_weights)
