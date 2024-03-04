import cv2
import numpy as np
import src.watershed.watershed as watershed

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def save_annotations(points, image_path, boxes=[]):
    points_array = np.array(points)
    boxes_array = np.array(boxes)
    np.savez('watershed.npz', points=points_array, boxes=boxes_array, image_path=image_path)
    print("Annotations and image path saved.")

def watershed_image(image_path):
    # Watershed parameters
    stain_vector = watershed.STAIN_VECTORS
    intensity_thresh = watershed.INTENSITY_THRESHOLDS
    size_thresh = watershed.SIZE_THRESHOLDS

    kwargs = {
        "stain_vector": stain_vector.get(0),
        "equalization_bins": 1000,
        "intensity_thresh": intensity_thresh.get(1),
        "size_thresh": size_thresh.get(2),
        "max_aspect_ratio": 1.50,
        "min_solidity": 0.75,
    }

    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    coords, _, _, _ = watershed.generate_centroid(image=rgb_image, **kwargs)

    save_annotations(coords, image_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        process_image(image_path)
    else:
        print("Please provide an image path.")

