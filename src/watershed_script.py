import cv2
import numpy as np

import src.watershed.watershed as watershed
import matplotlib.pyplot as plt

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)

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


image = cv2.imread("C:/Users/steve/Downloads/data/Data Aug (011924)/raw/test222_sliced.jpg")
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print("test")
coords, _ ,_,_= watershed.generate_centroid(image=rgb_image, **kwargs)
print(coords)


input_lbls = [1 for _ in range(len(coords))]

plt.figure(figsize=(10, 10))
plt.imshow(image)
show_points(coords, input_lbls, plt.gca())
plt.axis('off')
plt.show()