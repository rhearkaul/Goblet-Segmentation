import numpy as np
from scipy.linalg import inv
from scipy.ndimage import distance_transform_edt
from skimage.measure import label, regionprops
from skimage.segmentation import watershed


def _get_normalized_stain_vector(stain_type):
    "Returns the normalized stain vector"

    STAIN_CHANNELS = {
        0: np.array(
            [
                [0.65, 0.70, 0.29],
                [0.07, 0.99, 0.11],
                [-0.2101, -0.0512, 0.5945],
            ]
        ),
        1: np.array(
            [
                [0.6500286, 0.704031, 0.2860126],
                [0.07, 0.99, 0.11],
                [0.7110272, 0.42318153, 0.5615672],
            ]
        ),
    }

    vec = STAIN_CHANNELS.get(stain_type)
    return vec / vec.sum(axis=1)[:, np.newaxis]


def _deconvolve(image: np.ndarray, stain_type):
    """Performs deconvolution of the image."""
    he_to_rgb = _get_normalized_stain_vector(stain_type)
    rgb_to_heos = inv(he_to_rgb)
    image_col = image.reshape((-1, 3)).T
    deconv_img = rgb_to_heos @ image_col
    deconv_img = deconv_img.T.reshape(image.shape)
    return deconv_img


def _hist_equalization(image):
    """Performs histogram equalization on the image."""
    return image


def _threshold_and_binarize(image, thresh):
    """Performs thresholding on the image."""
    return image


def _filter_props(props, solidity_thresh):
    """Filteres out `regionprops` based on solidity."""
    # Can be restructured if more filters are required.
    return [prop.solidity for prop in props if prop.solidity >= solidity_thresh]


def _watershed(bin_img, distance_threshold):
    """Watershed segmentation on a binary image."""
    # Perform distance transform
    distances = -distance_transform_edt(~bin_img)
    mask = distances > distance_threshold
    masked_distances = -distance_transform_edt(mask)

    # Perform watershed
    markers = label(mask)
    labels = watershed(masked_distances, markers, mask=bin_img)

    segmented_img = bin_img.copy()
    segmented_img[labels == 0] = 0

    return labels, segmented_img


def generate_centroid(
    image,
    stain_type,
    bin_thresh,
    solidity_thresh,
    distance_thresh,
):
    """Generates the centroid locations given an image.

    Parameters
    ----------
        todo

    Returns
    -------
    list[tuple]:
        containing the centroid coordinates in (x,y).

    ndarray:
        binary image of the segmented image.
    """
    # Preprocess the image
    deconv_img = _deconvolve(image, stain_type)
    hist_equalized_img = _hist_equalization(deconv_img)
    bin_img = _threshold_and_binarize(hist_equalized_img, bin_thresh)

    # Filter out noise (certain shapes)
    labels = label(bin_img)
    props = regionprops(labels)
    filtered_props = _filter_props(labels, props, solidity_thresh)

    # Generated the filtered binary image
    bin_img = np.zeros_like(bin_img)

    for prop in filtered_props:
        bin_img += labels == prop.label

    # Generate promps from watershed
    labels, img_mask = _watershed(bin_img, distance_thresh)
    props = regionprops(labels, coordinates="xy")
    centroid_coords = np.array([prop.centroid for prop in props]).astype(int)

    return centroid_coords, img_mask
