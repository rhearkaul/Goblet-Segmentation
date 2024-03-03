import cv2
import numpy as np
from scipy.linalg import inv
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from skimage import morphology
from skimage.exposure import exposure
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops, regionprops_table
from skimage.segmentation import watershed

STAIN_VECTORS = {
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


INTENSITY_THRESHOLDS = {
    0: (0.4, 0.7),
    1: (0.6, 0.7),
}

SIZE_THRESHOLDS = {
    0: (50, 1e3),
    1: (50, 1e4),
}


def _get_normalized_stain_vector(stain_vector):
    "Returns the normalized stain vector"
    return stain_vector / stain_vector.sum(axis=1)[:, np.newaxis]


def _deconvolve(image: np.ndarray, stain_vector):
    """Performs deconvolution of the image."""
    he_to_rgb = _get_normalized_stain_vector(stain_vector)
    rgb_to_heos = inv(he_to_rgb)
    image_col = image.reshape((-1, 3)).T
    deconv_img = rgb_to_heos @ image_col
    deconv_img = deconv_img.T.reshape(image.shape)
    return deconv_img


def _hist_equalization(image, nbins):
    """Performs histogram equalization on the image."""
    return exposure.equalize_hist(image, nbins=nbins)


def _threshold_and_binarize(
    image, intensity_thresh=(0.75, 0.85), size_thresh=(100, 2500)
):
    """Performs thresholding on the image."""
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    adjusted_image = exposure.rescale_intensity(gray_img, in_range=intensity_thresh)

    bin_thresh = threshold_otsu(adjusted_image)
    bin_img = binary_fill_holes(adjusted_image > bin_thresh)

    filtered_img = morphology.remove_small_objects(
        bin_img, min_size=size_thresh[0], max_size=size_thresh[1]
    )

    return filtered_img


def _filter_img(bin_img, props_table, min_solidity, max_aspect_ratio):
    """Filteres out `regionprops` based on solidity."""
    # Can be restructured if more filters are required.

    props_table["LenWdRatio"] = (
        props_table["major_axis_length"] / props_table["minor_axis_length"]
    )
    props_table["isGreater"] = props_table["LenWdRatio"] < max_aspect_ratio
    remove_idxs = np.where(~props_table["isGreater"])[0]

    for idx in remove_idxs:
        slice_idx = props_table["subarray_idx"][idx]
        bin_img[slice_idx[0], slice_idx[1]] = False

    labels = label(bin_img)
    props = regionprops(labels)
    filtered_props = [prop.solidity for prop in props if prop.solidity >= min_solidity]

    return filtered_props


def _watershed(bin_img, dist_thresh=2):
    """Watershed segmentation on a binary image."""
    # Perform distance transform
    distances = -distance_transform_edt(~bin_img)
    mask = morphology.h_minima(distances, dist_thresh)
    masked_distances = np.minimum(distances, mask)

    # Perform watershed
    markers = label(mask)
    labels = watershed(masked_distances, markers, mask=bin_img)

    segmented_img = bin_img.copy()
    segmented_img[labels == 0] = 0

    return labels, segmented_img


REGION_PROP_KEYS = (
    "centroid",
    "bbox",
    "major_axis_length",
    "minor_axis_length",
    "solidity",
    "subarray_idx",
)


def generate_centroid(
    image,
    stain_vector,
    solidity_thresh,
    distance_thresh,
    equalization_bins,
    intensity_thresh,
    size_thresh,
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
    image = np.asfarray(image, np.float64)
    deconv_img = _deconvolve(image, stain_vector)
    hist_equalized_img = _hist_equalization(
        deconv_img[:, :, 1],
        equalization_bins,
    )
    bin_img = _threshold_and_binarize(hist_equalized_img, intensity_thresh, size_thresh)

    # Filter out noise (certain shapes)
    labels = label(bin_img)
    props = regionprops_table(
        labels,
    )
    filtered_props = _filter_img(labels, props, solidity_thresh)

    # Generated the filtered binary image
    bin_img = np.zeros_like(bin_img)

    for prop in filtered_props:
        bin_img[label == prop.label] = 1

    # Generate promps from watershed
    labels, img_mask = _watershed(bin_img, distance_thresh)
    props = regionprops(labels, coordinates="xy")
    centroid_coords = np.array([prop.centroid for prop in props]).astype(int)

    return centroid_coords, img_mask, deconv_img
