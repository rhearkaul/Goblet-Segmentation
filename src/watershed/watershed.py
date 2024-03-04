import numpy as np
from scipy.linalg import inv
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from skimage import morphology
from skimage.exposure import exposure
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
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

SIZE_THRESHOLDS = {0: (50, 1e3), 1: (50, 1e4), 2: (100, 2500)}


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
    adjusted_image = exposure.rescale_intensity(image, in_range=intensity_thresh)

    bin_thresh = threshold_otsu(adjusted_image)
    filled_img = binary_fill_holes(adjusted_image > bin_thresh)

    lbled_img = label(filled_img)
    props = regionprops(lbled_img)

    filtered_props = [
        prop for prop in props if size_thresh[0] <= prop.area <= size_thresh[1]
    ]

    bin_mask = np.zeros_like(filled_img, dtype=bool)

    for prop in filtered_props:
        y_min, x_min, y_max, x_max = prop.bbox
        bin_mask[y_min:y_max, x_min:x_max] = True

    filtered_img = filled_img.copy()
    filtered_img[~bin_mask] = 0

    return filtered_img


def _filter_img(bin_img, props, min_solidity, max_aspect_ratio):
    """Filteres out `regionprops` based on solidity."""
    # Can be restructured if more filters are required.

    for prop in props:
        if prop.minor_axis_length <= 1:
            continue  # Skip this region

        aspect_ratio = prop.major_axis_length / prop.minor_axis_length
        if aspect_ratio < max_aspect_ratio:
            # Set the corresponding pixel values to False in the binary image
            coords = prop.coords
            bin_img[coords[:, 0], coords[:, 1]] = False

    labels = label(bin_img)
    props = regionprops(labels)
    filtered_img = np.zeros_like(bin_img)

    for prop in props:
        if prop.solidity >= min_solidity:
            # Include the prop in the filtered binary image
            filtered_img[labels == prop.label] = 1

    return filtered_img


def _watershed(bin_img, footprint_kernel=(3, 3)):
    """Watershed segmentation on a binary image."""
    # Perform distance transform
    distances = distance_transform_edt(bin_img)

    coords = peak_local_max(
        distances, footprint=np.ones(footprint_kernel), labels=bin_img
    )

    mask = np.zeros(distances.shape, dtype=bool)
    mask[tuple(coords.T)] = True

    # Perform watershed
    markers = label(mask)
    segmented_img = watershed(-distances, markers, mask=bin_img)

    return segmented_img, distances


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
    equalization_bins,
    intensity_thresh,
    size_thresh,
    max_aspect_ratio,
    min_solidity,
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
    normalized_img = np.asfarray(image, np.float64) / 255.0  # Normalize
    deconv_img = _deconvolve(normalized_img, stain_vector)
    hist_equalized_img = _hist_equalization(
        deconv_img[:, :, 1],
        equalization_bins,
    )
    bin_img = _threshold_and_binarize(hist_equalized_img, intensity_thresh, size_thresh)

    # Filter out noise (certain shapes)
    labels = label(bin_img)
    props = regionprops(labels)
    filtered_img = _filter_img(bin_img, props, min_solidity, max_aspect_ratio)

    # Generate promps from watershed
    segmented_img, distances = _watershed(filtered_img, distance_thresh)
    labels = label(segmented_img)
    props = regionprops(labels, coordinates="xy")
    centroid_coords = np.array([prop.centroid for prop in props]).astype(int)

    return centroid_coords, distances, deconv_img
