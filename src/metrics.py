import numpy as np
import pandas as pd
from skimage.measure import label, regionprops_table

PROPERTIES = [
    "area",
    "preimeter",
    "axis_major_length",
    "axis_minor_length",
    "eccentricity",
    "orientation",
]


def get_prop(binary_mask: list, properties: list = PROPERTIES):
    """
    Gets the properties of the binary masks.

    Parameters:
    -----------
    binary_mask: list
        a collection of 2D images of binary masks.

    properties: list
        a collection of properties that is made available by
        `skimage.measure.regionprops`.

    Returns:
    --------
        pd.DataFrame of values representing each image as a row
        and designated properties as columns.

    Assuming we have .tiff we can convert the resolution to length

    Example usage:
    >>> from PIL import Image
    >>> img = Image.open("image.tiff")
    >>> res_x, res_y = img.info.get("resolution", (None, None))

    >>> if not res_x or not res_y:
        # Can be handled by default parameters as well...but assume
    >>>     logging.error("Resolution not found, skipping image...")

    >>> binary_masks = ...
    >>> a_random_prop = get_prop(binary_masks[5])

    >>> area = a_random_prop["area"] * res_x * res_y

    >>> orientation = a_random_prop["orientation"]
    >>> axis_major_length = np.linalg.norm([
        a_random_prop["axis_major_length"] * np.cos(orientation) * res_x,
        a_random_prop["axis_major_length"] * np.sin(orientation) * res_y,
    ])
    """
    labels = label(binary_mask)
    props = regionprops_table(labels, properties=properties)
    df = pd.DataFrame(props)

    # Return the largest mask in case of artefacts
    i_max = df["area"].idxmax()
    return df.loc(i_max)


def detect_outliers(data: pd.DataFrame | list, alpha: float = 3):
    """Uses Z-test as a method for outlier detection`.

    Developer Note: other implementations could work. Suggestion for
    future implementation can be found in `sklearn`.

    Parameters:
    -----------
    data: pd.DataFrame | list
        a one-dimensional dataset that contains some information about the data.

    alpha: float
        sigma threshold required to pass detection.

    Returns:
    --------
    tuple:
        of pd.DataFrames containing inliers and outliers, respectively.
    """
    df = pd.DataFrame(data)
    z_scores = (df - df.mean()) / df.std()
    filter = abs(z_scores) < alpha

    return df[filter], df[~filter]
