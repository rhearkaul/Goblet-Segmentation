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


def get_prop(binary_mask):

    labels = label(binary_mask)
    props = regionprops_table(labels, properties=PROPERTIES)
    df = pd.DataFrame(props)

    # Return the largest mask in case of artefacts
    i_max = df["area"].idxmax()
    return df.loc(i_max)


## Assuming we have .tiff we can convert the resolution to length

# from PIL import Image
# img = Image.open("image.tiff")
# res_x, res_y = img.info.get("resolution", (None, None))

# if not res_x or not res_y:
## Can be handled by default parameters as well...but assume
# logging.error("Resolution not found, skipping image...")

## Examples:
# binary_masks = ...
# a_random_prop = get_prop(binary_masks[5])
#
# area = a_random_prop["area"] * res_x * res_y
#
# orientation = a_random_prop["orientation"]
# axis_major_length = np.linalg.norm([
#   a_random_prop["axis_major_length"] * np.cos(orientation) * res_x,
#   a_random_prop["axis_major_length"] * np.sin(orientation) * res_y,
# ])
#
