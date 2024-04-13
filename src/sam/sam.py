"""Package that generates the SAM models and interface tools for 
prompted segmentation. 

Adapted use from: https://github.com/facebookresearch/segment-anything 

Author: Alvin Hendricks
"""


import logging
import os
from enum import Enum
from pathlib import Path

import numpy as np
import requests
import torch
from segment_anything import SamPredictor, sam_model_registry


class SAModelType(Enum):
    SAM_VIT_L = "vit_l"
    SAM_VIT_H = "vit_h"
    SAM_VIT_B = "vit_b"


class SAModel:
    CHECKPOINTS = {
        SAModelType.SAM_VIT_L: "sam_vit_l_0b3195.pth",
        SAModelType.SAM_VIT_H: "sam_vit_h_4b8939.pth",
        SAModelType.SAM_VIT_B: "sam_vit_b_01ec64.pth",
    }

    def __init__(self) -> None:
        self.model = None
        self._cuda_available = torch.cuda.is_available()

    def _download_weights(self, model_type: SAModelType):
        """Downloads the weights from the weight database linked in SAM's GitHub page.

        Parameters:
        -----------
        model_type: SAModelType
            specifies the model to be used.

        Raises:
        -------
        RuntimeError
            If weights were not able to be obtained.
        """

        url_base = "https://dl.fbaipublicfiles.com/segment_anything"

        selected_model = self.CHECKPOINTS[model_type]

        req = requests.get(f"{url_base}/{selected_model}")

        if req.status_code == 200:
            logging.info(f"Succesfully downloaded {selected_model}, proceeding...")
            with open(selected_model, "wb") as f:
                f.write(req.content)
            return selected_model
        else:
            raise RuntimeError("Unable to obtain weights.")

    def _load_base_weights(self, model_type: SAModelType = SAModelType.SAM_VIT_L):
        try:
            selected_model = self.CHECKPOINTS[model_type]

            if selected_model in os.listdir():
                logging.info(f"'{model_type.value}' weights exists, proceeding...")
                path_to_weights = selected_model
            else:
                logging.info(f"Getting '{model_type.value}' weights...")
                path_to_weights = self._download_weights(model_type)

        except RuntimeError:
            path_to_weights = None
            logging.error("Unable to obtain weights, empty model will be loaded.")
        else:
            sam = sam_model_registry[model_type.value](path_to_weights)

        return sam

    def load_weights(
        self,
        model_type: SAModelType,
        path_to_weights: Path | str = None,
    ):
        """Loads the weights for the selected model. If there are issues
        loading the weights, the `vit_l` base model will be loaded instead.

        Parameters:
        -----------
        path_to_weights: Path or str
            Points to the location of the weights.

        model_type: SAModelType
            Specifies the model to be used.
        """
        try:
            if not path_to_weights:
                logging.info("Paths to weights not set, default will be loaded")
                sam = self._load_base_weights()
            elif not path_to_weights.endswith(".pth"):
                raise ValueError("Not a weight file.")
            else:
                sam = sam_model_registry[model_type.value](path_to_weights)
        except (RuntimeError, FileNotFoundError, ValueError) as error:
            logging.error(f"Something went wrong, loading base instead: {error}")
            sam = self._load_base_weights()
        else:
            logging.info("Weights loaded sucessfully!")
        finally:
            sam.to("cuda" if self._cuda_available else "cpu")
            self.model = SamPredictor(sam)
            logging.info("SAM model generated, proceeding with predictions")

    def set_image(self, image):
        """Sets the image to the model. Does nothing if model is not loaded.

        Parameters:
        -----------
        image: array-like
            containing pixel values of type uint8. Note, the image will be casted
            to the appropriate type if not already.
        """

        if self.model:
            image = np.asarray(image, dtype=np.uint8)
            self.model.set_image(image)

    def predict(self, points: list = None, labels: list = None, bboxes: list = None):
        """Perform prediction on the image.

        Parameters:
        ----------
        points: list
            pixel coordinates acting as prompts. This is not required if points
            is set to none and bboxes are included.

        labels: list
            labels corresponding to prompts.

        bboxes:
            bboxes that bound the object.

        Returns:
        -------
        tuple:
            containing the mask and the respective iou scores.
        """
        masks, scores, _ = self.model.predict_torch(
            point_coords=points,
            point_labels=labels,
            boxes=bboxes,
            multimask_output=True,
        )

        output_masks = []
        iou_scores = []

        # Get the best performing mask
        for i, output in enumerate(masks):
            output_masks.append(output[scores[i].argmax()])
            iou_scores.append(scores[i].max())

        return output_masks, iou_scores
