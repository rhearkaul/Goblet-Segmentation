"""Pytest package to ensure that SAM can be instantiated correctly.

Author: Alvin Hendricks
"""

import cv2
import numpy as np
import pytest
import requests_mock
import torch

from src.sam.sam import SAModel, SAModelType


@pytest.fixture
def model():
    model = SAModel()
    return model


args = [(k, v) for k, v in SAModel.CHECKPOINTS.items()]
bad_weight_paths = ["noneExistantFile", "requirements.txt"]


@pytest.mark.parametrize("model_type, path", args)
def test_load_base_weights(model: SAModel, model_type: SAModelType, path: str):
    """Tests the loading for weights of the model, since
    we don't have base weights, it should make a request and download
    established weights. Passes if no issues arises...
    """
    model._load_base_weights(model_type)


@pytest.mark.parametrize("path", bad_weight_paths)
def test_load_bad_paths(model: SAModel, path: str):
    """Tests the loading if path is wrong or there is a bad weight file.
    Code should correctly handle the generation and still load the default
    SAM model. No errors should.
    """
    model.load_weights(SAModelType.SAM_VIT_B, path)


@pytest.mark.parametrize("model_type, name", args)
def test_fail_download(model: SAModel, model_type: SAModelType, name: str):
    """Tests failed http requests."""
    url_base = "https://dl.fbaipublicfiles.com/segment_anything"
    error_code = 404

    with pytest.raises(RuntimeError):
        with requests_mock.Mocker() as mock:
            mock.get(f"{url_base}/{name}", status_code=error_code)
            model._download_weights(model_type)


@pytest.mark.parametrize("model_type, name", args)
def test_predict_with_base_weights(model: SAModel, model_type: SAModelType, name: str):
    """Tests the predict function with at least the base weights."""
    model.load_weights(model_type)

    image = cv2.imread("test/images/test_mask.png")
    model.set_image(image)

    # Get prediction points
    n_row, n_col, _ = image.shape
    midpoint = [[n_row // 2, n_col // 2]]

    input_pts_tensor = torch.tensor(midpoint, device=model.model.device).unsqueeze(1)
    transformed_pts = model.model.transform.apply_coords_torch(
        input_pts_tensor, image.shape[:2]
    )

    input_lbls = [1]
    transformed_labels = torch.tensor(input_lbls, device=model.model.device).unsqueeze(
        1
    )

    output = model.predict(transformed_pts, transformed_labels)
    output = np.asarray(output[0][0], dtype=np.uint8)

    image2 = cv2.imread("test/images/test_mask.png", cv2.IMREAD_GRAYSCALE) / 255
    assert np.array_equal(image2, output)
