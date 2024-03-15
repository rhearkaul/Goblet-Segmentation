# These tests are to check if SAM is correctly built
import pytest
import requests_mock

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
    """Tests failed http requests"""
    url_base = "https://dl.fbaipublicfiles.com/segment_anything"
    error_code = 404

    with pytest.raises(RuntimeError):
        with requests_mock.Mocker() as mock:
            mock.get(f"{url_base}/{name}", status_code=error_code)
            model._download_weights(model_type)
