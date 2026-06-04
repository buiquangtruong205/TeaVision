import io

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from src.deploy.api import app
from src.models.inference import TeaVisionPredictor


def make_image_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color=(40, 130, 60)).save(buffer, format="JPEG")
    return buffer.getvalue()


def make_leaf_image_bytes() -> bytes:
    buffer = io.BytesIO()
    image = Image.new("RGB", (200, 140), color=(235, 235, 245))
    draw = ImageDraw.Draw(image)
    draw.ellipse((35, 35, 165, 105), fill=(35, 105, 45))
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_predictor_loads_exported_artifact():
    predictor = TeaVisionPredictor("models/cnn_zscore/model_artifact.json", device="cpu")
    prediction = predictor.predict_bytes(make_image_bytes())

    assert prediction.class_name in predictor.class_names
    assert 0.0 <= prediction.confidence <= 1.0
    assert set(prediction.probabilities) == set(predictor.class_names)


def test_predictor_localizes_leaf_for_prediction_metadata():
    predictor = TeaVisionPredictor("models/cnn_zscore/model_artifact.json", device="cpu")
    prediction = predictor.predict_bytes(make_leaf_image_bytes())

    left, top, right, bottom = prediction.leaf_bbox
    assert prediction.leaf_detected is True
    assert 0 < left < right < 200
    assert 0 < top < bottom < 140


def test_fastapi_health_model_info_and_predict():
    with TestClient(app) as client:
        health_response = client.get("/health")
        model_response = client.get("/model-info")
        prediction_response = client.post(
            "/predict",
            files={"file": ("leaf.jpg", make_image_bytes(), "image/jpeg")},
        )

    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok", "model_loaded": True}
    assert model_response.status_code == 200
    assert model_response.json()["normalizer"] == "zscore"
    assert prediction_response.status_code == 200
    assert prediction_response.json()["class_name"] in model_response.json()["class_names"]
    assert "leaf_bbox" in prediction_response.json()


def test_fastapi_returns_annotated_image():
    with TestClient(app) as client:
        response = client.post(
            "/predict/annotated",
            files={"file": ("leaf.jpg", make_leaf_image_bytes(), "image/jpeg")},
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["x-teavision-class"]
    assert len(response.content) > 0


def test_fastapi_rejects_non_image_upload():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={"file": ("notes.txt", b"not an image", "text/plain")},
        )

    assert response.status_code == 415


def test_fastapi_serves_demo_homepage():
    with TestClient(app) as client:
        homepage_response = client.get("/")
        script_response = client.get("/system/script.js")

    assert homepage_response.status_code == 200
    assert "TeaVision" in homepage_response.text
    assert script_response.status_code == 200
    assert "/predict/annotated" in script_response.text
