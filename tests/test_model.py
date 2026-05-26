from src.models.model import build_model


def test_build_model_resnet50_head():
    model = build_model(num_classes=2, model_name="resnet50", pretrained=False)
    assert model.fc.out_features == 2
