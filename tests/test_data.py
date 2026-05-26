import pandas as pd
from PIL import Image

from src.data.dataset import TeaDiseaseDataset
from src.data.splitter import create_class_mapping, scan_dataset


def test_dataset_length():
    dataset = TeaDiseaseDataset(samples=[])
    assert len(dataset) == 0


def test_scan_dataset_and_csv_dataset(tmp_path):
    dataset_dir = tmp_path / "dataset"
    healthy_dir = dataset_dir / "healthy"
    blight_dir = dataset_dir / "blight"
    healthy_dir.mkdir(parents=True)
    blight_dir.mkdir(parents=True)

    Image.new("RGB", (16, 16), color="green").save(healthy_dir / "a.jpg")
    Image.new("RGB", (16, 16), color="brown").save(blight_dir / "b.jpg")

    mapping = create_class_mapping(dataset_dir, tmp_path / "class_mapping.json")
    metadata = scan_dataset(dataset_dir, include_hashes=True, relative_to=tmp_path)
    csv_path = tmp_path / "samples.csv"
    metadata.to_csv(csv_path, index=False)

    dataset = TeaDiseaseDataset(csv_file=csv_path, root_dir=tmp_path)

    assert mapping == {"0": "blight", "1": "healthy"}
    assert len(metadata) == 2
    assert set(metadata["class_name"]) == {"healthy", "blight"}
    assert len(dataset) == 2
    image, label = dataset[0]
    assert image.mode == "RGB"
    assert isinstance(label, int)
