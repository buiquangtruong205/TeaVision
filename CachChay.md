Nhóm 1: KNN
python scripts/run_normalization_training.py ^
--image-size 128 ^
--max-total-samples 0 ^
--normalizers minmax zscore robust imagenet ^
--models knn

Chạy 4 cặp:

minmax   + knn
zscore   + knn
robust   + knn
imagenet + knn
Nhóm 2: SVM
python scripts/run_normalization_training.py ^
--image-size 128 ^
--max-total-samples 0 ^
--normalizers minmax zscore robust imagenet ^
--models svm

Chạy 4 cặp:

minmax   + svm
zscore   + svm
robust   + svm
imagenet + svm
Nhóm 3: CNN
python scripts/run_normalization_training.py ^
--image-size 128 ^
--max-total-samples 0 ^
--cnn-epochs 30 ^
--normalizers minmax zscore robust imagenet ^
--models cnn

Chạy 4 cặp:

minmax   + cnn
zscore   + cnn
robust   + cnn
imagenet + cnn
Nếu muốn nhanh hơn nữa

Với KNN và SVM không cần 128:

--image-size 64

Ví dụ:

python scripts/run_normalization_training.py ^
--image-size 64 ^
--max-total-samples 0 ^
--normalizers minmax zscore robust imagenet ^
--models svm

## Random Forest

Chạy Random Forest với cả 4 normalizer:

```bat
python scripts/run_normalization_training.py ^
--image-size 64 ^
--max-total-samples 0 ^
--normalizers minmax zscore robust imagenet ^
--models random_forest
```

Tham số thường dùng:

```bat
--rf-estimators 300 --rf-max-depth 20
```

Random Forest dùng `class_weight="balanced"`.

## XGBoost

Cài dependency nếu môi trường hiện tại chưa có XGBoost:

```bat
pip install -r requirements.txt
```

Chạy XGBoost với cả 4 normalizer:

```bat
python scripts/run_normalization_training.py ^
--image-size 64 ^
--max-total-samples 0 ^
--normalizers minmax zscore robust imagenet ^
--models xgboost
```

Tham số thường dùng:

```bat
--xgb-estimators 300 --xgb-max-depth 6 --xgb-learning-rate 0.1
```

XGBoost multiclass dùng `sample_weight` cân bằng được tính từ tập train.

## Chạy từng model và từng normalizer

Thay `<normalizer>` bằng `minmax`, `zscore`, `robust` hoặc `imagenet`.
Thay `<model>` bằng `svm`, `cnn`, `random_forest` hoặc `xgboost`.

```bat
python scripts/run_normalization_training.py ^
--image-size 64 ^
--max-total-samples 0 ^
--normalizers <normalizer> ^
--models <model>
```

Ví dụ chạy riêng Random Forest với z-score:

```bat
python scripts/run_normalization_training.py ^
--image-size 64 ^
--max-total-samples 0 ^
--normalizers zscore ^
--models random_forest
```

Ví dụ chạy riêng XGBoost với ImageNet normalization:

```bat
python scripts/run_normalization_training.py ^
--image-size 64 ^
--max-total-samples 0 ^
--normalizers imagenet ^
--models xgboost
```

CNN nên dùng `--image-size 128` và có thể thêm `--cnn-epochs 30`.

## Tinh chỉnh tiếp CNN + z-score

Kết quả hiện tại cho thấy `128x128`, batch size `64` tốt hơn các cấu hình đã thử.
Chạy file sau để tinh chỉnh learning rate và các tham số chống overfit:

```bat
python scripts/tune_cnn_zscore.py
```

Các kết quả validation được lưu tại:

```text
reports/cnn_zscore_regularization_tuning/tuning_summary.csv
reports/cnn_zscore_regularization_tuning/best_config.json
```

Script thử các tham số:

```text
cnn_learning_rate
cnn_weight_decay
cnn_dropout
cnn_label_smoothing
cnn_lr_factor
cnn_lr_patience
```

Chỉ chạy test một lần sau khi đã chọn xong cấu hình tốt nhất:

```bat
python scripts/tune_cnn_zscore.py --run-final-test
```

## Xuat CNN + z-score lam dau vao cho phan tiep theo

Artifact on dinh duoc luu tai:

```text
models/cnn_zscore/best_model.pt
models/cnn_zscore/model_artifact.json
```

Checkpoint chua trong so model, kich thuoc anh, thu tu lop va tham so z-score dung khi train.
Xuat lai artifact tu ket qua CNN + z-score da co:

```bat
python scripts/export_cnn_zscore_artifact.py
```

Nap model trong phan tiep theo:

```python
from src.models.simple_cnn import load_simple_cnn_checkpoint

model, checkpoint = load_simple_cnn_checkpoint("models/cnn_zscore/best_model.pt")
mean = checkpoint["normalization_params"]["mean"]
scale = checkpoint["normalization_params"]["scale"]
class_names = checkpoint["class_names"]
```
