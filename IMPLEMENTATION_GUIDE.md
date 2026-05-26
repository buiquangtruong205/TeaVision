# Hướng Dẫn Triển Khai Dự Án Phân Loại Bệnh Lá Chè

Tài liệu này mô tả quy trình làm việc và vai trò của từng file trong project
TeaVision. Mục tiêu là xây dựng pipeline phân loại bệnh lá chè có thể tái lập:
từ dữ liệu gốc, chia tập, tiền xử lý, huấn luyện, đánh giá, suy luận, triển khai
và theo dõi chất lượng mô hình.

## 1. Nguyên Tắc Chung

- Dữ liệu chuẩn đặt trong `data/dataset/`, mỗi lớp là một thư mục con.
- Không dùng ảnh từ `val` hoặc `test` để tính `mean/std`, tune threshold, hoặc chọn tham số.
- Mọi tham số quan trọng phải nằm trong `config/config.yaml`.
- Các file sinh ra tự động như split CSV, checkpoint, figure, log nên có đường dẫn rõ ràng.
- Code nên đọc config thay vì hard-code đường dẫn hoặc hyperparameter.
- Test set chỉ dùng một lần ở cuối để báo cáo kết quả cuối cùng.

## 2. Quy Trình Làm Việc Đề Xuất

1. Bắt đầu bằng xử lý ảnh đã chia theo bệnh: kiểm tra ảnh lỗi, ảnh trùng, ảnh quá nhỏ, sai định dạng hoặc sai thư mục lớp.
2. Xác nhận dữ liệu chuẩn nằm ở `data/dataset/<class_name>/`. Nếu dữ liệu đang ở `dataset/<class_name>/`, cần chuyển hoặc đồng bộ sang `data/dataset/`.
3. Tạo `config/class_mapping.json` từ danh sách thư mục lớp.
4. Scan dataset và tạo bảng metadata gồm `image_path`, `label`, `class_name`, `width`, `height`, `file_size`.
5. Sau khi dataset đã sạch và chuẩn, chạy `dvc add data/dataset` và commit file `.dvc` vào Git.
6. Nếu dataset nhỏ, đặc biệt dưới 10.000 ảnh, chạy 5-fold stratified cross-validation trước để đánh giá độ ổn định của model.
7. Chia dữ liệu stratified thành `train.csv`, `val.csv`, `test.csv` cho lần train/evaluate cuối cùng.
8. Tính `mean/std` trên tập train và lưu vào `config/preprocessing_params.json`.
9. Huấn luyện baseline model, lưu checkpoint tốt nhất theo `val_loss`.
10. Sau khi có best model, lưu version bằng MLflow model registry hoặc Git tag tương ứng với config, code và checkpoint.
11. Đánh giá trên validation, hiệu chỉnh threshold nếu cần.
12. Chạy test set một lần để tạo báo cáo cuối.
13. Export model cho inference hoặc deployment.
14. Benchmark model trên thiết bị mục tiêu trước khi đưa vào sử dụng.
15. Theo dõi drift nếu mô hình được đưa vào production.

Cross-validation không thay thế test set cuối. Nó giúp biết mô hình có ổn định giữa các fold hay không, còn test set cuối vẫn cần được giữ riêng để báo cáo kết quả khách quan.

Vì dữ liệu của project hiện đã được chia theo từng bệnh trong thư mục, nhãn sẽ được suy ra trực tiếp từ tên thư mục. Do đó, phần quan trọng đầu tiên là làm sạch và chuẩn hóa ảnh, không phải gán nhãn lại thủ công.

## 3. Cấu Hình

### `config/config.yaml`

File cấu hình chính cho toàn bộ project. Nên có các section sau:

```yaml
project:
  name: tea-disease-project
  seed: 42

paths:
  dataset: data/dataset
  splits: data/splits
  models: models
  reports: reports
  logs: logs
  mlruns: experiments/mlruns

data:
  image_size: 224
  batch_size: 32
  num_workers: 4
  val_size: 0.15
  test_size: 0.15

preprocessing:
  use_albumentations: true
  normalize: true

augmentation:
  horizontal_flip: true
  random_rotate: 15
  color_jitter: true

model:
  name: efficientnet_b0
  pretrained: true
  dropout: 0.3
  num_classes: 2

training:
  epochs: 20
  learning_rate: 0.0003
  weight_decay: 0.0001
  patience: 5
  mixed_precision: true

threshold:
  enabled: false
  unknown_class: -1
  min_confidence: 0.5

mlflow:
  tracking_uri: experiments/mlruns
  experiment_name: tea-disease-classification

deploy:
  export_onnx: true
  api_host: 0.0.0.0
  api_port: 8000
```

### `config/class_mapping.json`

Ánh xạ id lớp sang tên lớp.

```json
{
  "0": "healthy",
  "1": "blight"
}
```

Nên sinh file này từ danh sách thư mục trong `data/dataset/` để tránh nhập sai.

### `config/preprocessing_params.json`

File sinh tự động sau khi tính thống kê trên tập train.

```json
{
  "image_size": 224,
  "mean": [0.485, 0.456, 0.406],
  "std": [0.229, 0.224, 0.225]
}
```

## 4. Dữ Liệu

### `src/data/dataset.py`

Nên định nghĩa class `TeaDiseaseDataset` kế thừa `torch.utils.data.Dataset`.

Trách nhiệm:

- Đọc `train.csv`, `val.csv`, hoặc `test.csv`.
- Mỗi CSV có ít nhất các cột `image_path`, `label`, `class_name`.
- Đọc ảnh bằng OpenCV hoặc PIL.
- Áp dụng transform.
- Trả về `image_tensor`, `label`.

Khuyến nghị: dùng Albumentations với `ToTensorV2()` để transform train/eval nhất quán.

### `src/data/preprocessing.py`

Các hàm nên có:

- `compute_mean_std(loader)`: tính mean/std chỉ trên tập train.
- `get_train_transform(image_size, mean, std, config)`: resize, augment, normalize, tensor.
- `get_eval_transform(image_size, mean, std)`: resize, normalize, tensor.
- `save_preprocessing_params(path, params)`: lưu thống kê tiền xử lý.
- `load_preprocessing_params(path)`: đọc lại thống kê khi inference.

Lưu ý: augmentation chỉ dùng cho train, không dùng cho validation hoặc test.

Sau khi `compute_mean_std(loader)` chạy xong, pipeline phải gọi `save_preprocessing_params()` để ghi `image_size`, `mean`, `std` vào `config/preprocessing_params.json`. Trong `src/models/inference.py`, bắt buộc load file này bằng `load_preprocessing_params()` để tạo transform giống lúc train. Không nên hard-code mean/std riêng trong inference vì rất dễ làm sai phân phối input.

### `src/data/splitter.py`

Các hàm nên có:

- `scan_dataset(dataset_dir)`: tạo DataFrame từ thư mục ảnh.
- `create_class_mapping(dataset_dir, output_path)`: sinh mapping lớp.
- `create_stratified_split(df, val_size, test_size, seed, output_dir)`: tạo `train.csv`, `val.csv`, `test.csv`.
- `create_kfold_split(df, n_splits, seed, output_dir)`: tạo fold cho cross validation nếu cần.

Cách chia đúng khi muốn train/val/test là 70/15/15:

1. Tách `test_size = 0.15` từ toàn bộ dữ liệu.
2. Từ phần còn lại, tách validation với tỷ lệ `0.15 / 0.85`.
3. Luôn dùng `stratify=df["label"]` để giữ phân bố lớp.

### `src/data/versioning.py`

Chỉ nên bọc các lệnh DVC cần thiết:

- `dvc_add(path)`: chạy `dvc add <path>`.
- `dvc_pull()`: chạy `dvc pull`.
- `dvc_push()`: chạy `dvc push`.

Không nên tự động `git commit` trong hàm Python. Commit nên do script hoặc người dùng quyết định để tránh tạo commit ngoài ý muốn.

Quy trình versioning đề xuất:

```bash
dvc add data/dataset
git add data/dataset.dvc .gitignore
git commit -m "Track dataset with DVC"
```

Sau mỗi lần train quan trọng, cần lưu quan hệ giữa data version, config version và model version. Cách đơn giản là log đầy đủ vào MLflow; cách chặt hơn là tạo Git tag như `model-baseline-v1` trỏ tới commit đã sinh ra checkpoint.

## 5. Mô Hình

### `src/models/model.py`

Hàm chính:

- `build_model(num_classes, model_name="efficientnet_b0", pretrained=True, dropout=0.3)`

Nên hỗ trợ trước:

- `efficientnet_b0`: baseline nhẹ, dễ chạy.
- `efficientnet_b2`: tốt hơn nếu GPU đủ.
- `resnet50`: baseline phổ biến.
- `mobilenet_v3_large`: phù hợp hơn cho edge/mobile.

Model cần thay classifier head theo `num_classes` và trả về model sẵn sàng huấn luyện.

### `src/models/train.py`

Các hàm nên có:

- `train_one_epoch(model, loader, criterion, optimizer, device, scaler=None)`.
- `validate(model, loader, criterion, device)`.
- `train_loop(config, model, train_loader, val_loader)`.

`train_loop` cần:

- Chạy theo số epoch trong config.
- Hỗ trợ mixed precision nếu bật.
- Dùng `CrossEntropyLoss` làm loss mặc định cho bài toán multi-class classification.
- Nếu dữ liệu mất cân bằng lớp, tính `class_weight` từ train set và truyền vào `CrossEntropyLoss(weight=class_weight)`.
- Log `train_loss`, `train_acc`, `val_loss`, `val_acc`.
- Lưu checkpoint tốt nhất theo `val_loss`.
- Early stopping theo `patience`.
- Log tham số và metric vào MLflow.

### `src/models/evaluate.py`

Các hàm nên có:

- `evaluate_model(model, loader, device, threshold=None)`: trả về `y_true`, `y_pred`, `y_probs`.
- `compute_metrics(y_true, y_pred, y_probs, class_names)`: tính metric và tạo báo cáo.

Metric ưu tiên cho classification:

- Accuracy.
- Precision macro/weighted.
- Recall macro/weighted.
- F1 macro/weighted.
- Confusion matrix.
- ROC-AUC và PR-AUC nếu bài toán và số lớp phù hợp.

Không cần thêm mAP hoặc Dice nếu project chỉ làm classification.

### `src/models/tune.py`

Dùng Optuna khi baseline đã chạy ổn.

Các hàm nên có:

- `objective(trial, config, train_df, val_df)`.
- `run_tuning(config)`.

Hyperparameter nên tune:

- Learning rate.
- Weight decay.
- Dropout.
- Optimizer.
- Batch size nếu tài nguyên cho phép.

Kết quả tốt nhất lưu vào `config/best_params.yaml`.

### `src/models/inference.py`

Các hàm nên có:

- `load_model_for_inference(model_path, config, device)`.
- `predict_single_image(image_path, model, transform, class_names, device, threshold=None)`.
- `predict_batch(image_paths, model, transform, class_names, device, threshold=None)`.

Inference phải dùng đúng `image_size`, `mean`, `std` đã lưu từ train.

## 6. Tiện Ích

### `src/utils/logger.py`

Nên quản lý MLflow:

- `init_mlflow(config)`.
- `log_params(params)`.
- `log_metrics(metrics, step=None)`.
- `log_artifact(path)`.

Có thể thêm context manager để bắt đầu và kết thúc run gọn hơn.

### `src/utils/metrics.py`

Chỉ giữ các metric thật sự dùng cho classification:

- `macro_f1`.
- `weighted_f1`.
- `per_class_recall`.
- `classification_metrics`.

Nếu sau này chuyển sang detection hoặc segmentation mới thêm mAP, IoU, Dice.

### `src/utils/threshold_calibration.py`

Dùng khi muốn có lớp `unknown` hoặc cần ưu tiên recall.

Các hàm nên có:

- `find_best_threshold(y_true, y_probs, method="f1")`.
- `apply_threshold(y_probs, threshold, unknown_class=-1)`.
- `save_threshold(path, threshold)`.

Threshold chỉ được tune trên validation set.

### `src/utils/reproducibility.py`

Hàm chính:

```python
def set_seed(seed=42, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

## 7. Triển Khai

### `src/deploy/optimize.py`

Nên hỗ trợ:

- Export ONNX.
- Kiểm tra ONNX bằng `onnxruntime`.
- Benchmark thời gian inference.
- Export TFLite nếu thật sự cần chạy mobile/edge.

Benchmark cần đo rõ:

- Latency trung bình theo millisecond.
- P95 latency và P99 latency.
- FPS khi chạy liên tục.
- RAM sử dụng.
- GPU memory nếu có GPU.
- Kích thước model sau export.

Nếu triển khai realtime, cần đặt yêu cầu cụ thể theo thiết bị. Ví dụ: với Raspberry Pi hoặc camera realtime, mục tiêu thường là trên 30 FPS, latency ổn định và không vượt RAM cho phép. Nếu model không đạt, ưu tiên thử MobileNet, quantization, giảm `image_size`, hoặc export ONNX/TFLite tối ưu hơn.

### `src/deploy/api.py`

FastAPI nên có:

- `GET /health`: kiểm tra server.
- `POST /predict`: nhận ảnh và trả kết quả dự đoán.
- Load model một lần khi startup, không load lại mỗi request.
- Ghi log mỗi request để phục vụ monitoring và drift detection.

Response nên gồm:

```json
{
  "class_id": 0,
  "class_name": "healthy",
  "confidence": 0.98
}
```

Log production tối thiểu nên có:

- `timestamp`.
- `client_ip`.
- `class_id`.
- `class_name`.
- `confidence`.
- `latency_ms`.
- `model_version`.

Log có thể ghi ra file JSONL, SQLite/PostgreSQL, hoặc hệ thống logging tập trung. Không nên lưu ảnh người dùng nếu chưa có chính sách dữ liệu rõ ràng; nếu cần lưu ảnh để retrain, phải tách riêng quyền truy cập và metadata.

### `src/deploy/edge/`

Chứa script riêng cho Raspberry Pi hoặc thiết bị edge nếu cần.

## 8. Monitoring

### `src/monitoring/drift_detection.py`

Theo dõi data drift:

- So sánh phân phối confidence.
- So sánh thống kê ảnh cơ bản.
- So sánh embedding nếu có feature extractor.

Ngưỡng cảnh báo ban đầu có thể dùng:

- Độ tương tự embedding trung bình dưới `0.85`.
- P-value của KS test dưới `0.05` khi so sánh phân phối confidence hoặc đặc trưng ảnh.
- Confidence trung bình giảm hơn `10%` so với baseline validation.

Các ngưỡng này chỉ là điểm khởi đầu, cần điều chỉnh theo dữ liệu thực tế sau khi có log production.

### `src/monitoring/concept_drift.py`

Theo dõi concept drift:

- Tỷ lệ dự đoán sai tăng.
- Confidence trung bình giảm.
- Lớp `unknown` tăng bất thường.

Ngưỡng cảnh báo ban đầu: nếu tỷ lệ `unknown` tăng hơn `10%` so với giai đoạn baseline hoặc recall của một lớp quan trọng giảm liên tục qua nhiều batch dữ liệu, cần kiểm tra lại dữ liệu và cân nhắc retrain.

### `src/monitoring/retrain_trigger.py`

Quyết định retrain khi:

- Drift vượt ngưỡng.
- Có đủ số mẫu mới.
- Có nhãn xác thực cho dữ liệu mới.

## 9. Notebook

### `notebooks/01_eda_cleaning.ipynb`

- Kiểm tra số ảnh mỗi lớp.
- Kiểm tra ảnh lỗi, ảnh trùng, ảnh quá nhỏ.
- Kiểm tra ảnh trùng bằng perceptual hash hoặc so sánh ảnh gần giống trước khi split.
- Vẽ một số ảnh mẫu mỗi lớp.

### `notebooks/02_preprocess_split.ipynb`

- Scan dataset.
- Tạo class mapping.
- Chia train/val/test.
- Tính mean/std.

### `notebooks/03_baseline_model.ipynb`

- Huấn luyện baseline nhanh.
- Kiểm tra overfit/underfit.
- So sánh một vài model nhẹ.

### `notebooks/04_error_analysis.ipynb`

- Xem confusion matrix.
- Liệt kê ảnh bị dự đoán sai.
- Kiểm tra lớp nào dễ nhầm.
- Đề xuất thêm dữ liệu hoặc augmentation.

## 10. Scripts

### `scripts/run_train.sh`

Chạy train từ config:

```bash
python -m src.models.train --config config/config.yaml
```

### `scripts/run_tune.sh`

Chạy Optuna tuning:

```bash
python -m src.models.tune --config config/config.yaml
```

### `scripts/run_api.sh`

Chạy API:

```bash
uvicorn src.deploy.api:app --host 0.0.0.0 --port 8000
```

### `scripts/run_drift_monitor.py`

Chạy kiểm tra drift định kỳ nếu đã có dữ liệu production.

### `scripts/run_pipeline.py`

Script chính để tự động hóa pipeline từ split, train, evaluate đến export. Script này giúp tránh bỏ sót bước khi chạy lại thí nghiệm.

Ví dụ:

```bash
python scripts/run_pipeline.py --config config/config.yaml --mode full
```

Các mode nên hỗ trợ:

- `prepare`: scan dataset, tạo mapping, chia split, tính mean/std.
- `crossval`: chạy 5-fold stratified cross-validation để đánh giá độ ổn định.
- `train`: train model từ split đã có.
- `eval`: đánh giá checkpoint.
- `export`: export model.
- `full`: chạy toàn bộ từ prepare đến export.

## 11. Tests

### `tests/test_data.py`

- Dataset đọc được CSV.
- Transform trả về tensor đúng shape.
- Split không làm mất mẫu.
- Stratify giữ phân bố lớp tương đối.

### `tests/test_model.py`

- Build model đúng số output class.
- Forward pass chạy được với tensor giả.

### `tests/test_inference.py`

- Load checkpoint.
- Predict một ảnh mẫu.
- Output có `class_id`, `class_name`, `confidence`.

### `tests/test_drift.py`

- Hàm drift trả về đúng kiểu dữ liệu.
- Retrain trigger hoạt động theo ngưỡng.

## 12. Thứ Tự Ưu Tiên Triển Khai

Ưu tiên cao:

1. Chuẩn hóa `config/config.yaml`.
2. Hoàn thiện `splitter.py`, `dataset.py`, `preprocessing.py`.
3. Tạo CSV split thật từ dataset hiện có.
4. Chạy 5-fold stratified cross-validation nếu dataset nhỏ hoặc mất cân bằng.
5. Huấn luyện baseline model.
6. Đánh giá và tạo báo cáo.
7. Ghi version dataset/model bằng DVC, MLflow hoặc Git tag.

Ưu tiên sau:

1. MLflow đầy đủ.
2. Optuna tuning.
3. Threshold calibration.
4. FastAPI inference.
5. ONNX/TFLite export.
6. Drift monitoring.

## 13. Lưu Ý Cho Repo Hiện Tại

- Repo đang có thư mục `dataset/` ở root. Nên chuyển hoặc đồng bộ dữ liệu sang `data/dataset/` để khớp cấu trúc chuẩn.
- `data/processed/` và `data/augmented/` chỉ nên dùng làm cache, không nên commit ảnh sinh ra nếu dung lượng lớn.
- Checkpoint thật như `best_model.pth`, `best_model.onnx`, `best_model.tflite` chỉ xuất hiện sau khi train/export.
- Nếu dùng DVC, nên track dữ liệu lớn bằng DVC thay vì Git.
