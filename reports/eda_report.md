# EDA Report

## Summary

- Dataset directory: `data\dataset`
- Total images: 6854
- Valid images: 6854
- Corrupted images: 0
- Small images (< 32x32): 0
- Exact duplicate images removed from clean list: 987
- Perceptual duplicate images removed from clean list: 998
- Blurry images removed from clean list: 305
- Too dark images removed from clean list: 0
- Too bright images removed from clean list: 400
- Low contrast images removed from clean list: 2
- Low quality images removed from clean list: 682
- Clean images: 5330
- Image size for preprocessing: 224
- Train mean: [0.7106384931563183, 0.7042130995039385, 0.7148111907542388]
- Train std: [0.2802576240661145, 0.2607969792152289, 0.3392890280350243]

## Quality Thresholds

- Blur score threshold: 80.0
- Dark threshold: 40.0
- Bright threshold: 220.0
- Contrast threshold: 25.0

## Class Distribution

- algal_spot: 1072
- brown_blight: 979
- gray_blight: 1000
- healthy: 1000
- helopeltis: 1400
- red_spot: 1403

## Cleaning Summary by Class

| STT | Nhãn | Tổng ảnh | Ảnh mờ | Quá sáng | Quá tối | Tương phản thấp | Kém chất lượng | Trùng phash | Không dùng được | Còn dùng được |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | algal_spot | 1072 | 0 | 2 | 0 | 0 | 2 | 72 | 74 | 998 |
| 2 | brown_blight | 979 | 10 | 98 | 0 | 0 | 108 | 114 | 202 | 777 |
| 3 | gray_blight | 1000 | 0 | 23 | 0 | 0 | 23 | 0 | 23 | 977 |
| 4 | healthy | 1000 | 16 | 65 | 0 | 0 | 79 | 2 | 81 | 919 |
| 5 | helopeltis | 1400 | 216 | 167 | 0 | 0 | 364 | 404 | 659 | 741 |
| 6 | red_spot | 1403 | 63 | 45 | 0 | 2 | 106 | 406 | 485 | 918 |
|  | TỔNG | 6854 | 305 | 400 | 0 | 2 | 682 | 998 | 1524 | 5330 |

## Figures

![Class distribution](figures/class_distribution.png)

![Image issues by class](figures/quality_issues_by_class.png)

![Usable vs removed images by class](figures/usable_vs_removed_by_class.png)

![Initial vs clean images by class](figures/initial_vs_clean_by_class.png)

![Initial dataset t-SNE](figures/initial_tsne.png)

![Sample images](figures/sample_images.png)

## Outputs

- `data/splits/eda_metadata.csv`
- `data/splits/cleaned_metadata.csv`
- `data/splits/corrupted_images.csv`
- `data/splits/small_images.csv`
- `data/splits/exact_duplicates.csv`
- `data/splits/phash_duplicates.csv`
- `data/splits/blurry_images.csv`
- `data/splits/too_dark_images.csv`
- `data/splits/too_bright_images.csv`
- `data/splits/low_contrast_images.csv`
- `data/splits/low_quality_images.csv`
- `config/preprocessing_params.json`
- `reports/figures/class_distribution.png`
- `reports/figures/quality_issues_by_class.png`
- `reports/figures/usable_vs_removed_by_class.png`
- `reports/figures/initial_vs_clean_by_class.png`
- `reports/figures/initial_tsne.png`
- `reports/figures/sample_images.png`
- `data/splits/clean/train.csv`
- `data/splits/clean/val.csv`
- `data/splits/clean/test.csv`
