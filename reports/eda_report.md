# EDA Report

## Summary

- Dataset directory: `dataset`
- Total images: 5867
- Valid images: 5867
- Corrupted images: 0
- Small images (< 32x32): 0
- Exact duplicate images removed from clean list: 0
- Perceptual duplicate images removed from clean list: 11
- Clean images: 5856

## Class Distribution

- algal_spot: 1000
- brown_blight: 867
- gray_blight: 1000
- healthy: 1000
- helopeltis: 1000
- red_spot: 1000

## Outputs

- `data/splits/eda_metadata.csv`
- `data/splits/cleaned_metadata.csv`
- `data/splits/corrupted_images.csv`
- `data/splits/small_images.csv`
- `data/splits/exact_duplicates.csv`
- `data/splits/phash_duplicates.csv`
- `reports/figures/class_distribution.png`
- `reports/figures/sample_images.png`
- `data/splits/clean/train.csv`
- `data/splits/clean/val.csv`
- `data/splits/clean/test.csv`
