---
title: Aerial Building Analysis
emoji: 🏢
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: "6.3.0"
python_version: "3.12"
app_file: app.py
license: mit
models:
  - MindForgeTim/building-segmentation
  - MindForgeTim/dls-gsd-model
tags:
  - segmentation
  - aerial-imagery
  - buildings
  - gsd-estimation
  - computer-vision
  - pytorch
pinned: false
---

# Aerial Building Analysis

Automatic building segmentation and area estimation from aerial/satellite imagery.

## Features

- **Building Segmentation:** UNet with ResNet-50 encoder trained on Inria Aerial dataset
- **GSD Estimation:** Automatic Ground Sampling Distance detection using RegressionTreeCNN
- **Area Calculation:** Accurate building area measurement in square meters
- **Tiling Support:** Process high-resolution images via tile-based inference
- **Building Count:** Automatic detection of individual buildings using connected components

## Usage

1. Upload an aerial/satellite image (or select from examples below)
2. (Optional) Click "Calculate Scale" to auto-detect GSD from image
3. (Optional) Adjust GSD manually if known
4. Select processing mode:
   - **Tiling (Accurate)** — for high-resolution images, processes in tiles
   - **Resize (Fast)** — quick preview, resizes to 512×512
5. Click "Analyze" to run segmentation
6. View results: segmentation mask, overlay, and statistics (area, coverage, building count)

## Examples

The demo includes sample aerial images for testing. Click on any example to load it.

## Models

| Model | Repository | Description |
|-------|------------|-------------|
| Segmentation | [MindForgeTim/building-segmentation](https://huggingface.co/MindForgeTim/building-segmentation) | UNet with ResNet-50 encoder |
| GSD Estimation | [MindForgeTim/dls-gsd-model](https://huggingface.co/MindForgeTim/dls-gsd-model) | RegressionTreeCNN (ResNet-101) |

## Dataset

Training data: [Inria Aerial Image Labeling Dataset](https://project.inria.fr/aerialimagelabeling/)

- 180 aerial images (5000×5000 pixels)
- 0.3 m/pixel resolution
- Binary building/non-building labels
- Cities: Austin, Chicago, Kitsap, Vienna, Tyrol

## Technical Details

- **Input:** RGB aerial/satellite images (any resolution)
- **Output:** Binary segmentation mask + statistics
- **Inference:** CPU-based, tile-based processing for large images
- **GSD Range:** 0.1 - 5.0 m/pixel (auto-detection supported)

## License

MIT License

## Citation

If you use this application in your research, please cite the Inria Aerial dataset:

```bibtex
@inproceedings{maggiori2017dataset,
  title={Can semantic labeling methods generalize to any city? The inria aerial image labeling benchmark},
  author={Maggiori, Emmanuel and Tarabalka, Yuliya and Charpiat, Guillaume and Alliez, Pierre},
  booktitle={IEEE International Geoscience and Remote Sensing Symposium (IGARSS)},
  year={2017}
}
```
