# Building Simplification

A machine learning pipeline for building simplification using pretrained models on geometric features of local triangles. This project implements regression pretraining, classification, and building simplification tasks with a modular architecture.

## Overview

This project addresses building simplification through a multi-stage approach:

1. **Regression Pretraining**: Train models on geometric properties of local triangles using the TriangleFeatures dataset
2. **Classification**: Classify building features using pretrained representations
3. **Building Simplification**: Apply learned representations to simplify building geometries

## Project Structure

```
├── data/
│   ├── TriangleFeatures/          # Geometric properties dataset for pretraining
│   ├── Classification/            # Classification dataset
│   ├── BuildingSimplification/    # Building simplification dataset
│   ├── datasets.py                # Data loading and handling utilities
│   └── utils.py                   # Data-specific utility functions
├── src/
│   ├── model_building_blocks.py   # Shared model components
│   ├── plotting.py                # Visualization utilities
│   └── utils.py                   # Project-wide utility functions
├── regression/
│   ├── models.py                  # Regression model architectures
│   ├── train.py                   # Regression training script
│   └── utils.py                   # Regression-specific utilities
├── classification/
│   ├── models.py                  # Classification model architectures
│   ├── train.py                   # Classification training script
│   ├── predict.py                 # Classification prediction script
│   └── utils.py                   # Classification-specific utilities
├── simplification/
│   ├── models.py                  # Simplification model architectures
│   ├── train.py                   # Simplification training script
│   ├── predict.py                 # Simplification prediction script
│   └── utils.py                   # Simplification-specific utilities
├── trained_models/
│   ├── pretrained_unet/          # Pretrained U-Net checkpoints from regression task
│   ├── classification/           # Trained classification model checkpoints
│   └── simplification/           # Trained simplification model checkpoints
└── requirements.txt
```

## Installation

Clone this repository:

```bash
git clone <your-repository-url>
cd building-simplification
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training

**Regression Pretraining** (on TriangleFeatures dataset):
```bash
python -m regression.train
```

**Classification Training**:
```bash
python -m classification.train
```

**Building Simplification Training**:
```bash
python -m simplification.train
```

### Prediction

Use the provided trained models from the publication:

**Classification Prediction**:
```bash
python -m classification.predict
```

**Building Simplification Prediction**:
```bash
python -m simplification.predict
```

## Datasets

The project includes three main datasets located in the `data/` folder:

- **TriangleFeatures**: Geometric properties of local triangles used for regression pretraining
- **Classification**: Dataset for building feature classification tasks
- **BuildingSimplification**: Dataset for building geometry simplification

Data loading and preprocessing utilities are provided in `data/datasets.py` and `data/utils.py`.

## Model Architecture

The project uses a U-Net architecture with the following pipeline:

1. **Pretraining**: U-Net model trained on geometric triangle features (regression task)
2. **Transfer Learning**: Pretrained features used for classification and simplification tasks
3. **Fine-tuning**: Task-specific models built on pretrained representations

Pretrained U-Net checkpoints are available in `trained_models/pretrained_unet/`.

## Trained Models

Ready-to-use model checkpoints from the publication are provided:

- **Classification Model**: `trained_models/classification/`
- **Simplification Model**: `trained_models/simplification/`
- **Pretrained U-Net**: `trained_models/pretrained_unet/`

These models can be directly used for prediction without retraining.

## Configuration

Each training script accepts command-line arguments for customization. Use the `--help` flag to see available options:

```bash
python -m regression.train --help
python -m classification.train --help
python -m simplification.train --help
```

## Results and Visualization

The `src/plotting.py` module provides utilities for visualizing results, training progress, and model outputs.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

[Add your license information here]

## Citation

If you use this work in your research, please cite our publication:

```
[Add citation information here]
```

## Support

For questions or issues, please open an issue on GitHub or contact [your-contact-info].