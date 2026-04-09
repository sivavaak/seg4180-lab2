# Lab 2: Aerial House Segmentation

Semantic segmentation pipeline for detecting houses in aerial/satellite imagery using U-Net with a ResNet34 encoder.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # edit as needed
```

## Dataset

Place aerial images in `data/images/` and corresponding binary masks in `data/masks/`. Mask filenames must match image filenames with a `.png` extension.

If you have polygon annotations instead of masks, generate masks using:

```python
from src.dataset import generate_masks_from_geojson
generate_masks_from_geojson("data/images", "annotations.geojson", "data/masks")
```

Or from Label Studio JSON exports:

```python
from src.dataset import generate_masks_from_label_studio
generate_masks_from_label_studio("data/images", "annotations.json", "data/masks")
```

## Training

```bash
cd src
python train.py
```

Configuration is via `.env` or environment variables (see `.env.example`).

## Evaluation

```bash
cd src
python evaluate.py
```

Reports mean IoU and Dice score on the full dataset.

```bash
python plot_metrics.py
python plot_samples.py
```

Saves visual plots of loss and validation metrics, along with side-by-side comparisons of prediction samples.

## Inference

```bash
cd src
python predict.py --image ../data/images/test.png --mask ../data/masks/test.png --output ../outputs/result.png
```

## API

```bash
cd src
python app.py
# POST an image to http://localhost:5000/predict
curl -X POST -F "image=@test.png" http://localhost:5000/predict
```

## Docker

```bash
docker compose up --build
```

## CI/CD

GitHub Actions workflow runs tests on push/PR and optionally pushes to Docker Hub. Set `DOCKER_HUB_USERNAME` and `DOCKER_HUB_TOKEN` as repository secrets.

## Tests

```bash
pytest tests/ -v
```
