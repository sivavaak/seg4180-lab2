# Report Outline: Aerial House Segmentation with CI/CD Pipeline

## Page 1: Introduction and Background

- Problem statement: pixel-level house segmentation from aerial/satellite imagery
- Motivation: urban planning, disaster response, land use monitoring
- Connection to Lab 1: evolution from pretrained text classifier to custom-trained vision model
- Overview of semantic segmentation (vs. instance segmentation, object detection)
- Brief mention of relevant architectures: FCN, U-Net, DeepLab

## Page 2: Methodology

### Dataset Preparation
- Data source description (aerial imagery, resolution, geographic coverage)
- Pixel mask generation pipeline: annotation format to binary masks
- Supported annotation formats: GeoJSON polygons, Label Studio exports
- Data augmentation strategy: flips, rotations, brightness/contrast
- Train/validation split (80/20)

### Model Architecture
- U-Net with ResNet34 encoder (pretrained on ImageNet)
- Input: 256x256 RGB images, Output: single-channel binary mask
- Loss function: BCE + Dice loss (addresses class imbalance in segmentation)
- Optimizer: Adam with ReduceLROnPlateau scheduler

## Page 3: MLOps and Infrastructure

### Secrets Injection
- Environment-based configuration via python-dotenv
- Separation of secrets (.env, gitignored) from config template (.env.example)
- Runtime configuration: batch size, learning rate, paths

### CI/CD Pipeline
- GitHub Actions workflow: test, build, deploy stages
- Automated testing with pytest (model shape validation, metric correctness)
- Docker containerization for reproducible deployment
- Optional Docker Hub push via GitHub secrets
- Flask API for serving predictions

### Serving Architecture
- REST API accepting image uploads, returning base64 masks and coverage statistics
- Waitress WSGI server for production readiness

## Page 4: Results and Discussion

### Evaluation Metrics
- Intersection over Union (IoU / Jaccard Index): definition, interpretation
- Dice Coefficient (F1 Score): definition, relationship to IoU
- Per-batch and dataset-level aggregation

### Results
- Training curves: loss, IoU, Dice over epochs
- Final validation metrics (table)
- Qualitative results: side-by-side visualizations (image, ground truth, prediction)
- Analysis of failure cases (small buildings, shadows, occlusion)

### Discussion
- Comparison with baseline or related work
- Impact of data augmentation on performance
- Limitations: dataset size, geographic generalization, single-class segmentation
- Future work: multi-class segmentation, larger encoders, post-processing (CRF)
