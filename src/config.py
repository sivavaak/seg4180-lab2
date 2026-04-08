import os
from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

def _resolve(path):
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)

FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-key")
MODEL_CHECKPOINT_PATH = _resolve(os.getenv("MODEL_CHECKPOINT_PATH", "checkpoints/best_model.pth"))
DATA_DIR = _resolve(os.getenv("DATA_DIR", "data"))
OUTPUT_DIR = _resolve(os.getenv("OUTPUT_DIR", "outputs"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "25"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-4"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "256"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "2"))
KAGGLE_DATASET = os.getenv("KAGGLE_DATASET", "humansintheloop/semantic-segmentation-of-aerial-imagery")
BUILDING_COLOR_RGB = tuple(
    int(c) for c in os.getenv("BUILDING_COLOR_RGB", "60,16,152").split(",")
)
PATCH_SIZE = int(os.getenv("PATCH_SIZE", "256"))
TRAIN_RATIO = float(os.getenv("TRAIN_RATIO", "0.7"))
VAL_RATIO = float(os.getenv("VAL_RATIO", "0.15"))
TEST_RATIO = float(os.getenv("TEST_RATIO", "0.15"))
