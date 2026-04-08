import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from model import create_unet


def test_unet_output_shape():
    model = create_unet()
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    assert out.shape == (2, 1, 256, 256), f"Expected (2,1,256,256), got {out.shape}"


def test_unet_sigmoid_range():
    model = create_unet()
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    probs = torch.sigmoid(out)
    assert probs.min() >= 0.0 and probs.max() <= 1.0
