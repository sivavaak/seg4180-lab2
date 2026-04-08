import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from evaluate import compute_iou, compute_dice


def test_perfect_prediction():
    pred = torch.ones(1, 1, 64, 64)
    target = torch.ones(1, 1, 64, 64)
    assert abs(compute_iou(pred, target) - 1.0) < 1e-5
    assert abs(compute_dice(pred, target) - 1.0) < 1e-5


def test_no_overlap():
    pred = torch.ones(1, 1, 64, 64)
    target = torch.zeros(1, 1, 64, 64)
    assert compute_iou(pred, target) < 0.01
    assert compute_dice(pred, target) < 0.01


def test_partial_overlap():
    pred = torch.zeros(1, 1, 64, 64)
    target = torch.zeros(1, 1, 64, 64)
    pred[:, :, :32, :] = 1.0
    target[:, :, :32, :] = 1.0
    assert abs(compute_iou(pred, target) - 1.0) < 1e-5
    assert abs(compute_dice(pred, target) - 1.0) < 1e-5


def test_half_overlap():
    pred = torch.zeros(1, 1, 4, 4)
    target = torch.zeros(1, 1, 4, 4)
    pred[:, :, :2, :] = 1.0
    target[:, :, 1:3, :] = 1.0
    iou = compute_iou(pred, target)
    dice = compute_dice(pred, target)
    assert 0.2 < iou < 0.5
    assert 0.4 < dice < 0.7
