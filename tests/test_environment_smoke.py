"""Gate 0A Environment Smoke Test.

Verifies basic Python environment, pytest execution,
and CPU PyTorch tensor allocation.
Contains NO model architecture, NO training logic,
and NO CBS components.
"""

import sys
import torch


def test_python_version():
    assert sys.version_info.major == 3
    assert sys.version_info.minor == 12


def test_torch_cpu_tensor():
    x = torch.tensor([1.0, 2.0, 3.0])
    y = x + 1.0
    assert y.tolist() == [2.0, 3.0, 4.0]
    assert x.device.type == "cpu"
