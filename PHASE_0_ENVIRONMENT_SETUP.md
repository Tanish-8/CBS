# Phase-0 Environment Setup Report: Gate 0A Verification
## Clinical Belief State (CBS) Research Project

**Date:** 2026-10-08  
**Environment Path:** `C:\CBS\venv`  
**Host Platform:** Windows 11 Home Single Language (Build 10.0.26300)  
**Execution Scope:** Gate 0A — Environment Setup and Verification ONLY  
**Gate 0B Status:** BLOCKED (Research Design, Model Architecture, Dataset Acquisition, and Experiments Prohibited)

---

## A. Environment Creation
- **Virtual Environment Tool:** Python standard library `venv` module (`python -m venv C:\CBS\venv`)
- **Isolation Status:** [VERIFIED] Fully isolated from system Python (`C:\CBS\venv\Lib\site-packages`)
- **Virtual Environment Location:** `C:\CBS\venv`
- **Pip Upgrade:** Upgraded from initial bundle `24.2` to `26.2.1` via `python -m pip install --upgrade pip`

---

## B. Python Verification
- **Python Executable:** `C:\CBS\venv\Scripts\python.exe`
- **Reported Python Version:** `Python 3.12.7`
- **Python Build:** `tags/v3.12.7:0b05ead, Oct 1 2024, 03:06:41 [MSC v.1941 64 bit (AMD64)]`
- **Verification Status:** [VERIFIED FACT] Python version matches project baseline `Python == 3.12.7`.

---

## C. Installed Gate-0A Packages
Installed strictly within `C:\CBS\venv`:
1. `setuptools` (v84.0.0) — Core packaging utility
2. `wheel` (v0.48.0) — Wheel build packaging utility
3. `pytest` (v9.1.1) — Unit testing framework
4. `black` (v26.10.0) — Code formatting
5. `flake8` (v7.4.1) — Linting
6. `torch` (v2.14.1+cpu) — Official CPU PyTorch build (via official PyTorch CPU wheel index)
7. `torchvision` (v0.29.1+cpu) — Matching official CPU torchvision build

*Note: No research dependencies (transformers, monai, einops, datasets, etc.) were installed.*

---

## D. PyTorch CPU Verification
Executed via `C:\CBS\venv\Scripts\python.exe`:
```python
import torch

# 1. Version and Backend Inspection
torch.__version__        # '2.14.1+cpu'
torch.cuda.is_available() # False
torch.version.cuda       # None

# 2. CPU Tensor Allocation
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
# Result: tensor([1., 2., 3.], requires_grad=True), device='cpu'

# 3. CPU Tensor Operation & Autograd
y = x * 2 + 1            # tensor([3., 5., 7.], grad_fn=<AddBackward0>)
s = y.sum()
s.backward()
x.grad                   # tensor([2., 2., 2.])
```
- **Verification Outcome:** [VERIFIED FACT]
  - CPU tensor allocation and arithmetic execution succeeded.
  - Autograd backward propagation executed successfully on CPU.
  - `torch.cuda.is_available()` correctly reported `False`.
  - Host AMD Radeon RX 6650M is **not** targeted by ROCm or DirectML; local development is strictly CPU-bound.

---

## E. pytest Verification
- **Test Discovery & Execution:** Executed on `tests/test_environment_smoke.py`.
- **Smoke Test Scope:** Contains strictly basic Python version check and CPU PyTorch tensor arithmetic; contains **zero** model architecture code and **zero** CBS logic.
- **Result:**
  ```text
  platform win32 -- Python 3.12.7, pytest-9.1.1, pluggy-1.6.0 -- C:\CBS\venv\Scripts\python.exe
  collected 2 items
  tests/test_environment_smoke.py::test_python_version PASSED              [ 50%]
  tests/test_environment_smoke.py::test_torch_cpu_tensor PASSED            [100%]
  ============================== 2 passed in 1.92s ==============================
  ```
- **Status:** [VERIFIED FACT] Test runner operational.

---

## F. black Verification
- **Executable:** `C:\CBS\venv\Scripts\black.exe` (v26.10.0)
- **Execution:** `black --check tests`
- **Result:** `All done! 1 file would be left unchanged.` (Exit code: 0)
- **Status:** [VERIFIED FACT] Formatting tool operational.

---

## G. flake8 Verification
- **Executable:** `C:\CBS\venv\Scripts\flake8.exe` (v7.4.1)
- **Execution:** `flake8 tests`
- **Result:** 0 lint errors, clean exit (Exit code: 0)
- **Status:** [VERIFIED FACT] Linting tool operational.

---

## H. Hardware Metadata
Collected directly from system APIs (WMI & Python platform libraries):
- **Operating System:** Windows 11 Home Single Language (OS Version: `10.0.26300`, Platform: `Windows-11-10.0.26300-SP0`)
- **Architecture:** AMD64 / 64-bit
- **CPU Model:** AMD Ryzen 7 6800H with Radeon Graphics (8 physical cores, 16 logical processors)
- **System RAM:** ~15.21 GB available physical RAM (16 GB installed)
- **Integrated GPU:** AMD Radeon(TM) Graphics (Driver: `31.0.12024.10003`)
- **Discrete GPU:** AMD Radeon RX 6650M (Driver: `31.0.12024.10003`, VRAM: 4.0 GB)
- **ROCm Status:** [VERIFIED FACT] Not installed; not configured.
- **DirectML Status:** [VERIFIED FACT] Not installed; not configured.
- **CUDA Status:** [VERIFIED FACT] Not present on host; PyTorch CUDA is `False`.
- **Git Repository Status:** [VERIFIED FACT] Not initialized (`fatal: not a git repository`).

---

## I. Exact Package Versions
Exact package list from `pip list` in `C:\CBS\venv`:
| Package | Exact Version | Type |
|:---|:---|:---|
| `black` | 26.10.0 | Dev / Code Hygiene |
| `click` | 8.5.0 | Transitive dependency |
| `colorama` | 0.4.6 | Transitive dependency |
| `filelock` | 3.32.3 | Transitive dependency |
| `flake8` | 7.4.1 | Dev / Code Hygiene |
| `fsspec` | 2026.7.0 | Transitive dependency |
| `iniconfig` | 2.3.1 | Transitive dependency |
| `Jinja2` | 3.1.6 | Transitive dependency (PyTorch) |
| `MarkupSafe` | 3.0.3 | Transitive dependency |
| `mccabe` | 0.7.0 | Transitive dependency (flake8) |
| `mpmath` | 1.3.0 | Transitive dependency (sympy) |
| `mypy_extensions` | 1.1.0 | Transitive dependency (black) |
| `networkx` | 3.6.1 | Transitive dependency (torch) |
| `numpy` | 2.5.2 | Transitive dependency (torchvision) |
| `packaging` | 26.3 | Transitive dependency |
| `pathspec` | 1.1.1 | Transitive dependency (black) |
| `pillow` | 12.3.0 | Transitive dependency (torchvision) |
| `pip` | 26.2.1 | Package Manager |
| `platformdirs` | 4.12.4 | Transitive dependency |
| `pluggy` | 1.6.0 | Transitive dependency (pytest) |
| `pycodestyle` | 2.15.0 | Transitive dependency (flake8) |
| `pyflakes` | 4.0.3 | Transitive dependency (flake8) |
| `Pygments` | 2.21.0 | Transitive dependency (pytest) |
| `pytest` | 9.1.1 | Testing Framework |
| `pytokens` | 0.4.1 | Transitive dependency (black) |
| `setuptools` | 84.0.0 | Packaging Utility |
| `sympy` | 1.14.0 | Transitive dependency (torch) |
| `torch` | 2.14.1+cpu | PyTorch CPU Framework |
| `torchvision` | 0.29.1+cpu | Vision Utilities CPU |
| `typing_extensions` | 4.16.0 | Core Typing Utility |
| `wheel` | 0.48.0 | Packaging Utility |

---

## J. requirements-lock.txt Status
- **File Location:** `C:\CBS\requirements-lock.txt`
- **Encoding:** UTF-8
- **Generated Via:** `pip freeze` from activated `C:\CBS\venv`
- **Verification Status:** [VERIFIED FACT] File contains all 30 direct and transitive pinned packages, including `torch==2.14.1+cpu` and `torchvision==0.29.1+cpu`.

---

## K. Gate 0A Completion Status & Boundary Enforcement

### Verified Gate 0A Criteria:
- [x] `C:\CBS\venv` exists and is isolated
- [x] Python 3.12.7 active in venv
- [x] Pip 26.2.1 operational
- [x] `setuptools` and `wheel` installed
- [x] `pytest`, `black`, `flake8` installed and operational
- [x] CPU PyTorch (`2.14.1+cpu`) installed and verified
- [x] Basic CPU tensor and autograd smoke tests succeed
- [x] `torch.cuda.is_available()` confirmed False
- [x] No ROCm installed or configured
- [x] No DirectML installed or configured
- [x] `requirements-lock.txt` generated
- [x] Hardware metadata logged with verified values

### Strict Boundary Enforcement (Gate 0B Blockade):
- [x] **Zero CBS model code written**
- [x] **Zero belief-state functions or update equations implemented**
- [x] **Zero architecture selections made** (ViT/BiomedCLIP/DINOv2, GRU/Mamba/SSM, decoders unselected)
- [x] **Zero latent dimensions or hyperparameters chosen**
- [x] **Zero training loops or evaluation pipelines implemented**
- [x] **Zero dataset files downloaded** (no MIMIC-CXR, no Open-I, etc.)
- [x] **Zero research dependencies installed** (no transformers, no monai, no einops)

---

GATE 0A COMPLETE — ENVIRONMENT VERIFIED — GATE 0B REMAINS BLOCKED
