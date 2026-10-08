# Phase-0 Environment Strategy
## Clinical Belief State: A Persistent Patient Representation for Longitudinal Medical Vision-Language Models

**Date:** 2026-10-08  
**Purpose:** Design the development environment without installing anything  
**Status:** No installation or implementation performed  

---

## 1. LOCAL DEVELOPMENT ENVIRONMENT

### Recommended Strategy
- **Project-specific virtual environment** using Python's built-in `venv` module
- **Location:** `C:\CBS\venv` (isolated within project directory)
- **Isolation:** Completely isolated from global Python installation
- **Activation:** 
  - Windows PowerShell: `.\venv\Scripts\Activate.ps1`
  - Windows CMD: `.\venv\Scripts\activate.bat`
- **Justification:** 
  - Prevents dependency conflicts with other projects
  - Allows exact version control for reproducibility
  - Safe for experimentation without affecting system Python
  - Aligns with Python packaging best practices (PEP 405)

### Alternative Strategies (Not Recommended)
- Global Python installation: Risk of version conflicts and pollution
- Conda environment: Not installed on system; would require installation
- Poetry/Pipenv: Additional tools not currently available; adds complexity

---

## 2. PYTHON VERSION

### Determination
- **Retain Python 3.12.7** as the baseline version for development
- **Justification:**
  - Already installed and verified functional
  - Stable release with adequate package compatibility
  - No proposal-specified version constraints
  - Compatible with anticipated dependencies (PyTorch 2.x, transformers, etc.)
- **Version Recording:** 
  - Explicitly document in `requirements.txt` or `pyproject.toml`
  - Include in experiment metadata for reproducibility
- **Future Consideration:**
  - Only change if specific dependency requires different version
  - Any change must be documented and justified
  - Must maintain backward compatibility for research continuity

---

## 3. LOCAL COMPUTE ROLE

The laptop (Windows 11, AMD Ryzen 7 6800H, 16 GB RAM, AMD Radeon RX 6650M) will eventually serve as the local CPU development environment in later project phases.

> [!IMPORTANT]
> **Phase-0 Implementation Blockade:**
> This section describes the eventual hardware/computational role only. During Phase 0:
> - Model architecture code is **strictly prohibited**
> - Training loops are **strictly prohibited**
> - Evaluation implementation is **strictly prohibited**
> - Dataset downloads and processing are **strictly prohibited**
> No model code or training loops may be written at this stage.

### Eventual Permitted Activities (Post-Gate 0B Authorization)
Once the Phase-1 Research Design Gate (Gate 0B) has been formally approved, the local hardware will be permitted exclusively for:

1. **Lightweight Development**
   - Developing data loading and preprocessing utility functions
   - Writing utility functions and helper modules
   - Building configuration management systems

2. **Unit Tests**
   - Running tests for individual components (transforms, loss functions, utilities)
   - Validating configuration parsing and environment setup
   - Continuous integration-style checks on modified code

3. **Synthetic Data Tests**
   - Developing and testing with small, artificially generated synthetic tensors
   - Validating pipeline functionality without real data dependencies
   - Debugging shape mismatches and tensor operations

4. **Preprocessing Prototypes**
   - Developing and testing data transformation routines on small sample inputs
   - Validating normalization and augmentation logic

5. **Debugging**
   - Interactive debugging with breakpoints and inspection
   - Profiling code performance (CPU-focused)
   - Identifying and fixing logical errors in data flow

6. **CPU Smoke Tests**
   - Running minimal forward/backward execution sanity checks (1-2 batches on CPU)
   - Verifying loss computation and gradient flow on CPU
   - Validating checkpoint saving and loading mechanisms

### Prohibited Activities (Locally Across All Phases)
- Full-scale training runs (exceeds local GPU/VRAM capabilities)
- Processing full clinical datasets without caching strategies
- Attempting GPU-accelerated training locally (no supported CUDA/ROCm acceleration on host)
- Storing large intermediate unmanaged results (>1 GB) without explicit cleanup
- Executing any activity that violates the Phase-0 implementation blockade during Phase 0

---

## 4. CLOUD COMPUTE ROLE

### Designation
Cloud GPU resources (NVIDIA CUDA-based) are designated for:

### Required Activities (Post-Gate 0B Authorization)
1. **Full-Scale Phase 1 Training**
   - Complete training runs on the 1,000–2,000 MIMIC-CXR patient cohort
   - Experiments requiring GPU acceleration for feasible runtime
   - Hyperparameter sweeps and ablation studies
   - Final model training for evaluation

2. **Validation and Test Evaluation**
   - Running evaluation on full validation and test sets
   - Computing comprehensive metrics (RadGraph F1, CheXpert AUC, etc.)
   - Generating qualitative analysis samples
   - Statistical significance testing

3. **Baseline Comparisons**
   - Training and evaluating all Phase-1 baselines (A–E) for fair comparison
   - Ensuring identical hardware conditions across methods
   - Scaling studies for compute efficiency analysis

### Prohibited Activities on Cloud (Until Gate Is Passed)
- Any model implementation or architecture development
- Dependency installation or environment configuration
- Dataset downloading or preprocessing beyond access verification
- Writing training loops or evaluation scripts
- Making research decisions that remain blocked by Phase-0 review

### Portability Requirement
All code developed locally must be designed to run identically on cloud GPU resources with only:
- Device specification change (`cpu` → `cuda`)
- Potential batch size adjustment (based on available VRAM)
- Path remapping for data and output directories

---

## 5. DEPENDENCY GROUPS

Dependencies are logically grouped to maintain a minimal, reproducible footprint without dependency bloat.

> [!IMPORTANT]
> **Gate Separation Notice:**
> - Dependencies for **Gate 0A (Environment Setup Gate)** include only minimal packaging utilities, testing/code-hygiene tools, and a CPU PyTorch build strictly for environment verification.
> - Dependencies for **Gate 0B (Research Design Gate)** remain completely uninstalled until formal research design approval.

### 5.1 Gate 0A Minimal Tooling (Authorized for Environment Setup)
- **Runtime Baseline:** `Python == 3.12.7`
- **Packaging Utilities:**
  - `pip` (package manager)
  - `setuptools` & `wheel` (standard packaging utilities)
- **Testing & Code Hygiene:**
  - `pytest` (unit testing framework)
  - `black` (code formatting)
  - `flake8` (linting)
- **Environment Verification (CPU Only):**
  - `torch` (CPU-compatible wheel; strictly for verifying Python import, CPU tensor allocation, and basic sanity tests)
  - `torchvision` (CPU-compatible wheel; version-matched to `torch`)

### 5.2 Gate 0B Scientific & Model Dependencies (Deferred to Phase 1 Authorization)
*These packages remain strictly uninstalled until Gate 0B approval:*

- **Core Scientific Computing:**
  - `numpy` (array computing)
  - `pandas` (tabular metadata processing)
  - `scipy` (scientific computing)
  - `scikit-learn` (dataset splitting and evaluation utilities)
  - `matplotlib` (plotting/visualization)
  - `seaborn` (statistical visualization)
  - `tqdm` (progress bars)
  - `PyYAML` (configuration parsing)

- **Model Framework (Cloud CUDA-Ready):**
  - `torch` (CUDA-enabled build on cloud GPU instances)
  - `torchvision` (computer vision utilities - version-matched to torch)
  - `transformers` (Hugging Face transformers)
  - `einops` (tensor operations)

- **Evaluation & Analysis (Unresolved Candidates):**
  - `radgraph` or `chexpert` (medical report evaluation - candidate tools pending research design)
  - `rouge-score` or `bert-score` (language generation metrics)

- **Experiment Tracking (Unresolved Candidates):**
  - Candidates: `wandb`, `tensorboard`, `mlflow`
  - *Status:* Unresolved research decision. No tracking tool is selected or preferred at this stage. Tracking strategy will be formally chosen during the Gate 0B research design review.

### 5.3 Excluded Dependencies
The following packages have been explicitly audited and excluded:
- `torchaudio`: Excluded. CBS is a vision-language project on MIMIC-CXR; there is no audio modality.
- Documentation suites (`Sphinx`, `myst-parser`, `furo`, `mkdocs`, `mkdocstrings`): Excluded as unnecessary tooling bloat.
- Reporting/Office tools (`openpyxl`, `xlsxwriter`, `Jinja2`): Excluded as extraneous.
- Specialized statistical libraries (`pingouin`, `statsmodels`): Excluded unless later scientifically justified.
- Python standard library modules (`json`, `csv`, `pickle`, `multiprocessing`): Retained as built-in standard library utilities, not external pip dependencies.

---

## 6. ENVIRONMENT REPRODUCIBILITY

### Virtual Environment Strategy
- **Creation:** `python -m venv C:\CBS\venv` (within project root)
- **Activation:** Platform-specific activation scripts
- **Isolation:** No access to global site-packages
- **Documentation:** Record exact creation command and Python version

### Requirements Files
- **Staged Requirements Architecture:**
  - `requirements-dev.txt`: Gate 0A minimal development, testing, and CPU-verification dependencies
  - `requirements-cpu.txt`: CPU-only runtime dependencies (post-Gate 0B)
  - `requirements-gpu.txt`: Full cloud GPU-enabled dependencies (post-Gate 0B)
  - `requirements-lock.txt`: Exact `pip freeze` snapshot of installed packages
- **Generation:** Exact lockfiles generated only after verified installation inside `venv`
- **Archiving:** Store in version control for reproducibility

### Version Pinning Strategy
- **Two-Stage Pinning:**
  1. *Specification Stage:* Direct top-level requirements specified with compatible version constraints
  2. *Locking Stage:* Exact versions captured via `pip freeze` into `requirements-lock.txt`
- **Validation:** Verify integrity via hash-checking where appropriate

### Python Version Recording
- **Explicit Declaration:** In requirements metadata as `# Python == 3.12.7`
- **Runtime Check:** Early failure if runtime Python version mismatches 3.12.7
- **Metadata:** Record in experiment configuration and checkpoint headers

### Hardware/Backend Recording
- **Device Specification:** Record exact GPU model and driver version (if present)
- **Backend Identifier:** Log explicit backend (`"cpu"` or `"cuda"`)
- **Environment Variables:** Capture relevant runtime settings
- **Topology:** Record number of CPU cores, system RAM, and GPU memory
- **Software Stack:** Document exact Python, PyTorch, and CUDA versions

### Experiment Metadata Schema
Each experiment configuration shall capture execution environment metadata using the following abstract schema template (concrete values captured at runtime, no fabricated placeholders):

```json
{
  "environment": {
    "python_version": "<string: sys.version, e.g. 3.12.7>",
    "operating_system": {
      "platform": "<string: platform.platform()>",
      "os_name": "<string: os.name>",
      "release": "<string: platform.release()>",
      "version": "<string: platform.version()>"
    },
    "hardware": {
      "cpu_model": "<string: CPU model description>",
      "cpu_logical_cores": "<int: os.cpu_count()>",
      "system_ram_gb": "<float: total physical RAM in GB>",
      "gpu_model": "<string: GPU model identifier or 'None'>",
      "gpu_vram_gb": "<float: dedicated GPU VRAM in GB or 0.0>",
      "gpu_driver_version": "<string: GPU driver version or 'N/A'>"
    },
    "compute": {
      "backend": "<string: 'cpu' | 'cuda'>",
      "cuda_available": "<bool: torch.cuda.is_available()>",
      "cuda_version": "<string: torch.version.cuda if available else 'N/A'>"
    },
    "dependencies": {
      "<package_name>": "<string: exact package version string from pip freeze>"
    }
  },
  "experiment": {
    "experiment_id": "<string: unique experiment run identifier>",
    "timestamp_utc": "<string: ISO-8601 UTC timestamp>",
    "git_commit": "<string: 40-character git commit hash>"
  }
}
```

---

## 7. CUDA/CLOUD PORTABILITY

### Design Principles
1. **Device-Agnostic Code**
   - All tensor operations use `.to(device)` pattern
   - No hardcoded `.cuda()` or `.cpu()` calls in model or tensor scripts
   - Device selected explicitly at startup via configuration

2. **Explicit Configuration-Driven Device Selection**
   - Centralized, explicit device specification in configuration file
   - Supported values:
     - Local development: `device: "cpu"`
     - Cloud training: `device: "cuda"`
   - **No "auto" backend selection:** The ambiguous `"auto"` mode is prohibited.
   - **No silent fallback:** If the configured device is unavailable (e.g., `cuda` requested when CUDA is not present), the system must fail explicitly at initialization with a clear error. Under no circumstances may execution silently degrade or switch across backends.
   - **No unsupported local backends:** ROCm and DirectML are not supported targets for the local Windows development environment.

3. **Path Handling**
   - All data paths relative to project root or configurable environment base
   - Output directories configurable per run
   - No hardcoded absolute system paths in codebase

4. **Batch Size Adaptation**
   - Base batch size defined explicitly in configuration
   - Memory constraints managed explicitly without hidden dynamic switching
   - Minimum batch size of 1 enforced

5. **Mixed Precision**
   - Precision mode configured explicitly (e.g., FP32 on CPU; FP16 or BF16 on CUDA via `torch.autocast`)
   - Requires explicit configuration flag

6. **Seed Synchronization**
   - Identical random seeds across numpy, torch, and random
   - Deterministic algorithm flags enabled where required for reproducibility

### Validation Requirement
Before any cloud deployment:
1. Code must pass all local CPU unit tests
2. Synthetic data pipeline tests must complete successfully on CPU
3. Configuration system and device validation must be verified functional
4. No untested code paths permitted in cloud execution

---

## 8. GPU ABSTRACTION

### Recommended Configuration Concept
In the primary configuration file (e.g., `config/base.yaml` or `config/phase1.yaml`):

```yaml
# Compute configuration
compute:
  # Explicit device selection (mandatory: "cpu" or "cuda")
  # Local development:
  device: "cpu"

  # Cloud training (uncomment on cloud NVIDIA CUDA instance):
  # device: "cuda"

  # Execution policy:
  # If requested device is unavailable, fail immediately.
  # Silent fallback across compute backends is strictly prohibited.
```

### Implementation Notes (For Future Reference)
- **Do not implement now** — blocked until Gate 0B approval
- When implemented:
  - Validate device availability explicitly at application startup:
    - If `config.compute.device == "cuda"` and `not torch.cuda.is_available()`, raise `RuntimeError("Configured device 'cuda' is unavailable.")`.
    - If `config.compute.device == "cpu"`, assign `device = torch.device("cpu")`.
  - Use `torch.device(config.compute.device)` for device assignment
  - Apply `.to(device)` to all model parameters and buffers
  - Use device-aware tensor allocation (e.g., `torch.zeros(size, device=device)`)
  - Log actual validated device and CUDA runtime version at startup

### Current Status
This remains a **design concept only**. No implementation code shall be written until:
1. Environment strategy corrections are verified
2. Research design review (Gate 0B) is complete and approved

---

## 9. RESEARCH SAFETY

### Decisions That Must Remain Blocked
Per **PHASE_0_REVIEW.md** Section 2 (OPEN RESEARCH DECISIONS) and Section 10 (PHASE-1 GATE), the following **must not** be decided, implemented, or assumed until explicit authorization:

#### Architecture Selection
- Vision encoder: Choice between ViT, BiomedCLIP, DINOv2
- Disease memory: Choice between GRU, Mamba, State Space Model
- Decoder: Choice among specific LLaMA-family medical LLMs

#### Dimensional Parameters
- Latent dimension **Z_t** size
- Belief state dimension **H_t** size
- MLP hidden layer dimensions for prior/posterior networks
- Embedding dimensions for modality encoders
- Hidden size for GRU/Mamba/SSM disease memory

#### Loss Function Details
- Exact formulation of **L_report** (beyond negative log likelihood)
- Exact formulation of **L_consistency** (proposal only mentions term exists)
- Values or scheduling for **lambda_1** and **lambda_2**
- Whether KL term uses forward, reverse, or symmetric KL divergence

#### Training Protocol
- Optimizer choice (mentions only generic "optimizer")
- Learning rate value or schedule
- Batch size
- Gradient accumulation strategy
- Mixed precision usage
- Checkpointing frequency
- Early stopping criteria
- Random seed values
- Experiment tracking mechanism

#### Data Processing Details
- Minimum/maximum study interval for trajectory inclusion
- Image preprocessing details (resize, normalization, augmentation)
- Report preprocessing details (tokenization length, truncation, cleaning)
- Metadata variables to extract and encoding strategy
- Missing modality representation (proposal specifies mask-conditioned but not implementation)
- Temporal delta-time encoding strategy
- Train/validation/test split ratios or patient stratification method

#### Baseline Implementation Details
- Exact implementation of retrieval/history baseline (k value, aggregation method)
- Exact implementation of generic recurrent memory (architecture, hidden size)
- Exact implementation of context transformer (architecture, depth, heads)
- Clarification on CXRMate/CXRMate-2 implementation
- Details on MLRG-style missingness handling approach
- MAIRA-2 reimplementation specifics if attempting replication

#### Evaluation Metrics Details
- Specific metric for Disease Progression Accuracy (classification vs regression, specific labels)
- Specific robustness metrics for missing-history (beyond mentioning report quality)
- Specific calibration metrics (beyond mentioning belief calibration)
- Secondary metrics to implement later (proposal only mentions they should be identified)

### Consequences of Premature Decisions
- Invalidates research comparability
- Introduces uncontrolled variables
- Violates the scientific method for hypothesis testing
- May produce non-reproducible results
- Wastes computational resources on unsound experiments
- Undermines the purpose of the Phase-0 review process

---

## 10. GATING ARCHITECTURE

To eliminate procedural deadlocks while preserving the Phase-0 scientific blockade, project progression is governed by two sequential gates:

### Gate 0A — Environment Setup Gate
**Objective:** Establish a verified, isolated local execution environment for tooling and CPU verification.

**Allowed Actions under Gate 0A:**
- Create isolated project virtual environment at `C:\CBS\venv` using `Python 3.12.7`
- Upgrade `pip` to latest stable release
- Install minimal packaging utilities (`setuptools`, `wheel`)
- Install minimal testing and code-hygiene tools (`pytest`, `black`, `flake8`)
- Install an official CPU-compatible PyTorch development build (`torch`, `torchvision` CPU wheels) strictly for environment verification
- Execute basic CPU smoke tests (verifying Python import, `torch.__version__`, CPU tensor allocation, and `pytest` discovery)
- Record exact environment state into `requirements-lock.txt` and capture system metadata

**Gate 0A Completion Criteria:**
- `C:\CBS\venv` exists and activates cleanly
- CPU PyTorch imports successfully and executes simple CPU tensor operations
- Basic `pytest` runs without error
- Environment lockfile generated
- Zero model code, zero datasets, zero training loops created

---

### Gate 0B — Research Design Gate
**Objective:** Formally resolve and approve all open research decisions from PHASE_0_REVIEW.md.

**Strictly Prohibited Until Gate 0B is Formally Approved:**
- Architecture selection (vision encoder, disease memory, decoder choices)
- Dimensional parameter selection (latent dimensions $Z_t$, $H_t$, MLP hidden layers, embeddings)
- Loss function formulation and loss weights ($L_{\text{report}}$, $L_{\text{consistency}}$, $\lambda_1$, $\lambda_2$)
- Training protocol decisions (optimizers, learning rate schedules, batch sizes)
- Experiment tracking platform selection
- Dataset acquisition, downloading, or preprocessing (MIMIC-CXR, Open-I, etc.)
- CBS model architecture implementation
- Training loops or parameter optimization
- Empirical experiments or evaluation runs

**Gate 0B Completion Criteria:**
- All open research decisions listed in Section 9 resolved and documented
- Research design document produced, verified against proposal constraints, and approved

---

## 11. CONCLUSION

This corrected environment strategy establishes an isolated, reproducible development foundation while maintaining a strict, unambiguous blockade on all scientific research decisions. It delineates explicit roles for local CPU development and cloud NVIDIA CUDA training, eliminates speculative device fallbacks, minimizes dependency scope, and decouples technical environment hygiene (Gate 0A) from research design authorization (Gate 0B).

---

## FINAL GATE STATUS

Gate 0A — Environment Setup: READY AFTER DOCUMENT CORRECTIONS  
Gate 0B — Research Design: BLOCKED PENDING FORMAL RESEARCH DESIGN REVIEW  

ENVIRONMENT STRATEGY CORRECTED — READY FOR GATE 0A ENVIRONMENT SETUP