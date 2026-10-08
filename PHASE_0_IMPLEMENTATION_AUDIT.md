# Phase-0 Implementation Audit: Clinical Belief State (CBS)
## Clinical Belief State: A Persistent Patient Representation for Longitudinal Medical Vision-Language Models

**Date:** 2026-10-07  
**Environment:** Windows 11 Home Single Language 10.0.26300  
**Working Directory:** C:\CBS  
**Audit Status:** Complete - No model implementation started  

---

## 1. Current Environment

### Python Version
- **Python 3.12.7** (C:\Users\madis\AppData\Local\Programs\Python\Python312\python.exe)
- pip version: 24.2

### Environment Management
- No conda installation detected
- No virtualenv at default paths
- Standard Python 3.12 installation with pip package management
- No evidence of poetry, pyenv, or other environment managers

### Installed Packages (ML-Relevant)
| Package | Version | Purpose |
|---------|---------|---------|
| **datasets** | 4.0.0 | Hugging Face datasets library |
| **huggingface_hub** | 1.27.0 | HF Hub interaction |
| **langchain** | 0.3.27 | LLM orchestration framework |
| **langchain-core** | 0.3.86 | Core langchain components |
| **numpy** | 2.2.6 | Numerical computing |
| **pandas** | 3.0.5 | Data manipulation |
| **pillow** | 12.3.0 | Image processing |
| **scikit-learn** | 1.9.0 | Machine learning utilities |
| **scipy** | 1.16.0 | Scientific computing |
| **torch** | **NOT INSTALLED** | PyTorch deep learning framework |
| **transformers** | **NOT INSTALLED** | Hugging Face transformers |
| **monai** | **NOT INSTALLED** | Medical imaging AI framework |
| **timm** | **NOT INSTALLED** | PyTorch image models |
| **peft** | **NOT INSTALLED** | Parameter-efficient fine-tuning |
| **accelerate** | **NOT INSTALLED** | Training acceleration |
| **opencv-python** | 5.0.0.93 | Computer vision |
| **matplotlib** | 3.11.1 | Plotting/visualization |
| **tensorflow_cpu** | 2.21.0 | TensorFlow CPU-only |
| **tokenizers** | 0.23.1 | Fast tokenizers |
| **tiktoken** | 0.13.0 | OpenAI tokenizer |

### Missing Critical Dependencies
- **PyTorch** (core deep learning framework)
- **Transformers** (LLM/vision-language model support)
- **MONAI** (medical imaging specific tools)
- **timm** (vision encoder implementations)
- **PEFT** (parameter-efficient fine-tuning)
- **Accelerate** (distributed training)
- **torchvision** (computer vision utilities)
- **einops** (tensor operations)

---

## 2. Available Hardware

### CPU
- **AMD Ryzen 7 6800H with Radeon Graphics**
- **16 logical cores** (8 physical cores with SMT)
- Base frequency: ~3.2 GHz, Boost: up to 4.7 GHz

### Memory (RAM)
- **16 GB DDR4** system memory
- Single channel configuration (inferred from laptop platform)

### Storage
- **C: Drive:** 475.8 GB total, 70.3 GB free (NTFS)
- Sufficient for dataset storage and experiment outputs

### GPU Detection
#### Primary GPU (Integrated)
- **AMD Radeon(TM) Graphics**
- 512 MB VRAM (shared system memory)
- Driver: 31.0.12024.10003
- OpenCL 2.1 support

#### Secondary GPU (Discrete)
- **AMD Radeon RX 6650M**
- **4096 MB VRAM** (4 GB GDDR6)
- Driver: 31.0.12024.10003
- OpenCL 2.1 support
- 14 compute units, 2222 MHz clock
- **6458 MB global memory** (first device in clinfo)
- **8573 MB global memory** (second device in clinfo)

### Compute Capabilities
- **OpenCL 2.1** available on both AMD GPUs
- **No NVIDIA GPUs detected** (nvidia-smi not found)
- **No CUDA toolkit installed**
- **No ROCm installation** at default paths
- **No DirectML support** (torch-directml not installed)
- **ROCm platform detected** via clinfo (AMD Accelerated Parallel Processing)

### GPU Computing Limitations
- No CUDA support means **PyTorch cannot utilize GPU acceleration** without ROCm/DirectML
- OpenCL support available but not commonly used for deep learning
- Would require **ROCm installation** or **DirectML backend** for GPU acceleration
- Alternative: Use CPU-only training (significantly slower) or cloud GPU resources

---

## 3. Dependency Audit Summary

### Currently Available
- Basic scientific Python stack (numpy, scipy, scikit-learn, pandas)
- Data handling (datasets, huggingface_hub)
- LLM orchestration (langchain suite)
- Image processing (opencv-python, pillow)
- Visualization (matplotlib)
- NLP utilities (tokenizers, tiktoken)

### Missing for Phase 1
| Dependency | Status | Installation Command | Notes |
|------------|--------|----------------------|-------|
| **torch** | ❌ | `pip install torch` | Choose CPU-only or ROCm version |
| **transformers** | ❌ | `pip install transformers` | Required for LLMs/vision encoders |
| **monai** | ❌ | `pip install monai` | Medical imaging specific |
| **timm** | ❌ | `pip install timm` | Vision encoder implementations |
| **peft** | ❌ | `pip install peft` | Parameter-efficient fine-tuning |
| **accelerate** | ❌ | `pip install accelerate` | Distributed training |
| **torchvision** | ❌ | `pip install torchvision` | Computer vision utilities |
| **einops** | ❌ | `pip install einops` | Tensor operations |
| **wandb** | ❌ | `pip install wandb` | Experiment tracking (recommended) |
| **tensorboard** | ❌ | `pip install tensorboard` | Alternative experiment tracking |

### Environment Suitability for Phase 1
- **❌ NOT SUITABLE** - Critical ML dependencies missing
- **⚠️ LIMITED GPU UTILIZATION** - No CUDA/ROCm/DirectML configured
- **✅ SUFFICIENT RAM** - 16 GB adequate for small-scale experiments
- **✅ SUFFICIENT STORAGE** - 70+ GB free on C: drive
- **✅ ADEQUATE CPU** - 16 cores sufficient for data preprocessing

### Required Actions Before Phase 1
1. Install PyTorch with appropriate GPU backend (ROCm recommended for AMD)
2. Install transformers, MONAI, timm, PEFT, Accelerate
3. Verify GPU accessibility from Python
4. Consider experiment tracking setup (wandb/tensorboard)

---

## 4. Dataset Audit

### 4.1 MIMIC-CXR
**Purpose:** Primary dataset for longitudinal chest X-ray analysis and report generation  
**Required Files:**
- MIMIC-CXR-JPG-2.0.0.physionet.org (image files)
- MIMIC-CXR-ECG-2.0.0.physionet.org (ECG waveforms, optional)
- MIMIC-IV (hospital admissions, ICU stays, demographics)
- CheXpert labels (optional, for auxiliary tasks)
- MIMIC-CXR metadata files (cxr-study-list.csv, mimic-cxr-2.0.0-split.csv)

**Access Requirements:**
- PhysioNet credential (free account required)
- Data Usage Certificate (DUCS) completion
- CITI training completion
- Formal data use agreement

**Expected Structure:**
```
/mimic-cxr/
├── files/
│   ├── p10/
│   │   └── p10000032/
│   │       ├── s10000032/
│   │       │   ├── xxxxxx-xxxx-xx-xx-xx-xxxx.dcm
│   │       │   └── xxxxxx-xxxx-xx-xx-xx-xxxx.png
│   │       └── ...
└── mimic-cxr-2.0.0-split.csv
├── mimic-cxr-2.0.0-chexpert.csv
└── cxr-study-list.csv
```

**Identifiers:**
- **Patient ID:** subject_id (integer)
- **Study ID:** study_id (integer, DICOM series level)
- **Image ID:** dicom_id (string)
- **Hadm_ID:** Hospital admission ID (links to MIMIC-IV)

**Availability:**
- **Images:** JPEG/PNG conversions available (recommended for efficiency)
- **Reports:** Available via MIMIC-IV note events (radiology reports)
- **Temporal:** Study timestamps available (chartdate, charttime)
- **Labels:** CheXpert labels available for 14 pathologies

**Licensing/Data Constraints:**
- PhysioNet Credentialed Health Data License (requires DUCS)
- Non-commercial use only for some components
- HIPAA Safe Harbor compliant (de-identified)

**Phase 1 Relevance:** **REQUIRED** - Primary dataset for Phase 1 pilot (1K-2K patients)

**Postponement Possible:** No - core to Phase 1 objectives

### 4.2 Open-I (Open Access Series of Inpatient Chest X-rays)
**Purpose:** Auxiliary dataset for pretraining/evaluation, smaller scale  
**Required Files:**
- NIH Chest X-rays dataset (images + reports)
- CSV metadata files with labels and findings

**Access Requirements:**
- Publicly available via NIH Cloud or direct download
- No formal agreement required (public domain)

**Expected Structure:**
```
/NIH/
├── images/
│   ├── 00000001_000.png
│   ├── 00000001_001.png
│   └── ...
├── Data_Entry_2017.csv
├── BBox_List_2017.csv
└── train_val_list.txt
```

**Identifiers:**
- **Image ID:** UUID-based filenames
- **Patient Level:** Limited patient-level tracking (mostly de-identified)

**Availability:**
- **Images:** PNG format available
- **Reports:** Text reports included in metadata
- **Temporal:** Limited temporal information (mostly single timepoint)
- **Labels:** 14 disease labels (CheXpert-like)

**Licensing/Data Constraints:**
- Public domain (CC0 or similar)
- No restrictions on use

**Phase 1 Relevance:** **OPTIONAL** - Useful for pretraining vision encoders or auxiliary evaluation

**Postponement Possible:** Yes - can be postponed to Phase 2+ for auxiliary experiments

### 4.3 MI-CXR (MIMIC-IV Chest X-ray)
**Purpose:** Updated version of MIMIC-CXR with better alignment to MIMIC-IV  
**Required Files:**
- Similar structure to MIMIC-CXR but integrated with MIMIC-IV v2.x
- Enhanced metadata and temporal alignment

**Access Requirements:**
- Same as MIMIC-CXR (PhysioNet credential + DUCS)

**Expected Structure:**
- Integrated file structure with MIMIC-IV
- Enhanced temporal resolution

**Identifiers:**
- Aligns with MIMIC-IV subject_id, hadm_id, stay_id

**Availability:**
- Similar to MIMIC-CXR (images + reports)

**Licensing/Data Constraints:**
- Same as MIC-CXR (PhysioNet license)

**Phase 1 Relevance:** **ALTERNATIVE** - Could substitute MIMIC-CXR if preferred version

**Postponement Possible:** Yes - use MIMIC-CXR for Phase 1, evaluate MI-CXR later

### 4.4 CheXTemporal
**Purpose:** Temporal chest X-ray dataset for tracking disease progression  
**Required Files:**
- Paired temporal chest X-rays with change annotations
- Radiologist-generated change labels (improved/stable/worsened)

**Access Requirements:**
- Likely requires special access (temporal annotations)
- May be internal or collaboration-based

**Expected Structure:**
```
/chextemporal/
├── baseline/
│   ├── patient001_study001.png
│   └── ...
├── followup/
│   ├── patient001_study001_week02.png
│   └── ...
├── change_labels.csv
└── patient_metadata.csv
```

**Identifiers:**
- Paired baseline/followup study IDs
- Patient identifiers linking to source

**Availability:**
- **Images:** Likely available
- **Reports:** Source reports may be available
- **Temporal:** Explicit temporal pairing
- **Labels:** Change annotations (primary outcome)

**Licensing/Data Constraints:**
- [PROPOSAL DOES NOT SPECIFY] - Likely requires data use agreement
- **Options:** 1) Internal collaboration dataset, 2) Public temporal dataset substitute, 3) Synthetic generation

**Phase 1 Relevance:** **LATER PHASE** - Not needed for initial frozen-decoder pilot

**Postponement Possible:** Yes - Phase 3+ for temporal analysis validation

### 4.5 MS-CXR-T (MIMIC-Stillwater Chest X-ray Temporal)
**Purpose:** Multi-institutional temporal chest X-ray dataset  
**Required Files:**
- Temporal chest X-ray pairs from multiple institutions
- Standardized protocols and annotations

**Access Requirements:**
- [PROPOSAL DOES NOT SPECIFY] - Likely institutional access required
- May require data sharing agreements

**Expected Structure:**
```
/ms-cxr-t/
├── site_A/
│   ├── baseline/
│   └── followup/
├── site_B/
│   ├── baseline/
│   └── followup/
├── unified_labels.csv
└── site_metadata.csv
```

**Identifiers:**
- Multi-site patient/study identifiers
- Harmonized across institutions

**Availability:**
- Similar temporal structure to CheXTemporal

**Licensing/Data Constraints:**
- [PROPOSAL DOES NOT SPECIFY] - Multi-institutional likely requires IRB/DSAs

**Phase 1 Relevance:** **LATER PHASE** - Validation of generalizability

**Postponement Possible:** Yes - Phase 3+ for multi-site validation

### 4.6 CheXpert
**Purpose:** Auxiliary labels for chest X-ray pathology detection  
**Required Files:**
- CheXpert-v1.0-small/chexpert.csv (train/valid split)
- Associated image files (typically from MIMIC-CXR or similar)

**Access Requirements:**
- Publicly available via Stanford ML Group
- Standard data use agreement

**Expected Structure:**
```
/chexpert/
├── CheXpert-v1.0-small/
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv
└── CheXpert-v1.0/
    ├── train/
    └── valid/
```

**Identifiers:**
- Aligns with source dataset (typically MIMIC-CXR paths)
- Study-level identifiers

**Availability:**
- **Images:** Requires pairing with source dataset (e.g., MIMIC-CXR)
- **Reports:** Not included (labels only)
- **Temporal:** Not inherently temporal
- **Labels:** 14 pathology labels (No Finding, Enlarged Cardiomediastinum, etc.)

**Licensing/Data Constraints:**
- Non-commercial research use (CheXpert license)
- Attribution required

**Phase 1 Relevance:** **SUPPLEMENTAL** - Useful for auxiliary pretraining or evaluation

**Postponement Possible:** Yes - can be used for pretraining in Phase 1 or evaluation in Phase 2

### Dataset Summary for Phase 1
- **Primary:** MIMIC-CXR (REQUIRED)
- **Auxiliary/Pretraining:** Open-I, CheXpert (OPTIONAL)
- **Temporal Validation:** CheXTemporal, MS-CXR-T (LATER PHASES)
- **Total Estimated Phase 1 Size:** ~1,500 patients × 1.5 studies/patient ≈ 2,250 studies

---

## 5. Architecture Candidates Audit

### 5.1 Vision Encoders

#### Vision Transformer (ViT)
- **Public Implementation:** `torchvision.models.vit_*` or `timm.models.vit_*`
- **HF Availability:** Yes (google/vit-*, facebook/dino-*)
- **Model Sizes:** ViT-Tiny (5M), ViT-Small (22M), ViT-Base (86M), ViT-Large (307M)
- **Input/Output:** 224×224×3 → 768-1024 dim features
- **GPU Requirements:** 2-8 GB VRAM (batch size dependent)
- **Frozen-Decoder Suitability:** Good - produces fixed-size image embeddings
- **Implementation Complexity:** Low (standard implementation)
- **Licensing:** Apache 2.0 (torchvision), MIT (timm)

#### BiomedCLIP
- **Public Implementation:** `huggingface:Microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
- **HF Availability:** Yes
- **Model Sizes:** ViT-Base (86M vision) + PubMedBERT (110M text) = ~200M total
- **Input/Output:** 224×224×3 → 512 dim joint embedding
- **GPU Requirements:** 4-6 GB VRAM
- **Frozen-Decoder Suitability:** Excellent - designed for vision-language tasks, trained on medical data
- **Implementation Complexity:** Low (HF transformers)
- **Licensing:** Microsoft Research License (non-commercial use permitted)

#### DINOv2
- **Public Implementation:** `facebookresearch/dinov2` or `timm.models.vit_*`
- **HF Availability:** Yes (facebook/dinov2-*)
- **Model Sizes:** ViT-Small (22M), ViT-Base (86M), ViT-Large (307M)
- **Input/Output:** 224×224×3 → 384-1024 dim features
- **GPU Requirements:** 2-8 GB VRAM
- **Frozen-Decoder Suitability:** Good - self-supervised pretraining yields strong features
- **Implementation Complexity:** Low-Medium (requires specific preprocessing)
- **Licensing:** Apache 2.0 (research use), CC-BY-NC 4.0 (some weights)

**Recommendation for Phase 1:** **BiomedCLIP** - Medical domain pretraining, vision-language alignment, proven effectiveness in medical VLMs

### 5.2 Report Encoders

#### BioClinicalBERT
- **Public Implementation:** `emilyalsentzer/Bio_ClinicalBERT`
- **HF Availability:** Yes
- **Model Size:** 110M parameters (BERT-base)
- **Input/Output:** Token sequences → 768 dim pooled output
- **GPU Requirements:** 2-4 GB VRAM
- **Frozen-Decoder Suitability:** Excellent - clinical text understanding
- **Implementation Complexity:** Low (standard HF model)
- **Licensing:** MIT (clinical use permitted)

**Alternatives Considered:**
- ClinicalBERT (yamanda/ClinicalBERT-base)
- PubMedBERT (microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract)
- RoBERTa-based clinical variants

**Recommendation:** **BioClinicalBERT** - Specifically trained on MIMIC-III notes, optimal for radiology reports

### 5.3 Disease Memory Modules

#### GRU (Gated Recurrent Unit)
- **Public Implementation:** `torch.nn.GRU`
- **Model Size:** Scales with hidden size (e.g., 256-1024 dim)
- **Input/Output:** Sequence → hidden state (same dim)
- **GPU Requirements:** Minimal (lightweight recurrent)
- **Frozen-Decoder Suitability:** Good - proven for temporal modeling
- **Implementation Complexity:** Very Low (built-in PyTorch)
- **Licensing:** BSD-style (PyTorch license)

#### Mamba (State Space Model)
- **Public Implementation:** `state-spaces/mamba` or `mamba-ssm` package
- **Model Size:** Comparable to transformer/RNN of similar width
- **Input/Output:** Sequence → state/hidden output
- **GPU Requirements:** Efficient (linear scaling vs quadratic)
- **Frozen-Decoder Suitability:** Excellent - designed for long sequences, efficient inference
- **Implementation Complexity:** Medium (requires custom installation)
- **Licensing:** BSD 3-Clause

#### State Space Model (S4/S5 Variants)
- **Public Implementation:** Various research implementations
- **Model Size:** Similar to Mamba
- **Input/Output:** Sequence → transformed state
- **GPU Requirements:** Efficient
- **Frozen-Decoder Suitability:** Good - theoretical advantages for long-range dependencies
- **Implementation Complexity:** Medium-High (less standardized)
- **Licensing:** Varies (often MIT/BSD)

**Recommendation for Phase 1:** **GRU** - Simplicity, reliability, sufficient for 1K-2K patient trajectories  
**Alternative for Phase 2+:** **Mamba** - Better scaling for longer trajectories

### 5.4 Prior/Posterior Networks

#### MLP (Multi-Layer Perceptron)
- **Public Implementation:** `torch.nn.Sequential` with Linear layers
- **Model Size:** Configurable (e.g., 2×[768→512→256] for 256-dim latent)
- **Input/Output:** Context vector → distribution parameters (μ, σ)
- **GPU Requirements:** Minimal
- **Frozen-Decoder Suitability:** Standard approach
- **Implementation Complexity:** Very Low
- **Licensing:** BSD-style (PyTorch)

**Recommendation:** Standard 2-layer MLPs with ReLU activation, LayerNorm optional

### 5.5 Decoders (Frozen for Phase 1)

#### LLaMA-family Medical LLMs
- **Candidates:**
  - **BioMedLM** (stanford-crfm/BioMedLM) - 2.7B parameters
  - **ClinicalT5** (yam360/ClinicalT5-base) - 220M parameters
  - **PMC-LLaMA** (various) - 7B/13B parameters
  - **Med-Alpaca** / **MedLLaMA** variants - 7B parameters
  - **OpenBioLLM** (openbiollm/OpenBioLLM-Llama3-70B) - 70B parameters
- **HF Availability:** Varies by model
- **Model Sizes:** 220M (ClinicalT5) to 70B+ (OpenBioLLM)
- **Input/Output:** Token sequences → next token logits
- **GPU Requirements for Inference:** 
  - 220M: ~1.5 GB VRAM (quantized)
  - 3B: ~6 GB VRAM (quantized)
  - 7B: ~12 GB VRAM (quantized) - **feasible for Phase 1**
  - 13B+: Requires multi-GPU or advanced quantization
- **Frozen-Decoder Suitability:** Excellent - designed for medical text generation
- **Implementation Complexity:** Low-Medium (HF transformers)
- **Licensing:** Varies (often research/commercial variants)

**Recommendation for Phase 1:** **ClinicalT5-base** or **BioMedLM** - Smaller size fits 16GB RAM/4GB GPU constraints, medical domain pretraining  
**Alternative:** **PMC-LLaMA-7B** with 4-bit quantization (~4GB VRAM) if higher performance needed

---

## 6. Baseline Comparison Matrix

### Phase 1 Confound-Isolation Matrix

| Baseline | Description | Key Mechanism | Temporal Handling | Confounds Isolated | Implementation Notes |
|----------|-------------|---------------|-------------------|-------------------|----------------------|
| **A. Cross-sectional Current-Only** | Uses only current study (O_t) | No history | None | History bias, temporal confounding | Simplest baseline - establishes lower bound |
| **B. Retrieval/History** | Retrieves k most priors, averages/concatenates | Explicit retrieval | Fixed window | Recency bias, fixed-window limitation | Simple but effective history baseline |
| **C. Generic Recurrent Memory** | Standard RNN/GRU over report/image sequence | Learned recurrence | Variable-length | Architecture-specific biases | Controls for recurrent vs specialized memory |
| **D. CBS (Proposed)** | Belief state (H_t, Z_t) with recursive update | Structured belief tracking | Infinite horizon | All previous confounds | Main experimental condition |
| **E. Context Transformer** | Transformer over full history + current | Self-attention | Full history (quadratic) | Recurrent vs attention bias | Controls for sequence modeling approach |

### Later Phase Baselines (Post-Phase 1)

| Baseline | Description | Key Mechanism | Phase Relevance |
|----------|-------------|---------------|-----------------|
| **Previous Report Prompting** | Conditions decoder on R_{t-1} + X_t | Simple concatenation | Phase 2 - establishes reporting baseline |
| **Multi-image Longitudinal Encoder** | CNN/RNN over image sequence only | Visual temporal modeling | Phase 2 - isolates visual temporal vs report |
| **CXRMate** | [PROPOSAL DOES NOT SPECIFY] - Likely temporal patch-based approach | [PROPOSAL DOES NOT SPECIFY] | Phase 2+ - if defined in literature |
| **CXRMate-2** | [PROPOSAL DOES NOT SPECIFY] - Enhanced variant | [PROPOSAL DOES NOT SPECIFY] | Phase 2+ - if defined |
| **MAIRA-2 Reimplementation** | Medical AI Radiology Assistant v2 | Retrieval-augmented generation | Phase 2 - SOTA comparison |
| **MAIRA-2 Public Checkpoint** | Official MAIRA-2 weights | Retrieval-augmented generation | Phase 2 - if publicly available |
| **MLRG-style Missingness Handling** | Missingness-aware gating/imputation | Explicit missingness modeling | Phase 2+ - robustness testing |

**Notes on Unspecified Baselines:**
- [PROPOSAL DOES NOT SPECIFY] exact implementation of CXRMate/CXRMate-2
- **Options:** 1) Temporal patch matching, 2) Longitudinal feature aggregation, 3) Siamese temporal networks
- [PROPOSAL DOES NOT SPECIFY] MAIRA-2 public availability
- **Options:** 1) Check if released, 2) Reimplement from paper, 3) Use alternative SOTA

---

## 7. Phase-1 Experiment Protocol

### 7.1 Patient-Level Split
- **Total Target:** 1,500 patients (midpoint of 1K-2K range)
- **Split Strategy:** 
  - **Training:** 1,000 patients (66.7%)
  - **Validation:** 250 patients (16.7%) 
  - **Test:** 250 patients (16.7%)
- **Stratification:** By age, sex, and primary pathology prevalence (to ensure representative splits)
- **Patient Identification:** `subject_id` from MIMIC-IV/MIMIC-CXR
- **Leakage Prevention:** Strict patient-level split - no patient appears in multiple splits

### 7.2 Trajectory Construction
- **Selection Criteria:** Patients with ≥2 chest X-ray studies
- **Study Ordering:** Chronological by `chartdate` (primary), `charttime` (secondary tiebreaker)
- **Minimum Interval:** [PROPOSAL DOES NOT SPECIFY]
  - **Options:** 1) No minimum (all pairs), 2) ≥6 months (clinically meaningful), 3) ≥1 year (disease progression focus)
  - **Recommendation:** ≥6 months - balances sample size with clinical relevance
- **Maximum Interval:** [PROPOSAL DOES NOT SPECIFY]
  - **Options:** 1) No maximum, 2) ≤5 years (avoid unrelated comorbidities), 3) ≤10 years (full EHR span)
  - **Recommendation:** ≤5 years - focuses on active disease trajectories
- **Trajectory Length:** Variable (2 to N studies per patient)
- **Padding Strategy:** [PROPOSAL DOES NOT SPECIFY]
  - **Options:** 1) No padding (variable-length batches), 2) Pad to max length in batch, 3) Fixed-length sampling
  - **Recommendation:** Variable-length with packing - preserves all data without waste

### 7.3 Missing-Modality Represention
- **Missing Image:** Use learnable `[IM_NULL]` token embedding
- **Missing Report:** Use learnable `[REP_NULL]` token embedding  
- **Implementation:** Concatenate modality embeddings with null indicators
- **Alternative:** [PROPOSAL DOES NOT SPECIFY] mask-conditioned approach
  - **Options:** 1) Separate modality encoders with masking, 2) Modality-type embeddings + masking, 3) Gated fusion based on availability
  - **Recommendation:** Learnable null tokens + modality-type embeddings - simple and effective

### 7.4 Preprocessing Pipelines

#### Image Preprocessing
- **Source:** DICOM → PNG/JPEG (use provided conversions if available)
- **Resize:** 224×224 (standard for vision encoders)
- **Normalization:** 
  - **Option A:** ImageNet stats [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
  - **Option B:** BiomedCLIP-specific stats (if available)
  - **Option C:** Per-channel normalization from training set
  - **Recommendation:** Option B if available, else Option A
- **Augmentation (Train only):** 
  - Random rotation (±5°)
  - Random translation (±2%)
  - Random scale (0.9-1.1)
  - Random contrast (0.8-1.2)
  - **Note:** Minimal augmentation to preserve pathology fidelity

#### Report Preprocessing
- **Source:** MIMIC-IV `noteevents` (category = 'Radiology')
- **Text Cleaning:**
  - De-identification (already done in MIMIC)
  - Lowercase
  - Basic punctuation normalization
  - [PROPOSAL DOES NOT SPECIFY] handling of section headers
    - **Options:** 1) Preserve, 2) Remove, 3) Convert to standard format
    - **Recommendation:** Preserve - may contain useful structural information
- **Tokenization:** BioClinicalBERT tokenizer
- **Truncation:** [PROPOSAL DOES NOT SPECIFY] max length
  - **Options:** 1) 128 tokens, 2) 256 tokens, 3) 512 tokens
  - **Recommendation:** 256 tokens - balances context with efficiency
- **Padding:** Right-pad to batch max length

#### Metadata Preprocessing
- **Variables to Extract:**
  - **Demographics:** Age (at study time), sex, ethnicity, insurance
  - **Clinical:** Hospital admission time, ICU stay, ventilator usage
  - **Temporal:** Hours since admission, study delay from admission
  - **Prior History:** Number of prior studies, time since last study
- **Encoding Strategy:** [PROPOSAL DOES NOT SPECIFY]
  - **Options:** 1) Standardization (z-score) for continuous, 2) Embedding lookup for categorical, 3) Bucketing + embedding
  - **Recommendation:** Mixed approach - standardization for continuous, embeddings for categorical
- **Fusion:** Concatenate with image/report embeddings before belief state update

### 7.5 Temporal Representation
- **Delta-Time:** Hours since previous study (Δt = t_current - t_previous)
- **Encoding:** [PROPOSAL DOES NOT SPECIFY]
  - **Options:** 1) Linear scaling, 2) Log scaling (log(1+Δt)), 3) Bucketed embedding, 4) Sinusoidal positioning
  - **Recommendation:** Log scaling + linear projection - handles wide dynamic range
- **Integration:** Concatenate delta-time encoding with current observation before belief update

### 7.6 Model Configuration (Phase 1 - Frozen Decoder)

#### Frozen Components
- **Vision Encoder:** BiomedCLIP vision tower (weights frozen)
- **Report Encoder:** BioClinicalBERT (weights frozen) 
- **Decoder LLM:** ClinicalT5-base or BioMedLM (weights frozen)
- **Rationale:** Isolates belief state learning effectiveness

#### Trainable Components
- **Prior Network:** MLP (2 layers: 768→512→latent_dim×2 for μ, logσ)
- **Posterior Network:** MLP (2 layers: 768+768→512→latent_dim×2 for μ, logσ)  
- **Belief State Updater (f_θ):** GRU (input: [H_{t-1}, Z_{t-1}, O_t], hidden: belief_dim)
- **Latent Dimension (Z_t):** [PROPOSAL DOES NOT SPECIFY]
  - **Options:** 1) 32, 2) 64, 3) 128
  - **Recommendation:** 64 - good capacity/efficiency tradeoff
- **Belief State Dimension (H_t):** [PROPOSAL DOES NOT SPECIFY]
  - **Options:** 1) Same as Z_t, 2) 2×Z_t, 3) Fixed 512
  - **Recommendation:** 128 - allows rich history representation
- **Distribution Type:** Diagonal Gaussian (output μ, logσ from MLPs)

#### Loss Terms (Phase 1)
- **L_report:** Negative log likelihood of generated report R_t
  - **Implementation:** Cross-entropy loss on decoder output
  - **Teacher Forcing:** [PROPOSAL DOES NOT SPECIFY] ratio
    - **Options:** 1) 0.0 (no teacher forcing), 2) 0.5, 3) 1.0 (full teacher forcing)
    - **Recommendation:** 0.5 - balances exposure bias with training stability
- **L_consistency:** [PROPOSAL DOES NOT SPECIFY] formulation
  - **Options:** 1) KL(q_t||p_t) symmetry loss, 2) L2 distance between means, 3) Wasserstein distance
  - **Recommendation:** Symmetric KL: 0.5*(KL(q||p) + KL(p||q)) - encourages agreement
- **L_KL:** Standard KL divergence KL(q_t||p_t) 
  - **Weight (λ₂):** [PROPOSAL DOES NOT SPECIFY] 
    - **Options:** 1) 0.01, 2) 0.1, 3) 1.0 (annealed from 0.001 to 1.0)
    - **Recommendation:** 0.1 - moderate regularization strength
- **Consistency Weight (λ₁):** [PROPOSAL DOES NOT SPECIFY]
  - **Options:** 1) 0.1, 2) 0.5, 3) 1.0
  - **Recommendation:** 0.5 - balanced with report loss

#### Training Configuration
- **Optimizer:** AdamW
- **Learning Rate:** [PROPOSAL DOES NOT SPECIFY]
  - **Options:** 1) 1e-4, 2) 3e-4, 3) 5e-4
  - **Recommendation:** 3e-4 - standard for transformer adaptation
- **LR Strategy:** [PROPOSAL DOES NOT SPECIFY]
  - **Options:** 1) Constant, 2) Linear warmup + cosine decay, 3) Step decay
  - **Recommendation:** Linear warmup (10%) + cosine decay
- **Batch Size:** [PROPOSAL DOES NOT SPECIFY] (sequences per batch)
  - **Options:** 1) 8, 2) 16, 3) 32
  - **Recommendation:** 16 - fits in 4-6GB VRAM with BiomedCLIP + ClinicalT5
- **Gradient Accumulation:** [PROPOSAL DOES NOT SPECIFY] steps
  - **Options:** 1) 1 (no accumulation), 2) 4, 3) 8
  - **Recommendation:** 4 - effective batch size of 64
- **Mixed Precision:** FP16 (torch.autocast) - **Recommended** for memory/speed
- **Checkpointing:** Every epoch + best validation loss
- **Early Stopping:** Patience = 5 epochs on validation loss
- **Random Seeds:** 
  - **Torch:** 42
  - **Numpy:** 42
  - **Python:** 42
  - **Recommendation:** Fixed seeds for reproducibility
- **Experiment Tracking:** Weights & Biases (wandb) - project: "clinical-belief-state-phase1"
- **Expected Outputs:**
  - Trained belief state updater (f_θ)
  - Learned prior/posterior networks
  - Validation/test set metrics
  - Generated reports for qualitative analysis
  - Latent trajectory visualizations (t-SNE/UMAP of H_t, Z_t)

### 7.7 Evaluation Protocol
- **Primary Metrics:**
  1. **Disease Progression Accuracy:** 
     - Task: Predict change label (improved/stable/worsened) between t-1 and t
     - Method: Classify from [H_t, Z_t] or [H_t - H_{t-1}, Z_t - Z_{t-1}]
     - Baseline Comparison: Against retrieval and recurrent baselines
  2. **Missing-History Robustness:**
     - Task: Report generation with varying history lengths (0 to N priors)
     - Metric: Report quality (RadGraph F1, CheXpert label accuracy) vs history length
     - Expectation: CBS should degrade gracefully vs sharp drops in baselines
  3. **Belief Calibration:**
     - Task: Uncertainty estimation from posterior q(Z_t|H_t,O_t)
     - Metric: Expected Calibration Error (ECE) on pathology presence prediction
     - Method: Bin predictions by confidence, compare accuracy vs confidence
- **Secondary Metrics (Phase 2+):**
  - RadGraph F1 score (report clinical correctness)
  - CheXpert label AUC-ROC (14 pathologies)
  - BERTScore / BLEU / ROUGE-L (language quality)
  - Human radiologist evaluation (later phases)

---

## 8. Proposed Repository Structure

```
clinical-belief-state/
├── .github/
│   └── workflows/                 # CI/CD pipelines
├── .gitignore
├── CITATION.cff                   # Citation metadata
├── LICENSE                        # License file (MIT/Apache-2.0)
├── README.md                      # Project overview
├── CONTRIBUTING.md                # Contribution guidelines
├── CODE_OF_CONDUCT.md             # Conduct standards
│
├── configs/                       # Configuration files
│   ├── base.yaml                  # Base configuration
│   ├── phase1.yaml                # Phase 1 specific config
│   ├── phase2.yaml                # Phase 2 config
│   └── datasets/                  # Dataset-specific configs
│       ├── mimic_cxr.yaml
│       ├── open_i.yaml
│       └── chexpert.yaml
│
├── data/                          # Data management (symlinks to external)
│   ├── manifests/                 # Dataset manifests
│   │   ├── mimic_cxr_manifest.json
│   │   └── open_i_manifest.json
│   ├── processed/                 # Preprocessed data (cache)
│   │   ├── mimic_cxr/
│   │   └── open_i/
│   └── raw/                       # Raw data (DO NOT COMMIT - gitignore)
│       ├── mimic_cxr/             # Symlink to external storage
│       └── open_i/
│
├── docs/                          # Documentation
│   ├── methodology/               # Technical approach
│   ├── experiments/               # Experiment logs
│   └── api/                       # API reference
│
├── src/                           # Source code
│   ├── __init__.py
│   ├── belstate/                  # Core belief state implementation
│   │   ├── __init__.py
│   │   ├── model.py               # CBS model architecture
│   │   ├── prior.py               # Prior network
│   │   ├── posterior.py           # Posterior network
│   │   └── updater.py             # Belief state updater (f_θ)
│   │
│   ├── data/                      # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── datasets/              # Dataset-specific loaders
│   │   │   ├── mimic_cxr.py
│   │   │   ├── open_i.py
│   │   │   └── chexpert.py
│   │   ├── transforms/            # Image/text transforms
│   │   │   ├── image_transforms.py
│   │   │   └── text_transforms.py
│   │   └── collate.py             # Batch collation with missingness handling
│   │
│   ├── models/                    # Encoder/decoder wrappers
│   │   ├── __init__.py
│   │   ├── vision_encoder.py      # BiomedCLIP/VIT wrapper
│   │   ├── report_encoder.py      # BioClinicalBERT wrapper
│   │   └── text_decoder.py        # ClinicalT5/BioMedLM wrapper
│   │
│   ├── training/                  # Training loops and utilities
│   │   ├── __init__.py
│   │   ├── trainer.py             # Main training loop
│   │   ├── optimizer.py           # Optimizer/scheduler setup
│   │   ├── losses.py              # Loss function implementations
│   │   └── callbacks.py           # Training callbacks (checkpointing, early stopping)
│   │
│   ├── evaluation/                # Evaluation and metrics
│   │   ├── __init__.py
│   │   ├── metrics.py             # Metric calculations
│   │   ├── evaluator.py           # Evaluation loop
│   │   └── viz/                   # Visualization utilities
│   │
│   └── utils/                     # Utility functions
│       ├── __init__.py
│       ├── helpers.py             # General helpers
│       ├── logging.py             # Logging setup
│       └── seeds.py               # Random seed management
│
├── experiments/                   # Experiment outputs
│   ├── phase1/                    # Phase 1 experiment outputs
│   │   ├── run_20261007_001/      # Timestamped experiment dir
│   │   │   ├── config.yaml        # Used configuration
│   │   │   ├── logs/              # Training logs
│   │   │   │   ├── train.log
│   │   │   │   └── events.out.tfevents.*  # TensorBoard
│   │   │   ├── checkpoints/       # Model checkpoints
│   │   │   │   ├── epoch_000.pt
│   │   │   │   └── best_model.pt
│   │   │   ├── metrics/           # Evaluation metrics
│   │   │   │   ├── train_metrics.json
│   │   │   │   ├── val_metrics.json
│   │   │   │   └── test_metrics.json
│   │   │   ├── samples/           # Generated report samples
│   │   │   │   ├── epoch_000_samples.json
│   │   │   │   └── best_samples.json
│   │   │   └── latents/           # Latent trajectory analysis
│   │   │       ├── train_latents.npy
│   │   │       └── val_latents.npy
│   │   └── aggregated_results.csv # Across multiple seeds
│   │
│   └── phase2/                    # Phase 2 experiment outputs (future)
│
├── requirements/                  # Dependency specifications
│   ├── base.txt                   # Core dependencies
│   ├── phase1.txt                 # Phase 1 specific
│   └── dev.txt                    # Development dependencies
│
├── scripts/                       # Utility scripts
│   ├── prepare_data.py            # Data preparation/download
│   ├── train_phase1.py            # Phase 1 training launcher
│   ├── evaluate.py                # Evaluation launcher
│   └── analyze_latents.py         # Latent space analysis
│
├── tests/                         # Unit and integration tests
│   ├── __init__.py
│   ├── test_model.py
│   ├── test_data_loaders.py
│   └── test_losses.py
│
└── run.py                         # Entry point (optional)
```

**Notes:**
- All paths use POSIX style (works on Windows with Python)
- `data/raw/` contents are symlinked to external storage (not committed)
- Experiment outputs stored under `experiments/` with timestamped runs
- Configuration managed via YAML files (using OmegaConf or similar)
- Logging configured to output to both console and files

---

## 9. Missing Specification Items

### [PROPOSAL DOES NOT SPECIFY] Items with Options

#### 1. **Minimum/Maximum Study Intervals for Trajectories**
- **Options:** 
  - A) No interval constraints (use all available pairs)
  - B) ≥6 months minimum, ≤5 years maximum (clinically meaningful)
  - C) ≥1 year minimum, ≤10 years maximum (long-term progression)
- **Recommendation:** **B** - Balances sample size with clinical relevance for disease progression

#### 2. **Latent Dimensions (Z_t, H_t)**
- **Options for Z_t:**
  - A) 32-dimensional (compact)
  - B) 64-dimensional (balanced)
  - C) 128-dimensional (expressive)
- **Options for H_t:**
  - A) Same dimension as Z_t
  - B) 2×Z_t dimension
  - C) Fixed 512-dimensional (matching encoder output)
- **Recommendation:** **Z_t=64, H_t=128** - Good tradeoff of capacity and efficiency

#### 3. **Teacher Forcing Ratio for Report Generation**
- **Options:**
  - A) 0.0 (no teacher forcing - pure generation)
  - B) 0.5 (balanced approach)
  - C) 1.0 (full teacher forcing - maximum likelihood)
- **Recommendation:** **B (0.5)** - Reduces exposure bias while maintaining training stability

#### 4. **Consistency Loss Formulation (L_consistency)**
- **Options:**
  - A) Symmetric KL: 0.5*(KL(q\|p) + KL(p\|q))
  - B) L2 distance between means: ‖μ_q - μ_p‖²
  - C) Wasserstein-2 distance for Gaussians
- **Recommendation:** **A** - Principled probabilistic agreement measure

#### 5. **Weight Hyperparameters (λ₁, λ₂)**
- **Options for λ₁ (consistency):**
  - A) 0.1 (weak consistency pressure)
  - B) 0.5 (balanced)
  - C) 1.0 (strong consistency pressure)
- **Options for λ₂ (KL weight):**
  - A) 0.01 (weak regularization)
  - B) 0.1 (moderate, common default)
  - C) 1.0 (strong regularization)
  - D) Annealed schedule (0.001 → 1.0 over training)
- **Recommendation:** **λ₁=0.5, λ₂=0.1** - Balanced approach with moderate regularization

#### 6. **Missing History Encoding Strategy**
- **Options:**
  - A) Learnable null tokens per modality ([IMG_NULL], [REP_NULL])
  - B) Modality-type embeddings + binary missing indicators
  - C) Gated fusion network that conditions on availability
- **Recommendation:** **A** - Simple, effective, learns optimal null representations

#### 7. **Temporal Delta-Time Encoding**
- **Options:**
  - A) Linear scaling of hours
  - B) Log scaling: log(1 + hours)
  - C) Bucketed embedding (0-24h, 1-7d, 1-30d, etc.)
  - D) Sinusoidal positional encoding (like Transformers)
- **Recommendation:** **B** - Handles wide dynamic range of clinical intervals

#### 8. **Validation/Test Metrics for Disease Progression**
- **Options:**
  - A) Accuracy on 3-way change classification (improved/stable/worsened)
  - B) F1-score per class (macro-averaged)
  - C) AUC-ROC for binary worsening vs non-worsening
- **Recommendation:** **A** - Simple, interpretable clinical metric

#### 9. **Experiment Tracking Tool**
- **Options:**
  - A) Weights & Biases (wandb)
  - B) TensorBoard
  - C) MLflow
  - D) Simple CSV logging
- **Recommendation:** **A** - Best balance of features, usability, and medical ML adoption

#### 10. **Checkpoint Naming Convention**
- **Options:**
  - A) `epoch_{epoch:03d}.pt`
  - B) `step_{step:06d}.pt`
  - C) `{metric_name}_{value:.4f}.pt`
  - D) Combined: `epoch_{epoch:03d}_{val_loss:.4f}.pt`
- **Recommendation:** **D** - Informative and sortable

---

## 10. Risks and Mitigations

### Technical Risks
| Risk | Probability | Impact | Mitigation Strategy |
|------|-------------|--------|---------------------|
| **GPU Inaccessibility** | Medium | High | 1) Install ROCm for AMD GPU, 2) Use CPU fallback for development, 3) Plan for cloud GPU credits |
| **Memory OOM with Large Models** | Medium | Medium | 1) Gradient checkpointing, 2) Model parallelism, 3) Activation recomputation, 4) Smaller batch sizes |
| **Data Loading Bottleneck** | Low | Medium | 1) Preprocess and cache, 2) Use efficient formats (LMDB, TFRecord), 3) Multi-process data loading |
| **Convergence Failure** | Medium | High | 1) Learning rate sweeps, 2) Gradient clipping, 3) Loss function diagnostics, 4) Simpler architectures first |

### Data Risks
| Risk | Probability | Impact | Mitigation Strategy |
|------|-------------|--------|---------------------|
| **MIMIC-CXR Access Delay** | High | High | 1) Start DUCS/CITI process immediately, 2) Use Open-I/CheXpert for initial prototyping, 3) Have IRB backup plan |
| **Data Quality Issues** | Medium | Medium | 1) Comprehensive data validation scripts, 2) Visual inspection samples, 3) Statistical profiling |
| **Label Noise in Reports** | Medium | Low | 1) Use multiple metric types, 2) Focus on structured labels (CheXpert) for primary outcomes, 3) Manual spot-checking |

### Implementation Risks
| Risk | Probability | Impact | Mitigation Strategy |
|------|-------------|--------|---------------------|
| **Integration Complexity** | Medium | Medium | 1) Modular design with clear interfaces, 2) Unit tests for each component, 3) Integration tests early |
| **Reproducibility Issues** | Low | High | 1) Fixed seeds throughout, 2) Environment locking (requirements.txt), 3) Experiment tracking of all hyperparameters |
| **Scope Creep** | Medium | Medium | 1) Strict Phase 1 objectives, 2) Monthly review checkpoints, 3) Defer enhancements to Phase 2+ |

### Timeline Risks
| Risk | Probability | Impact | Mitigation Strategy |
|------|-------------|--------|---------------------|
| **Underestimated Data Prep Time** | High | Medium | 1) Parallel data downloading/preprocessing, 2) Use existing conversion scripts, 3) Start with subset for rapid iteration |
| **Dependency Installation Issues** | Medium | Low | 1) Document exact installation commands, 2) Test in clean environment, 3) Use virtual environments |

---

## 11. Compute Estimate

### Phase 1 Training Requirements (1,500 patients, ~2.25M studies)

#### Per-Epoch Computation
- **Studies per Patient:** ~1.5 (conservative estimate)
- **Total Studies:** 1,500 × 1.5 = 2,250 studies
- **Batch Size:** 16 sequences
- **Batches per Epoch:** 2,250 / 16 ≈ 141 batches
- **Forward/Backward Pass Time:** [PROPOSAL DOES NOT SPECIFY]
  - **Estimate:** 0.5-2.0 seconds/batch (BiomedCLIP + ClinicalT5 + GRU)
  - **Epoch Time:** 141 × 1.0s ≈ 2.35 minutes
- **Total Training Time:** 20-50 epochs × 2.35 min ≈ **47-118 minutes**

#### Memory Requirements
- **BiomedCLIP Vision Tower (frozen):** ~400 MB activations (batch=16)
- **ClinicalT5 Decoder (frozen):** ~600 MB activations (batch=16, seq=64)
- **GRU Belief Updater:** ~50 MB activations
- **MLP Networks:** ~20 MB activations
- **Total Activation Memory:** ~1.1 GB
- **Model Weights (frozen):** ~1.2 GB (BiomedCLIP vision + ClinicalT5)
- **Trainable Parameters:** ~50 MB (MLPs + GRU)
- **Optimizer States (AdamW):** ~100 MB (2× parameters)
- **Total VRAM Requirement:** ~2.5 GB
- **System RAM:** ~4-6 GB (data loading, preprocessing)

#### Storage Requirements
- **MIMIC-CXR Images:** ~150 GB (JPEG conversions)
- **Preprocessed Features:** ~50 GB (cached image/report embeddings)
- **Experiment Outputs:** ~10 GB (checkpoints, logs, metrics)
- **Total Storage:** ~210 GB

#### Recommended Hardware for Phase 1
- **GPU:** AMD RX 6650M (4 GB VRAM) **with ROCm** OR cloud GPU (T4, A10G)
- **Fallback:** CPU-only training (4-8× slower, ~3-15 hours)
- **Minimum RAM:** 8 GB (16 GB recommended)
- **Minimum Storage:** 50 GB free (200 GB recommended for comfort)

### Cloud Cost Estimate (if needed)
- **AWS g5.xlarge** (A10G, 24GB VRAM): ~$1.26/hour
- **Azure NCas_T4_v3** (T4, 16GB VRAM): ~$0.95/hour
- **Estimated Cost for Phase 1:** $1-3 (assuming 2-3 hours training time)

---

## 12. Recommended Implementation Order

### Immediate Next Steps (Post-Audit)
1. **Environment Setup** (Week 1)
   - Install ROCm for AMD GPU support
   - Install PyTorch with ROCm backend
   - Install transformers, MONAI, timm, PEFT, Accelerate
   - Verify GPU accessibility from Python (`torch.cuda.is_available()`)

2. **Data Access Preparation** (Week 1-2, parallel)
   - Create PhysioNet account
   - Complete CITI training
   - Submit and obtain DUCS for MIMIC-CXR access
   - Download Open-I and CheXpert for preliminary work

3. **Core Infrastructure** (Week 2)
   - Set up repository with proposed structure
   - Implement configuration management system
   - Create data loading abstractions (MIMIC-CXR, Open-I, CheXpert)
   - Implement preprocessing pipelines with caching

4. **Component Development** (Week 3-4)
   - Implement frozen encoder wrappers (BiomedCLIP, BioClinicalBERT, ClinicalT5)
   - Implement belief state components (prior, posterior, updater GRU)
   - Implement loss functions and training loop
   - Add missing modality handling and temporal encoding

5. **Phase 1 Experiment** (Week 5-6)
   - Execute data download and preprocessing for MIMIC-CXR subset
   - Run initial training experiments (small scale for debugging)
   - Full Phase 1 training run (1,500 patients)
   - Evaluation and analysis
   - Results documentation

### Dependencies Between Steps
- Environment setup must precede all development
- Data access can proceed in parallel with infrastructure
- Component development requires working data loaders
- Full experiment requires all components functional

---

## 13. Exact Next Step After Phase 0

**Immediate Action:** Begin environment setup for GPU-accelerated deep learning

**Specific Commands to Execute:**
```bash
# 1. Check current environment and install ROCm prerequisites
# (Verify Windows 11 supports ROCm - may require WSL2 or dual boot)

# 2. Install PyTorch with ROCm support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7

# 3. Install remaining dependencies
pip install transformers monai timm peft accelerate datasets opencv-python scikit-learn pandas

# 4. Install experiment tracking
pip install wandb tensorboard

# 5. Verify installation
python -c "
import torch; print('PyTorch:', torch.__version__)
import torch; print('ROCm available:', torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else 'Checking ROCm...')
import transformers; print('Transformers:', transformers.__version__)
print('Environment ready for Phase 1 development')
"
```

**Note:** If ROCm installation proves problematic on Windows, consider:
- **Option A:** Use Windows Subsystem for Linux 2 (WSL2) with Ubuntu ROCm
- **Option B:** Use cloud GPU resources (AWS/Azure/GCP) for development
- **Option C:** Proceed with CPU-only development initially (slower but functional)

---

## 14. Questions Requiring Human/Researcher Confirmation

### Critical Design Questions
1. **Trajectory Interval Constraints:** Should we use ≥6 months min / ≤5 years max for study pairs, or different thresholds?
2. **Latent Dimensions:** Confirm Z_t=64, H_t=128 or alternative dimensions?
3. **Teacher Forcing Ratio:** Use 0.5 or alternative value for report generation training?
4. **Consistency Loss:** Use symmetric KL divergence or alternative formulation?
5. **Hyperparameters:** Confirm λ₁=0.5, λ₂=0.1 or different weighting strategy?

### Data-Specific Questions
6. **Missing History Strategy:** Use learnable null tokens or alternative missing modality encoding?
7. **Temporal Encoding:** Use log-scaled delta-time or alternative temporal representation?
8. **Evaluation Metrics:** Use 3-way change classification accuracy or alternative progression metric?

### Resource Questions
9. **GPU Strategy:** Pursue ROCm installation on AMD GPU, use cloud resources, or CPU-only fallback?
10. **Data Access Timeline:** Proceed with Open-I/CheXpert prototyping while awaiting MIMIC-CXR access, or wait for full access?

### Scope Questions
11. **Baseline Implementation:** Implement all 5 Phase-1 baselines simultaneously or sequentially?
12. **Experiment Tracking:** Use Weights & Biases as primary tracking tool or alternative?

### [PROPOSAL DOES NOT SPECIFY] Clarifications Needed
13. **CXRMate/CXRMate-2:** Request clarification or literature references for these baselines?
14. **MAIRA-2 Availability:** Check if MAIRA-2 weights are publicly available for comparison?
15. **MLRG-style Missingness:** Request details on specific missingness handling approach referenced?

---

## EXPLICIT COMPLETION STATEMENT

PHASE 0 COMPLETE — NO MODEL IMPLEMENTATION STARTED.

**Audit Completion Timestamp:** 2026-10-07  
**Next Authorized Action:** Environment setup for Phase 1 development  
**Prohibition:** No model architecture implementation, no source code modification, no dataset downloading beyond access verification, no training experiments may commence until explicit approval to proceed to Phase 1 implementation.

This audit establishes the foundation for Phase 1 by documenting:
- Current environment limitations requiring dependency installation
- Hardware configuration and GPU accessibility considerations  
- Dataset access requirements and alternatives
- Architecture selections with implementation tradeoffs
- Baseline matrix for confound isolation
- Detailed Phase 1 experiment protocol with all [PROPOSAL DOES NOT SPECIFY] items flagged
- Repository structure for organized development
- Risks, mitigations, and compute requirements
- Clear next steps and open questions requiring researcher input

All implementation decisions are deferred until after researcher confirmation on open questions and successful environment setup.