# EchoNet-Dynamic Arrhythmia Detection System

This repository implements a 9-agent clinical pipeline designed to classify cardiac rhythms solely from echocardiogram (ultrasound) videos, without requiring an electrocardiogram (ECG).

The system models the Left Ventricular (LV) volume displacement curve over time as a **mechanical ECG**, extract cardiac cycles, computes heart rate variability (HRV) metrics, tracks myocardial wall velocities via dense optical flow, fusses features, and predicts rhythms via an XGBoost classifier.

---

## 9-Agent Collective Architecture

1. **Project Manager Agent (`"The Conductor"`)**: Coordinates pipeline execution, validates video input characteristics (FPS, format), manages agent retry logic, and handles review flags.
2. **Data Agent (`"The Curator"`)**: Loads echocardiogram videos, extracts and resizes frames, normalizes pixel intensities, and applies CLAHE contrast enhancement.
3. **Segmentation Agent (`"The Anatomist"`)**: Tracks Left Ventricle boundaries using a PyTorch U-Net or a classical contour-based fallback, generating the volume curve $V(t)$ via Simpson's Method of Discs.
4. **Cardiac Cycle Agent (`"The Rhythm Reader"`)**: Performs bandpass filtering of $V(t)$, detects End-Systole (ES/R-peaks) and End-Diastole (ED) frames, and calculates Ejection Fraction (EF), Heart Rate (HR), and rhythm irregularity indices.
5. **Mechanical HRV Agent (`"The Pulse Decoder"`)**: Computes time-domain (SDNN, RMSSD, pNN50), frequency-domain (Welch PSD), and non-linear (Poincaré plot) heart rate variability parameters.
6. **Motion Analysis Agent (`"The Motion Tracker"`)**: Extracts dense optical flow vectors in the LV region, segments the wall into four quadrants, and computes wall velocities, dyssynchrony index, and motion irregularity.
7. **Feature Engineering Agent (`"The Pattern Weaver"`)**: Integrates 15 clinical and mechanical wall motion parameters (including interaction terms) and normalizes the vector.
8. **Arrhythmia Detection Agent (`"The Diagnostician"`)**: Classifies the cardiac rhythm into **Normal Sinus Rhythm (NSR)**, **Atrial Fibrillation (AFib)**, **Premature Ventricular Contractions (PVC)**, **Bradycardia**, or **Tachycardia** using a trained XGBoost classifier (falling back to clinical rules if absent).
9. **Clinical Report Agent (`"The Scribe"`)**: Compiles diagnostic findings, metrics, and evidence into structured txt and EHR-compatible JSON reports.

---

## Workspace Directory Structure

```
d:/Arrhythmia (EchoNet-Dynamic)/
├── arrhythmia_pipeline.py     # Main CLI entry point
├── requirements.txt           # Dependency requirements
├── train_classifier.py        # XGBoost training utility
├── README.md                  # Documentation
├── agents/                    # Multi-agent package
│   ├── __init__.py
│   ├── base_agent.py          # Base agent class
│   ├── project_manager_agent.py
│   ├── data_agent.py
│   ├── segmentation_agent.py
│   ├── cardiac_cycle_agent.py
│   ├── hrv_agent.py
│   ├── motion_agent.py
│   ├── feature_agent.py
│   ├── arrhythmia_agent.py
│   └── report_agent.py
├── utils/
│   └── synthetic_generator.py # Synthetic ultrasound video generator
├── models/                    # Models and parameter storage
│   ├── scaler_params.json     # Saved normalization scaler parameters
│   └── xgboost_arrhythmia.json# Saved XGBoost weights
├── reports/                   # Output clinical report directory
└── logs/                      # Log files
```

---

## Installation and Environment Setup

1. **Initialize Virtual Environment**:
   ```bash
   python -m venv .venv
   ```

2. **Install Dependencies**:
   ```bash
   .\.venv\Scripts\pip install -r requirements.txt
   ```

---

## Getting Started: Verification Workflow

Follow these steps to run a complete end-to-end verification of the pipeline:

### 1. Generate Synthetic Video Data
If EchoNet-Dynamic videos are not initially available, generate a synthetic ultrasound video containing a simulated left ventricle contracting under Atrial Fibrillation (AFib) dynamics:
```bash
.\.venv\Scripts\python utils/synthetic_generator.py --rhythm afib --output data/synthetic_afib.avi
```

### 2. Train the XGBoost Classifier
Generate a simulated training dataset representing typical clinical parameters, fit the standardization scaler, and train the XGBoost classifier model:
```bash
.\.venv\Scripts\python train_classifier.py
```
This saves:
- `models/scaler_params.json` (normalization coefficients)
- `models/xgboost_arrhythmia.json` (booster parameters)

### 3. Execute the Full Pipeline
Run the multi-agent detection pipeline on the generated video:
```bash
.\.venv\Scripts\python arrhythmia_pipeline.py --video data/synthetic_afib.avi --patient-id PAT_002
```

### 4. Inspect Clinical Reports
Review the diagnostic findings and measurements outputted by **Agent 9 (The Scribe)**:
- Text Clinical Report: `reports/PAT_002_report.txt`
- Structured EHR JSON: `reports/PAT_002_report.json`
- Execution Log: `logs/pipeline.log`
# Arrhythmia-EchoNet 
