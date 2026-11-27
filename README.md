# Deep-vein-thrombosis
##### Elements may be subject to change.
# AI-Driven Hemodynamic Risk Stratification System  
### Early Detection of Perfusion Instability & Immobility-Associated Complications

---

## Problem

Hospitalized and homebound patients are at increased risk of:

- Deep Vein Thrombosis (DVT)
- Pressure injuries
- Acute Kidney Injury (AKI)
- Perfusion-related organ dysfunction

These complications are often driven by:

- Sustained hypotension  
- Hemodynamic instability  
- Poor mobility compliance  
- Inadequate repositioning  

Current monitoring systems are:

- Manual  
- Inconsistently documented  
- Reactive rather than predictive  
- Difficult to audit at scale  

---

## Objective

Develop a **non-invasive, multimodal AI system** that:

- Models hemodynamic instability using ICU physiological data  
- Tracks patient mobility and repositioning patterns  
- Generates continuous patient-level risk scores  
- Provides explainable, real-time clinical decision support  

The system shifts monitoring from **static threshold alerts** to **dynamic predictive risk stratification**.

---

## Conceptual Framework

Rather than directly predicting DVT (a multifactorial condition), the system models:

> **Sustained hypotension and perfusion impairment**  

as physiologically grounded precursors to downstream complications such as:

- Thrombotic risk  
- Acute kidney injury  
- Pressure-related tissue injury  

This ensures:
- Clinical validity  
- Alignment with available vital sign data  
- Reduced speculative inference  

---

# 🏗 System Architecture

![Architecture Diagram](./Assets/Architecture_diagram_for_DVT.png)

The system consists of two primary pipelines:

---

# Hemodynamic Risk Prediction Pipeline

## Data Sources

- MIMIC-IV (structured ICU dataset)
- MIMIC-IV Waveform Database
- eICU dataset
- NTU RGB+D 60 / 120 (Body joint model training)

### Available Vitals

- Heart Rate (HR)
- Systolic / Diastolic / Mean Arterial Pressure (SBP / DBP / MAP)
- Respiratory Rate (RR)
- SpO₂
- Temperature
- Arterial waveform data (where available)
- Laboratory markers (lactate, creatinine, WBC, etc.)

---

## Preprocessing

- Removal of incomplete ICU stays  
- Temporal alignment of vitals  
- Feature extraction from:
  - `icustays`
  - `chartevents`
  - `inputevents`
  - `labevents`
  - `diagnoses_icd`

Derived features include:

- MAP trend slopes  
- HR variability proxies  
- Hypotension duration windows  
- Lactate progression markers  

---

## Model Development

- Supervised learning models (e.g., XGBoost)
- Branching decision tree architecture
- Continuous patient-level risk scoring

### Outputs

- Risk band classification  
- Probability of sustained hypotension  
- Perfusion instability risk score  

---

## Explainability Layer

- Feature attribution methods (e.g., SHAP)
- Identification of physiological drivers
- Clinician-interpretable outputs

---

# Movement Detection & Compliance Pipeline

## Pose Extraction

- ml5.js / pose-based joint detection  
- Real-time or recorded video input  
- Extraction of skeletal joint coordinates only  
- No raw video storage  

---

## Feature Engineering

- Movement frequency  
- Joint displacement magnitude  
- Repositioning interval detection  
- Inactivity duration tracking  

Bias mitigation:

- Model trained on skeletal coordinates only  
- Removal of demographic visual features  

---

## Movement Risk Modeling

- Classifier trained on joint movement patterns  
- Real-time inference capability  
- Mobility compliance scoring  

---

# Risk Fusion Layer

The system integrates:

- Hemodynamic instability risk  
- Movement/inactivity risk signals  

To generate:

> A unified patient-level deterioration risk response

This enables:

- Multi-factor deterioration tracking  
- Combined physiological + behavioral modeling  
- Earlier intervention opportunities  

---

# Clinical Dashboard

Web-based dashboard providing:

- Continuous risk score visualization  
- Risk band categorization  
- Movement compliance metrics  
- Physiological trend graphs  
- Explainability outputs  

Designed for:

- Workflow compatibility  
- Audit-readiness  
- Decision-support integration  

---

# Core Components

- Hemodynamic predictive ML model  
- Movement detection model  
- Continuous risk scoring engine  
- Compliance analytics dashboard  
- Explainability module  
- API-based deployment framework  

---

# Deliverables

- Trained predictive ML model  
- Risk scoring engine prototype  
- Prototype of the clinical dashboard  
- Evaluation metrics report  
- Deployment feasibility assessment  

---

# Ethical & Translational Considerations

## Risks

- ICU dataset bias  
- Limited generalizability to home settings  
- False alarm burden  
- Overreliance on automated scoring  

## Mitigation Strategies

- Transparent feature attribution  
- Clinical validation workflows  
- Conservative alert threshold tuning  
- Modular deployment architecture  


# Tech Stack

## Programming Language: Python
## Prose and Feature Extraction: OpenCV, Mediapipe, Numpy
## Deep Learning Framework: Pytorch
## Model Architecture: RNNs, Bi-LSTM
## Evaluation: Scikit-learn, Grad-cam.


---

# Status
The prototype development has been competed at both conceptual and technical levels
