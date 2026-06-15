
# Reducing Hospital Costs Through ML-Driven Diabetic Readmission Prevention

![Python](https://img.shields.io/badge/Language-Python-blue.svg)
![Model](https://img.shields.io/badge/Model-LightGBM-success)


Unplanned 30-day hospital readmissions cost the U.S. healthcare system over **$26 billion annually**, with diabetic patients accounting for a disproportionately large share of avoidable re-hospitalization costs. This project builds a machine learning pipeline to flag high-cost readmission risk at discharge — giving care coordinators an actionable window to intervene before costs escalate.

Using the UCI Diabetes 130-US Hospitals dataset, I engineered clinically meaningful features, handled severe class imbalance, and optimized for **recall** over raw accuracy — because in a cost-containment context, a missed high-risk patient is far more expensive than a false alarm.

## Project Structure

```
diabetic-readmission-cost-prevention/
├── diabetic-prediction.md
├── diabetic-prediction.pdf               # Full project documentation  
├── Data/
│   ├── diabetic_data.csv
│   ├── IDS_mapping.csv
│   └── cleaned_data.csv
├── Code.ipynb                            # End-to-end notebook with code & analysis
├── requirements.txt
└── tuned_model.joblib                    # Production-ready saved model
```

## Business Problem

Every hospital readmission within 30 days is a signal of discharge risk that wasn't caught in time. For diabetic patients specifically, the complexity of comorbidities, polypharmacy, and post-discharge care gaps makes this prediction especially valuable.

**Goal**: Identify patients at discharge who are likely to return within 30 days — so care managers can schedule follow-ups, arrange home health services, or extend observation before costs compound.

## Dataset Overview

- **Source**: UCI Diabetes 130-US Hospitals (1999–2008)
- **Records**: ~100,000 hospital encounters across 130 U.S. hospitals
- **Features**: 50 variables covering demographics, diagnoses, procedures, lab results, medications, and discharge details
- **Target Variable**: Binary — readmitted within 30 days (`<30`) vs. not

## Methodology

### 1. Data Cleaning & Standardization
- Removed administrative identifiers with no predictive value
- Replaced `"?"` placeholders with proper missing value handling
- Filled missing fields by clinical context (e.g., no lab test = "none", not unknown)

### 2. Cost-Relevant Feature Engineering
- **ICD-9 Grouping**: Mapped raw diagnosis codes into 12 clinical categories to reduce noise and surface patterns that drive expensive readmissions
- **Discharge Risk Flags**: Created binary indicators for discharge destination (home vs. facility vs. hospice) — discharge routing is one of the strongest cost signals
- **Insulin & Medication Signals**: Encoded treatment intensity as binary features — under-treated patients at discharge are disproportionately represented in readmissions
- **Interaction Features**: `age × inpatient history` captures the compound risk of older patients with high prior utilization
- **Admission Pathway**: Flagged emergency admissions, facility transfers, and ER-direct admissions — these pathways predict both readmission and higher downstream costs

### 3. Exploratory Data Analysis
- Confirmed 11.1% positive class rate — severe imbalance requiring explicit handling
- Identified age, prior inpatient visits, and discharge destination as strongest cost-linked signals
- Analyzed medication complexity, lab result severity, and length of stay distributions

### 4. Model Development
- **Algorithms evaluated**: Logistic Regression, Random Forest, XGBoost, LightGBM
- **Imbalance strategies**: `class_weight='balanced'`, `scale_pos_weight`, SMOTE
- **Validation**: Stratified 5-fold cross-validation with F1-score optimization
- **Interpretability**: Feature importance rankings and SHAP values for audit trail

### 5. Evaluation Framework
Optimized for **recall** (catching high-risk patients) rather than accuracy, since the cost of missing a readmission far outweighs the cost of a false positive intervention.

## Model Performance Summary

| Model                  | Recall | Precision | F1-Score | Accuracy |
|------------------------|--------|-----------|----------|----------|
| LightGBM (Tuned)       | 0.62   | 0.18      | 0.28     | 0.65     |
| XGBoost (Tuned)        | 0.60   | 0.18      | 0.27     | 0.65     |
| Logistic Regression    | 0.54   | 0.17      | 0.25     | 0.66     |
| SMOTE + LightGBM (0.1) | 0.81   | 0.16      | 0.24     | 0.52     |

The tuned LightGBM model with `class_weight='balanced'` offers the best balance of recall and precision for operational deployment.

## Top Cost-Driving Risk Factors

| Feature | Cost Implication |
|---|---|
| `number_inpatient` | Repeat inpatient history is the single strongest predictor — these patients have known care gaps |
| `discharged_to_facility` / `discharged_home` | Facility discharge reduces readmission risk; home discharge elevates it |
| `insulin_used` | Active insulin adjustments signal unstable glycemic control at discharge |
| `diabetesMedication` | Medication-managed patients need tighter post-discharge follow-up |
| `age_inpatient_interaction` | Elderly patients with high utilization history carry compounding readmission costs |
| `num_lab_procedures` | High lab order volume reflects clinical complexity and instability |
| `num_medications` | Polypharmacy increases post-discharge adherence risk and adverse events |

## Why Precision Is Low — and Why That's Acceptable

At ~18% precision, the model flags roughly 5–6 patients for every true readmission it catches. In a cost-reduction context, this is defensible:

- A care coordinator follow-up call costs ~$25–$50 in staff time
- An average diabetic readmission costs $14,000–$17,000 in hospital charges
- Even at 18% precision, the expected value per flagged patient is strongly positive

The real limitation is what this dataset can't see: medication adherence after discharge, home support availability, mental health status, and social determinants of health. These gaps explain why precision plateaus, and why this model works best as a **triage tool** feeding into a broader discharge planning workflow — not as a standalone predictor.

## Installation

```bash
pip install -r requirements.txt
```

## Reproducibility

- Python 3.9+
- All preprocessing encapsulated in `sklearn.Pipeline` — no data leakage
- Trained in Jupyter Notebook; model serialized with `joblib`
- `random_state=42` set throughout for reproducibility

## Author

**Nikhitha Jella**  
Data Professional  
Email: [nikhithajvv3@gmail.com](mailto:nikhithajvv3@gmail.com)
