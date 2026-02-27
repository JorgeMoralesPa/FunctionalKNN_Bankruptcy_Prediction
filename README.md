# FunctionalKNN_Bankruptcy_Prediction

Code and fully reproducible data pipeline for a functional k-NN model with a custom distance metric for bankruptcy prediction.

---

## Overview

This repository implements a complete experimental pipeline for bankruptcy prediction using a functional k-nearest neighbors (k-NN) classifier with a custom-designed distance metric.

The methodological components include:

- Construction of multivariate financial trajectories
- Definition of a penalized and weighted functional distance
- Inclusion of categorical and temporal components in the metric
- Hyperparameter optimization using Optuna
- Evaluation via cross-validation, learning curves, and bootstrap analysis

The repository allows full reproducibility from scratch.

---

## Repository Structure

```
FunctionalKNN_Bankruptcy_Prediction/
│
├── Data/
│   └── functional_space_base.parquet
│
├── scripts/
│   ├── 09_hyperparameter_optimization.py
│   ├── 10_compute_distance_matrix_knn.py
│   ├── 11_evaluate_model_with_distance_matrix.py
│   └── 12_learning_curve_and_bootstrap.py
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Full Reproducibility (Google Colab)

Open a new Google Colab notebook and run the following single cell:

```python
# Clean previous versions
!rm -rf FunctionalKNN_Bankruptcy_Prediction

# Clone repository
!git clone https://github.com/JorgeMoralesPa/FunctionalKNN_Bankruptcy_Prediction.git
%cd FunctionalKNN_Bankruptcy_Prediction

# Install dependencies
!pip -q install -r requirements.txt

# Create output directories
!mkdir -p outputs/figures outputs/data outputs/models outputs/matrices outputs/reports outputs/results

# Run full pipeline
!python scripts/09_hyperparameter_optimization.py
!python scripts/10_compute_distance_matrix_knn.py
!python scripts/11_evaluate_model_with_distance_matrix.py
!python scripts/12_learning_curve_and_bootstrap.py

# Display generated outputs
!find outputs -type f | sort
```

---

## Computational Notes

- Step 09 (Optuna optimization) is computationally intensive.
- Runtime depends on the hardware provided by Google Colab.
- The optimization stage may take a considerable amount of time.
- All subsequent steps rely on artifacts generated in previous stages.

---

## Pipeline Description

### Step 09 — Hyperparameter Optimization
- Builds the functional space.
- Optimizes k (number of neighbors) and lambda (penalization parameter).
- Saves optimized parameters and functional representation.

### Step 10 — Distance Matrix Computation
- Computes the full functional distance matrix.
- Saves the matrix for reproducibility.

### Step 11 — Model Evaluation
- Performs stratified cross-validation.
- Computes Accuracy, Precision, Recall, F1-score, AUC, and Log-loss.
- Generates summary report.

### Step 12 — Robustness Analysis
- Learning curve analysis.
- Bootstrap resampling.
- Stability assessment.
- Generates final robustness report.

---

## License

This project is released under the MIT License.
