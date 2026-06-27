# SHAP Explainability Reproduction
### Based on: "A Unified Approach to Interpreting Model Predictions"
**Lundberg & Lee, NIPS 2017** | [Original Paper](https://arxiv.org/abs/1705.07874)

---

## 📌 Objective
This project is a comprehensive reproduction of the SHAP (SHapley Additive 
exPlanations) methodology proposed in the landmark NIPS 2017 paper by Scott 
Lundberg and Su-In Lee. The paper introduced a unified framework for 
interpreting machine learning model predictions by assigning each feature a 
contribution score based on Shapley values from cooperative game theory.

This reproduction systematically verifies the theoretical properties claimed 
in the paper, implements both model-specific (TreeExplainer) and model-agnostic 
(KernelExplainer) SHAP methods, reproduces key experimental results from 
Section 5 of the paper, and demonstrates Deep SHAP on MNIST digit classification.

---

## 📂 Project Structure
SHAP-Explainability-Reproduction/
│
├── data/
│ └── adult_income.csv  # Adult Income dataset (45222 rows, 13 features)
│
├── outputs/
│ ├── summary_plot.png # Global feature importance (beeswarm)
│ ├── bar_plot.png # Mean absolute SHAP values
│ ├── waterfall_plot.png # Single prediction explanation
│ ├── dependence_plot_age.png # SHAP dependence plot for Age
│ ├── tree_vs_kernel_shap.png # Tree SHAP vs Kernel SHAP comparison
│ ├── section51_convergence.png # Figure 3 reproduction - convergence
│ ├── section52_human_studies.png # Figure 4 reproduction - human studies
│ ├── section53_mnist_deepshap.png # Section 5.3 - Deep SHAP on MNIST
│ └── shap_vs_lime_vs_xgb.png # Full method comparison plot
│
├── shap_reproduction.py # Main Python script (10 sections)
├── requirements.txt # Project dependencies
└── README.md # Project documentation


---

## 🧠 Background: What is SHAP?

### The Problem
Modern ML models like XGBoost, Random Forests and Deep Neural Networks 
achieve high accuracy but are essentially black boxes. Understanding why 
a model makes a specific prediction is as crucial as the prediction itself.

### The Solution
SHAP assigns each feature an importance value for a specific prediction, 
rooted in cooperative game theory (Shapley values, 1953). It provides a 
mathematically fair way to distribute credit among features based on their 
contribution to the prediction.

### The Unified Framework
The paper proves that six existing methods (LIME, DeepLIFT, Layer-wise 
Relevance Propagation, Shapley regression, Shapley sampling, and Quantitative 
Input Influence) all belong to the same class of additive feature attribution 
methods:
g(z) = φ₀ + φ₁z₁ + φ₂z₂ + ... + φₘzₘ


### The Three Desirable Properties
SHAP values are the only solution satisfying all three simultaneously:

| Property | Description | Why It Matters |
|---|---|---|
| **Local Accuracy** | base_value + sum(SHAP) = model output | Explanations faithful to model |
| **Missingness** | Absent features get zero attribution | Missing data handled correctly |
| **Consistency** | If feature impact grows, SHAP never decreases | Rankings logically consistent |

---

## 📊 What This Project Reproduces

### Section 1 — Adult Income Dataset
The UCI Adult Income dataset (45,222 samples, 13 features) predicts whether 
a person earns more than $50K/year. Features include age, education, 
occupation, marital status, and more.

### Section 2 — XGBoost Classifier
- 100 estimators, max depth 4, learning rate 0.1
- 80/20 train-test split with stratification
- Final accuracy: **86.53%**

### Section 3 — Tree SHAP (TreeExplainer)
Exact SHAP value computation using TreeExplainer — the most efficient 
method for tree-based models. Computes exact Shapley values in polynomial 
time by exploiting the tree structure.

### Section 4 — Kernel SHAP (KernelExplainer) — Core Paper Contribution
Implements the actual Kernel SHAP method from Section 4.1 of the paper.
Kernel SHAP connects Shapley values with weighted linear regression using 
a specific kernel weighting function:
π(z') = (M-1) / (C(M, |z'|) × |z'| × (M - |z'|))


This is the key theoretical novelty — any model-agnostic explanation 
that satisfies all 3 properties must use this exact kernel.

### Section 5 — Local Accuracy Verification (Paper Property 1)
Mathematically verified for 3 test persons:
base_value + sum(SHAP values) = model output (log-odds)

✅ Local Accuracy holds: True for all persons

### Section 6 — SHAP Visualizations
- **Summary Plot** — global feature importance with direction
- **Bar Plot** — ranked mean absolute SHAP values
- **Waterfall Plot** — step-by-step explanation for one person
- **Dependence Plot** — how Age interacts with income predictions
- **Tree vs Kernel SHAP** — comparison of both methods

### Section 7 — Section 5.1 Reproduction: Computational Efficiency
Reproduces Figure 3 from the paper using synthetic datasets:

**Dense Model (10 features, all informative)**
- Compares Kernel SHAP vs Shapley Sampling vs LIME
- All methods have similar error on dense models

**Sparse Model (100 features, only 3 informative)**
- Kernel SHAP error drops: 0.71 → 0.04 as samples increase
- LIME error stays high (5-9) regardless of sample size
- Key finding: **SHAP converges better on sparse models**

### Section 8 — Section 5.2 Reproduction: Human Allocation Studies
Reproduces Figure 4 from the paper:

**Study 1: Sickness Score Model**
Fever=1, Cough=1 → score = 2
Fever=1, Cough=0 → score = 5
Fever=0, Cough=1 → score = 5
Fever=0, Cough=0 → score = 0

SHAP correctly assigns negative values when both symptoms present.

**Study 2: Max Allocation Problem**
Man 1: 5 pts, Man 2: 4 pts, Man 3: 0 pts → Profit = $5

SHAP rankings match human intuition: Man1 > Man2 > Man3

### Section 9 — MNIST Experiment (Section 5.3): Deep SHAP
Reproduces the deep learning experiment from the paper:
- Built and trained a CNN on MNIST (60,000 training images)
- **CNN Test Accuracy: 98.68%**
- Used `shap.DeepExplainer` to compute pixel-level SHAP values
- Red pixels increase prediction probability, Blue pixels decrease it
- Shows which pixels the model focuses on for each digit class

### Section 10 — Full Method Comparison
Three-way comparison:
- **SHAP** — mathematically principled, satisfies all 3 properties
- **XGBoost built-in** — gain-based, no consistency guarantee
- **LIME** — local linear approximation, can violate local accuracy

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/TanishkaPagar/SHAP-Explainability-Reproduction.git
cd SHAP-Explainability-Reproduction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
pip install tensorflow
```

### 3. Run the script
```bash
python shap_reproduction.py
```

### 4. View outputs
All plots are saved in the `outputs/` folder.

Note: Script takes several minutes due to KernelExplainer,
Section 5.1 experiments, and MNIST CNN training.

---

## 📈 Results Summary

| Metric | Value |
|---|---|
| Dataset | Adult Income (45,222 samples, 13 features) |
| Model | XGBoost Classifier |
| Accuracy | 86.53% |
| Local Accuracy holds | ✅ True (all persons) |
| Kernel SHAP | ✅ KernelExplainer (model-agnostic) |
| Tree SHAP | ✅ TreeExplainer (exact) |
| Section 5.1 | ✅ Dense + Sparse + Shapley Sampling |
| Section 5.2 | ✅ Sickness + Max allocation studies |
| Section 5.3 | ✅ MNIST Deep SHAP (98.68% CNN accuracy) |
| Total output plots | 9 |

### Key Findings
- **Relationship** and **Marital Status** are the most important features
- **Age**, **Education**, and **Hours-per-week** also significantly impact income
- Kernel SHAP and Tree SHAP agree on feature rankings — validates consistency
- SHAP outperforms LIME on sparse models — reproduced from Figure 3
- Deep SHAP correctly identifies digit-specific pixels on MNIST

---

## 🔑 Key Concepts from the Paper

| Concept | Description |
|---|---|
| **Additive Feature Attribution** | Explanation is a linear sum of feature contributions |
| **Shapley Values** | From cooperative game theory — fairly distributes credit |
| **Local Accuracy** | base_value + sum(SHAP) = actual model output |
| **Missingness** | Features absent from input get zero attribution |
| **Consistency** | More important features always get higher SHAP values |
| **TreeExplainer** | Exact, fast SHAP for tree-based models |
| **KernelExplainer** | Model-agnostic SHAP via weighted linear regression |
| **DeepExplainer** | SHAP for deep neural networks via DeepLIFT |
| **Shapley Sampling** | Approximation method compared in Section 5.1 |
| **SHAP vs LIME** | LIME heuristically chooses kernel; SHAP uses provably correct one |

---

## 📚 Reference
Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting
model predictions. Advances in Neural Information Processing Systems, 30.
https://arxiv.org/abs/1705.07874

---

## 👩‍💻 Author
**Tanishka Pagar** | Data Science Intern, LABTECH
Savitribai Phule Pune University | B.Sc. Data Science (2023–2027)






