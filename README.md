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

This reproduction goes beyond a simple implementation — it systematically 
verifies the theoretical properties claimed in the paper, implements both 
model-specific (TreeExplainer) and model-agnostic (KernelExplainer) SHAP 
methods, and reproduces key experimental results from Section 5 of the paper 
including computational efficiency comparisons (Figure 3) and human allocation 
studies (Figure 4).

The Titanic survival prediction dataset is used as the primary application, 
with synthetic datasets used for Section 5.1 experiments — exactly as in the 
original paper.

---

## 📂 Project Structure
SHAP-Explainability-Reproduction/
│
├── data/
│ └── titanic.csv # Titanic dataset (891 rows, 12 columns)
│
├── outputs/
│ ├── summary_plot.png # Global feature importance (beeswarm)
│ ├── bar_plot.png # Mean absolute SHAP values
│ ├── waterfall_plot.png # Single prediction explanation
│ ├── dependence_plot_age.png # SHAP dependence plot for Age
│ ├── tree_vs_kernel_shap.png # Tree SHAP vs Kernel SHAP comparison
│ ├── section51_convergence.png # Figure 3 reproduction - convergence
│ ├── section52_human_studies.png # Figure 4 reproduction - human studies
│ └── shap_vs_lime_vs_xgb.png # Full method comparison plot
│
├── shap_reproduction.py # Main Python script (9 sections)
├── requirements.txt # Project dependencies
└── README.md # Project documentation


---

## 🧠 Background: What is SHAP?

### The Problem
Modern machine learning models like XGBoost, Random Forests, and Deep Neural 
Networks achieve high accuracy but are essentially black boxes — it is very 
difficult to understand why they make a specific prediction. This creates a 
tension between accuracy and interpretability.

### The Solution
SHAP (SHapley Additive exPlanations) solves this by assigning each feature 
an importance value for a specific prediction. The key insight is borrowed 
from cooperative game theory — specifically Shapley values (Shapley, 1953) — 
which provide a mathematically fair way to distribute "credit" among features 
based on their contribution to the prediction.

### The Unified Framework
The paper's main contribution is showing that six existing explanation methods 
(LIME, DeepLIFT, Layer-wise Relevance Propagation, Shapley regression values, 
Shapley sampling values, and Quantitative Input Influence) all belong to the 
same class of **additive feature attribution methods**:
g(z) = φ₀ + φ₁z₁ + φ₂z₂ + ... + φₘzₘ


Where φᵢ is the SHAP value (contribution) of feature i.

### The Three Desirable Properties
The paper proves that SHAP values are the **only** solution satisfying all 
three properties simultaneously:

| Property | Description | Why It Matters |
|---|---|---|
| **Local Accuracy** | base_value + sum(SHAP) = model output | Explanations are faithful to the model |
| **Missingness** | Absent features get zero attribution | Missing data is handled correctly |
| **Consistency** | If a feature's impact grows, its SHAP never decreases | Rankings are logically consistent |

---

## 📊 What This Project Reproduces

### Section 1 — Data Loading and Preprocessing
The Titanic dataset (891 passengers, 12 features) is loaded and preprocessed:
- Dropped irrelevant columns: Name, Ticket, Cabin, PassengerId
- Imputed missing Age values with median
- Imputed missing Embarked values with mode
- Label encoded Sex and Embarked columns
- Final feature set: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked

### Section 2 — XGBoost Classifier
An XGBoost classifier is trained with the following configuration:
- 100 estimators, max depth 4, learning rate 0.1
- 80/20 train-test split with stratification
- Final accuracy: **79.89%**

### Section 3 — Tree SHAP (TreeExplainer)
Exact SHAP value computation using TreeExplainer — the most efficient 
method for tree-based models. TreeExplainer computes exact Shapley values 
in polynomial time by exploiting the tree structure, unlike the exponential 
time required for brute-force computation.

Key output:
- SHAP values for all 179 test passengers
- Base value (E[f(x)]): -0.4822
- Feature-level attribution for every prediction

### Section 4 — Kernel SHAP (KernelExplainer) — Core Paper Contribution
This is the most important section — implementing the actual Kernel SHAP 
method proposed in the paper (Section 4.1).

**What is Kernel SHAP?**
Kernel SHAP connects Shapley values from game theory with weighted linear 
regression (LIME). The paper proves that by choosing a specific kernel 
weighting function:
π(z') = (M-1) / (C(M, |z'|) × |z'| × (M - |z'|))


The solution to the weighted linear regression problem recovers exact 
Shapley values. This is the key theoretical novelty of the paper.

**Why this matters:**
- TreeExplainer only works for tree-based models
- KernelExplainer works for ANY model (model-agnostic)
- It is slower but universally applicable
- The paper shows it converges faster than naive Shapley sampling

Implementation uses `shap.KernelExplainer` with:
- 50 background samples for efficiency
- predict_proba wrapper to handle XGBoost compatibility
- 100 samples per explanation

### Section 5 — Local Accuracy Verification (Paper Property 1)
Mathematically verified for 3 test passengers:
base_value + sum(SHAP values) = model output (log-odds)

✅ Local Accuracy holds: True for all passengers

This directly reproduces Theorem 1 from the paper.

### Section 6 — SHAP Visualizations
Five plots generated to visualize SHAP results:

**Summary Plot (Beeswarm)**
Shows global feature importance with direction — each dot is one passenger.
Red = high feature value, Blue = low feature value.
Horizontal position = SHAP value (impact on prediction).

**Bar Plot**
Ranked mean absolute SHAP values across all test passengers.
Shows which features matter most on average.

**Waterfall Plot**
Step-by-step explanation for a single passenger showing exactly how 
each feature pushed the prediction from the base value to the final output.

**Dependence Plot**
Shows how Age interacts with survival predictions — young children 
have high positive SHAP values (prioritized for rescue) while older 
passengers have negative SHAP values.

**Tree vs Kernel SHAP Comparison**
Side-by-side comparison of feature importance from TreeExplainer 
and KernelExplainer — both methods agree on feature rankings, 
validating the consistency property.

### Section 7 — Section 5.1 Reproduction: Computational Efficiency
Reproduces Figure 3 from the paper using synthetic datasets:

**Dense Model (10 features, all informative)**
- 1000 samples, DecisionTreeRegressor, max_depth=6
- Compares Kernel SHAP vs LIME at sample sizes: 50, 100, 200, 300, 500
- Result: Both methods have similar error on dense models

**Sparse Model (100 features, only 3 informative)**
- 1000 samples, DecisionTreeRegressor, max_depth=6
- Result: Kernel SHAP error drops from 0.22 → 0.05 as samples increase
- LIME error stays high (5-9) regardless of sample size
- This reproduces the paper's key finding: SHAP converges better on sparse models

### Section 8 — Section 5.2 Reproduction: Human Allocation Studies
Reproduces Figure 4 from the paper — two simple studies where SHAP 
is compared against human intuition.

**Study 1: Sickness Score Model**
Model output:
Fever=1, Cough=1 → score = 2 (both present)
Fever=1, Cough=0 → score = 5 (only one present)
Fever=0, Cough=1 → score = 5 (only one present)
Fever=0, Cough=0 → score = 0 (none present)

For a patient with both Fever and Cough:
- SHAP correctly assigns negative values to both symptoms
- This matches the paper's finding that SHAP handles XOR-like 
  interactions better than LIME or DeepLIFT

**Study 2: Max Allocation Problem**
Three men share profit based on maximum score:
Man 1: 5 questions correct
Man 2: 4 questions correct
Man 3: 0 questions correct
Profit = max(5, 4, 0) = $5

SHAP correctly assigns highest credit to Man 1, less to Man 2, 
and near zero to Man 3 — matching human intuition about fair 
credit allocation.

### Section 9 — Full Method Comparison
Three-way comparison of feature importance methods:
- **SHAP** — mathematically principled, satisfies all 3 properties
- **XGBoost built-in** — gain-based, fast but no consistency guarantee
- **LIME** — local linear approximation, can violate local accuracy

All three agree that Sex, Pclass and Fare are top features, but 
SHAP provides more nuanced and theoretically guaranteed rankings.

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
```

### 3. Run the script
```bash
python shap_reproduction.py
```

### 4. View outputs
All plots are saved in the outputs/ folder.

Note: The script takes a few minutes to run due to KernelExplainer 
and Section 5.1 synthetic experiments.

---

## 📈 Results Summary

| Metric | Value |
|---|---|
| Model | XGBoost Classifier |
| Dataset | Titanic (891 samples, 7 features) |
| Accuracy | 79.89% |
| Local Accuracy holds | ✅ True (all passengers) |
| Kernel SHAP implemented | ✅ KernelExplainer (model-agnostic) |
| Tree SHAP implemented | ✅ TreeExplainer (exact) |
| Section 5.1 reproduced | ✅ Dense + Sparse convergence experiments |
| Section 5.2 reproduced | ✅ Sickness score + Max allocation studies |
| Total output plots | 8 |

### Key Findings
- **Sex** is the most important feature by far (SHAP value ~1.3)
  — being female strongly increases survival probability
- **Pclass** is second most important — first class passengers 
  survived at much higher rates
- **Age** shows an interesting pattern — very young children have 
  high positive SHAP values (evacuation priority) while older 
  passengers have negative values
- **Kernel SHAP and Tree SHAP agree on feature rankings** — 
  validating the consistency property from the paper
- **SHAP outperforms LIME on sparse models** — reproduced from 
  Figure 3 of the paper

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
| **SHAP vs LIME** | LIME heuristically chooses kernel; SHAP uses the provably correct one |
| **SHAP vs DeepLIFT** | DeepLIFT violates consistency; SHAP is guaranteed |

---

## 📚 Reference
Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting
model predictions. Advances in Neural Information Processing Systems, 30.
https://arxiv.org/abs/1705.07874

---

## 👩‍💻 Author
**Tanishka Pagar** | Data Science Intern, LABTECH
Savitribai Phule Pune University | B.Sc. Data Science (2023–2027)




