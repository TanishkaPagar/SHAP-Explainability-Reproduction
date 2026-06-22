# SHAP Explainability Reproduction
### Based on: "A Unified Approach to Interpreting Model Predictions"
**Lundberg & Lee, NIPS 2017** | [Original Paper](https://arxiv.org/abs/1705.07874)

---

## 📌 Objective
This project reproduces the SHAP (SHapley Additive exPlanations) methodology
proposed in the original NIPS 2017 paper. SHAP values are applied to the
classic Titanic dataset to explain individual predictions of an XGBoost
classifier.

---

## 📂 Project Structure

```
SHAP-Explainability-Reproduction/
│
├── data/
│   └── titanic.csv               # Titanic dataset (891 rows, 12 columns)
│
├── outputs/
│   ├── summary_plot.png          # Global feature importance (beeswarm)
│   ├── bar_plot.png              # Mean absolute SHAP values
│   ├── waterfall_plot.png        # Single prediction explanation
│   ├── dependence_plot_age.png   # SHAP dependence plot for Age
│   └── shap_vs_lime_vs_xgb.png  # Method comparison plot
│
├── shap_reproduction.py          # Main Python script
├── requirements.txt              # Dependencies
└── README.md                     # Project documentation
```

## 🧠 What is SHAP?
SHAP assigns each feature an importance value for a specific prediction,
rooted in cooperative game theory (Shapley values). It is the **only**
additive feature attribution method satisfying three key properties:

| Property | Description |
|---|---|
| **Local Accuracy** | SHAP values + base value = actual model output |
| **Missingness** | Absent features get zero attribution |
| **Consistency** | If a feature's impact grows, its SHAP value never decreases |

---

## 📊 What This Project Reproduces

### 1. SHAP Value Computation
Using TreeExplainer on a trained XGBoost classifier.

### 2. Local Accuracy Verification (Paper Property 1)
Mathematically verified that:
base_value + sum(SHAP values) = model output (log-odds)
✅ Confirmed True for all test passengers.

### 3. Visualizations
- **Summary Plot** — which features matter most globally and in which direction
- **Bar Plot** — ranked mean absolute SHAP values
- **Waterfall Plot** — step-by-step explanation for one passenger
- **Dependence Plot** — how Age interacts with survival predictions

### 4. Method Comparison (Paper Section 5)
- SHAP (Shapley-based, mathematically principled)
- XGBoost built-in importance (gain-based, no consistency guarantee)
- LIME (local linear approximation, can violate local accuracy)

---

## 🚀 How to Run

### 1. Clone the repository
git clone https://github.com/TanishkaPagar/SHAP-Explainability-Reproduction.git
cd SHAP-Explainability-Reproduction

### 2. Install dependencies
pip install -r requirements.txt

### 3. Run the script
python shap_reproduction.py

### 4. View outputs
All plots are saved in the outputs/ folder.

---

## 📈 Results

| Metric | Value |
|---|---|
| Model | XGBoost Classifier |
| Dataset | Titanic (891 samples, 7 features) |
| Accuracy | 79.89% |
| Local Accuracy holds | ✅ True (all passengers) |

### Key Finding
Sex, Pclass and Age are the most influential features — consistent
with domain knowledge that women, children and first-class passengers
had higher survival rates.

---

## 🔑 Key Concepts from the Paper

- **Additive Feature Attribution** — explanation is a linear sum of feature contributions
- **Shapley Values** — from cooperative game theory, fairly distributes credit among features
- **TreeExplainer** — exact, efficient SHAP computation for tree-based models
- **Kernel SHAP** — model-agnostic approximation using weighted linear regression
- **SHAP vs LIME** — LIME violates local accuracy; SHAP is mathematically guaranteed

---

## 📚 Reference
Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting
model predictions. Advances in Neural Information Processing Systems, 30.

---

## 👩‍💻 Author
**Tanishka Pagar** | Data Science Intern, LABTECH
Savitribai Phule Pune University | B.Sc. Data Science (2023–2027)