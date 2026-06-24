# ============================================================
# SHAP Explainability Reproduction
# Based on: "A Unified Approach to Interpreting Model Predictions"
# Lundberg & Lee, NIPS 2017
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings('ignore')

import xgboost
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_regression
from lime.lime_tabular import LimeTabularExplainer
import time

print("All libraries imported successfully!")

# ============================================================
# SECTION 1: LOAD & PREPROCESS TITANIC DATA
# ============================================================

print("\n" + "=" * 60)
print("SECTION 1: Loading and Preprocessing Titanic Dataset")
print("=" * 60)

df = pd.read_csv('data/titanic.csv')
print(f"Dataset shape: {df.shape}")

df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

print(f"Preprocessed shape: {df.shape}")
print(f"Class distribution:\n{df['Survived'].value_counts()}")

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# SECTION 2: TRAIN XGBOOST MODEL
# ============================================================

print("\n" + "=" * 60)
print("SECTION 2: Training XGBoost Classifier")
print("=" * 60)

model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc:.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

# ============================================================
# SECTION 3: TREE SHAP (TreeExplainer)
# ============================================================

print("\n" + "=" * 60)
print("SECTION 3: Tree SHAP using TreeExplainer")
print("=" * 60)

tree_explainer = shap.TreeExplainer(model)
tree_shap_values = tree_explainer.shap_values(X_test)

print(f"Tree SHAP values shape: {tree_shap_values.shape}")
print(f"Base value: {tree_explainer.expected_value:.4f}")
print(f"\nTree SHAP values for first passenger:")
for feat, val in zip(X_test.columns, tree_shap_values[0]):
    print(f"  {feat:12s}: {val:+.4f}")

# ============================================================
# SECTION 4: KERNEL SHAP (KernelExplainer) - PROPER IMPLEMENTATION
# ============================================================

print("\n" + "=" * 60)
print("SECTION 4: Kernel SHAP using KernelExplainer")
print("(Model-agnostic approximation - Linear LIME + Shapley values)")
print("=" * 60)

# Use a small background dataset for efficiency
background = shap.sample(X_train, 50)

# Wrap predict_proba to avoid XGBoost feature_names issue
def model_predict(X):
    return model.predict_proba(X)

# KernelExplainer works with predict_proba for classifiers
kernel_explainer = shap.KernelExplainer(model_predict, background.values)

# Explain a small subset of test data (KernelSHAP is slower)
X_test_small = X_test.iloc[:20]
kernel_shap_values = kernel_explainer.shap_values(X_test_small.values, nsamples=100)

# kernel_shap_values is a list [class_0, class_1] - we take class_1 (survived)
# kernel_shap_values shape: [n_classes, n_samples, n_features]
kernel_arr_s4 = np.array(kernel_shap_values)
kernel_shap_class1 = kernel_arr_s4[:, :, 1]
print(f"Kernel SHAP class1 shape: {kernel_shap_class1.shape}")
print(f"kernel_shap_values type: {type(kernel_shap_values)}")
print(f"kernel_shap_values length: {len(kernel_shap_values)}")
print(f"kernel_shap_values[0] shape: {np.array(kernel_shap_values[0]).shape}")
print(f"Kernel SHAP values shape: {kernel_shap_class1.shape}")
print(f"Base value (class 1): {kernel_explainer.expected_value[1]:.4f}")
print(f"\nKernel SHAP values for first passenger:")
for feat, val in zip(X_test.columns, kernel_shap_class1[0]):
    print(f"  {feat:12s}: {val:+.4f}")

# ============================================================
# SECTION 5: VERIFY LOCAL ACCURACY PROPERTY (FROM PAPER)
# ============================================================

print("\n" + "=" * 60)
print("SECTION 5: Verifying Local Accuracy Property")
print("(Paper Property 1: base_value + sum(SHAP) = model output)")
print("=" * 60)

for i in range(3):
    base_val = tree_explainer.expected_value
    shap_sum = tree_shap_values[i].sum()
    shap_prediction = base_val + shap_sum
    actual_log_odds = model.get_booster().predict(
        xgboost.DMatrix(X_test.iloc[[i]]), output_margin=True
    )[0]

    print(f"\nPassenger {i+1}:")
    print(f"  Base value              : {base_val:.4f}")
    print(f"  Sum of SHAP values      : {shap_sum:.4f}")
    print(f"  Base + SHAP (reproduced): {shap_prediction:.4f}")
    print(f"  Actual log-odds output  : {actual_log_odds:.4f}")
    print(f"  Local Accuracy holds    : {abs(shap_prediction - actual_log_odds) < 0.01}")

# ============================================================
# SECTION 6: VISUALIZATIONS
# ============================================================

print("\n" + "=" * 60)
print("SECTION 6: Generating SHAP Visualizations")
print("=" * 60)

# --- Plot 1: Summary Plot ---
plt.figure()
shap.summary_plot(tree_shap_values, X_test, show=False)
plt.title("SHAP Summary Plot - Global Feature Importance")
plt.tight_layout()
plt.savefig('outputs/summary_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/summary_plot.png")

# --- Plot 2: Bar Plot ---
plt.figure()
shap.summary_plot(tree_shap_values, X_test, plot_type="bar", show=False)
plt.title("SHAP Feature Importance (Mean |SHAP|)")
plt.tight_layout()
plt.savefig('outputs/bar_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/bar_plot.png")

# --- Plot 3: Waterfall Plot ---
shap_explanation = shap.Explanation(
    values=tree_shap_values[0],
    base_values=tree_explainer.expected_value,
    data=X_test.iloc[0].values,
    feature_names=X_test.columns.tolist()
)
plt.figure()
shap.plots.waterfall(shap_explanation, show=False)
plt.title("SHAP Waterfall Plot - Passenger 1")
plt.tight_layout()
plt.savefig('outputs/waterfall_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/waterfall_plot.png")

# --- Plot 4: Dependence Plot ---
plt.figure()
shap.dependence_plot('Age', tree_shap_values, X_test, show=False)
plt.title("SHAP Dependence Plot - Age")
plt.tight_layout()
plt.savefig('outputs/dependence_plot_age.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/dependence_plot_age.png")

# --- Plot 5: Kernel SHAP vs Tree SHAP comparison ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

tree_importance = pd.Series(
    np.abs(tree_shap_values).mean(axis=0),
    index=X_test.columns
).sort_values(ascending=False)

# Fix kernel SHAP shape issue
kernel_arr = np.array(kernel_shap_values)
print(f"kernel_arr shape: {kernel_arr.shape}")
# Shape is (20, 7, 2) - samples, features, classes
# Take class 1 (survived) from last dimension
kernel_shap_fixed = kernel_arr[:, :, 1]
print(f"kernel_shap_fixed shape: {kernel_shap_fixed.shape}")

kernel_importance = pd.Series(
    np.abs(kernel_shap_fixed).mean(axis=0),
    index=X_test.columns.tolist()
).sort_values(ascending=False)

tree_importance.plot(kind='barh', ax=axes[0], color='steelblue')
axes[0].set_title('Tree SHAP\n(TreeExplainer)', fontsize=12)
axes[0].set_xlabel('Mean |SHAP value|')
axes[0].invert_yaxis()

kernel_importance.plot(kind='barh', ax=axes[1], color='darkorange')
axes[1].set_title('Kernel SHAP\n(KernelExplainer)', fontsize=12)
axes[1].set_xlabel('Mean |SHAP value|')
axes[1].invert_yaxis()

plt.suptitle('Tree SHAP vs Kernel SHAP Feature Importance',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/tree_vs_kernel_shap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/tree_vs_kernel_shap.png")    

# ============================================================
# SECTION 7: SECTION 5.1 FROM PAPER
# Computational Efficiency - Synthetic Dense & Sparse Experiments
# ============================================================

print("\n" + "=" * 60)
print("SECTION 7: Section 5.1 - Synthetic Experiments")
print("(Reproducing Figure 3 from the paper)")
print("=" * 60)

def get_kernel_shap_estimate(model, X_train, X_test_single, nsamples):
    background = shap.sample(pd.DataFrame(X_train), 50)
    explainer = shap.KernelExplainer(model.predict, background)
    shap_vals = explainer.shap_values(
        pd.DataFrame(X_test_single.reshape(1, -1)),
        nsamples=nsamples,
        silent=True
    )
    return np.array(shap_vals)[0]

def get_lime_estimate(model, X_train, X_test_single, n_samples):
    explainer = LimeTabularExplainer(
        X_train,
        mode='regression',
        feature_names=[f'f{i}' for i in range(X_train.shape[1])]
    )
    exp = explainer.explain_instance(
        X_test_single,
        model.predict,
        num_features=X_train.shape[1],
        num_samples=n_samples
    )
    vals = dict(exp.as_list())
    result = np.zeros(X_train.shape[1])
    for k, v in vals.items():
        for i in range(X_train.shape[1]):
            if f'f{i}' in k:
                result[i] = v
    return result

# --- Dense Model (10 features, all matter) ---
print("\nRunning Dense Model experiment (10 features)...")
np.random.seed(42)
X_dense, y_dense = make_regression(n_samples=1000, n_features=10,
                                    n_informative=10, noise=0.1, random_state=42)
X_dense_train, X_dense_test = X_dense[:800], X_dense[800:]
y_dense_train = y_dense[:800]

dense_model = DecisionTreeRegressor(max_depth=6, random_state=42)
dense_model.fit(X_dense_train, y_dense_train)

# True SHAP values using full KernelExplainer as reference
background_dense = shap.sample(pd.DataFrame(X_dense_train), 100)
true_explainer_dense = shap.KernelExplainer(
    dense_model.predict, background_dense
)
true_shap_dense = true_explainer_dense.shap_values(
    X_dense_test[0:1], nsamples=1000, silent=True
)[0]

sample_sizes = [50, 100, 200, 300, 500]
shap_errors_dense = []
lime_errors_dense = []

for n in sample_sizes:
    shap_est = get_kernel_shap_estimate(
        dense_model, X_dense_train, X_dense_test[0], n
    )
    lime_est = get_lime_estimate(
        dense_model, X_dense_train, X_dense_test[0], n
    )
    shap_errors_dense.append(np.mean(np.abs(shap_est - true_shap_dense)))
    lime_errors_dense.append(np.mean(np.abs(lime_est - true_shap_dense)))
    print(f"  n={n}: SHAP error={shap_errors_dense[-1]:.4f}, "
          f"LIME error={lime_errors_dense[-1]:.4f}")

# --- Sparse Model (100 features, only 3 matter) ---
print("\nRunning Sparse Model experiment (100 features, 3 informative)...")
X_sparse, y_sparse = make_regression(n_samples=1000, n_features=100,
                                      n_informative=3, noise=0.1, random_state=42)
X_sparse_train, X_sparse_test = X_sparse[:800], X_sparse[800:]
y_sparse_train = y_sparse[:800]

sparse_model = DecisionTreeRegressor(max_depth=6, random_state=42)
sparse_model.fit(X_sparse_train, y_sparse_train)

background_sparse = shap.sample(pd.DataFrame(X_sparse_train), 100)
true_explainer_sparse = shap.KernelExplainer(
    sparse_model.predict, background_sparse
)
true_shap_sparse = true_explainer_sparse.shap_values(
    X_sparse_test[0:1], nsamples=1000, silent=True
)[0]

shap_errors_sparse = []
lime_errors_sparse = []

for n in sample_sizes:
    shap_est = get_kernel_shap_estimate(
        sparse_model, X_sparse_train, X_sparse_test[0], n
    )
    lime_est = get_lime_estimate(
        sparse_model, X_sparse_train, X_sparse_test[0], n
    )
    shap_errors_sparse.append(np.mean(np.abs(shap_est - true_shap_sparse)))
    lime_errors_sparse.append(np.mean(np.abs(lime_est - true_shap_sparse)))
    print(f"  n={n}: SHAP error={shap_errors_sparse[-1]:.4f}, "
          f"LIME error={lime_errors_sparse[-1]:.4f}")

# --- Plot Section 5.1 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(sample_sizes, shap_errors_dense, 'b-o', label='Kernel SHAP')
axes[0].plot(sample_sizes, lime_errors_dense, 'g-o', label='LIME')
axes[0].set_title('(A) Dense Model\n(10 features, all informative)', fontsize=12)
axes[0].set_xlabel('Number of model evaluations')
axes[0].set_ylabel('Mean absolute error from true SHAP')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(sample_sizes, shap_errors_sparse, 'b-o', label='Kernel SHAP')
axes[1].plot(sample_sizes, lime_errors_sparse, 'g-o', label='LIME')
axes[1].set_title('(B) Sparse Model\n(100 features, 3 informative)', fontsize=12)
axes[1].set_xlabel('Number of model evaluations')
axes[1].set_ylabel('Mean absolute error from true SHAP')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Section 5.1: Kernel SHAP vs LIME Convergence\n'
             '(Reproduction of Figure 3 from Lundberg & Lee, NIPS 2017)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/section51_convergence.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: outputs/section51_convergence.png")

# ============================================================
# SECTION 8: SECTION 5.2 FROM PAPER
# Human Allocation Studies
# ============================================================

print("\n" + "=" * 60)
print("SECTION 8: Section 5.2 - Human Allocation Studies")
print("(Sickness Score + Max Allocation Problem)")
print("=" * 60)

# --- Study 1: Sickness Score ---
print("\nStudy 1: Sickness Score Model")
print("Model: output=5 if exactly one symptom, 2 if both, 0 if none")

def sickness_model(X):
    results = []
    for row in X:
        fever, cough = row[0], row[1]
        if fever == 1 and cough == 1:
            results.append(2)
        elif fever == 1 or cough == 1:
            results.append(5)
        else:
            results.append(0)
    return np.array(results)

# Patient has both fever and cough
patient = np.array([[1, 1, 0]])  # fever=1, cough=1, congestion=0
background_sickness = np.array([
    [0, 0, 0], [1, 0, 0], [0, 1, 0],
    [1, 1, 0], [0, 0, 1], [1, 0, 1]
])

sick_explainer = shap.KernelExplainer(sickness_model, background_sickness)
sick_shap = sick_explainer.shap_values(patient, nsamples=512, silent=True)

feature_names_sick = ['Fever', 'Cough', 'Congestion']
human_values_sick = [0.5, 0.5, 0.0]  # Human intuition from paper

print(f"\nPatient: Fever=1, Cough=1, Congestion=0")
print(f"Model output: {sickness_model(patient)[0]}")
print(f"\nFeature attributions:")
print(f"{'Feature':<12} {'SHAP':>8} {'Human':>8}")
print("-" * 30)
for f, s, h in zip(feature_names_sick, sick_shap[0], human_values_sick):
    print(f"{f:<12} {s:>8.3f} {h:>8.3f}")

# --- Study 2: Max Allocation Problem ---
print("\n\nStudy 2: Max Allocation Problem")
print("Model: profit = max(score) among 3 men")
print("Man 1=5, Man 2=4, Man 3=0 → Profit=$5")

def max_model(X):
    return np.max(X, axis=1).astype(float)

players = np.array([[5, 4, 0]])
# Larger background for better SHAP estimation
np.random.seed(42)
background_max = np.random.randint(0, 6, size=(50, 3)).astype(float)

max_explainer = shap.KernelExplainer(max_model, background_max)
max_shap = max_explainer.shap_values(players, nsamples=512, silent=True)

feature_names_max = ['Man 1 (5pts)', 'Man 2 (4pts)', 'Man 3 (0pts)']
human_values_max = [3.0, 1.5, 0.0]  # Approximate human intuition from paper

print(f"\nPlayers: Man1=5, Man2=4, Man3=0 → Profit=$5")
print(f"\nFeature attributions:")
print(f"{'Player':<14} {'SHAP':>8} {'Human':>8}")
print("-" * 34)
for f, s, h in zip(feature_names_max, max_shap[0], human_values_max):
    print(f"{f:<14} {s:>8.3f} {h:>8.3f}")

# --- Plot Section 5.2 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x1 = np.arange(len(feature_names_sick))
w = 0.35
axes[0].bar(x1 - w/2, sick_shap[0], w, label='SHAP', color='steelblue')
axes[0].bar(x1 + w/2, human_values_sick, w, label='Human', color='gray')
axes[0].set_xticks(x1)
axes[0].set_xticklabels(feature_names_sick)
axes[0].set_title('(A) Sickness Score\n(Fever=1, Cough=1, Congestion=0)', fontsize=11)
axes[0].set_ylabel('Feature Attribution')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=0, color='black', linewidth=0.8)

x2 = np.arange(len(feature_names_max))
axes[1].bar(x2 - w/2, max_shap[0], w, label='SHAP', color='steelblue')
axes[1].bar(x2 + w/2, human_values_max, w, label='Human', color='gray')
axes[1].set_xticks(x2)
axes[1].set_xticklabels(feature_names_max)
axes[1].set_title('(B) Max Allocation\n(Man1=5, Man2=4, Man3=0)', fontsize=11)
axes[1].set_ylabel('Feature Attribution')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Section 5.2: SHAP vs Human Intuition\n'
             '(Reproduction of Figure 4 from Lundberg & Lee, NIPS 2017)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/section52_human_studies.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: outputs/section52_human_studies.png")

# ============================================================
# SECTION 9: SHAP vs LIME vs XGBoost Comparison
# ============================================================

print("\n" + "=" * 60)
print("SECTION 9: SHAP vs LIME vs XGBoost Feature Importance")
print("=" * 60)

shap_importance = pd.Series(
    np.abs(tree_shap_values).mean(axis=0),
    index=X_test.columns
).sort_values(ascending=False)

xgb_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

lime_explainer = LimeTabularExplainer(
    X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=['Not Survived', 'Survived'],
    mode='classification'
)

lime_importances = np.zeros(X_test.shape[1])
for i in range(20):
    exp = lime_explainer.explain_instance(
        X_test.iloc[i].values,
        model.predict_proba,
        num_features=X_test.shape[1]
    )
    for feat, val in exp.as_list():
        for j, col in enumerate(X_test.columns):
            if col in feat:
                lime_importances[j] += abs(val)

lime_importance = pd.Series(
    lime_importances / 20,
    index=X_test.columns
).sort_values(ascending=False)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

shap_importance.plot(kind='barh', ax=axes[0], color='steelblue')
axes[0].set_title('SHAP Importance\n(Mean |SHAP Value|)', fontsize=12)
axes[0].set_xlabel('Importance')
axes[0].invert_yaxis()

xgb_importance.plot(kind='barh', ax=axes[1], color='coral')
axes[1].set_title('XGBoost Built-in\nFeature Importance', fontsize=12)
axes[1].set_xlabel('Importance')
axes[1].invert_yaxis()

lime_importance.plot(kind='barh', ax=axes[2], color='green')
axes[2].set_title('LIME Importance\n(Avg over 20 samples)', fontsize=12)
axes[2].set_xlabel('Importance')
axes[2].invert_yaxis()

plt.suptitle('Comparison: SHAP vs XGBoost vs LIME Feature Importance',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/shap_vs_lime_vs_xgb.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/shap_vs_lime_vs_xgb.png")

# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 60)
print("ALL SECTIONS COMPLETE!")
print("Check the outputs/ folder for all saved plots.")
print("=" * 60)