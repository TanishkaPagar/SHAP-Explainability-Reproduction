# ============================================================
# SHAP Explainability Reproduction
# Based on: "A Unified Approach to Interpreting Model Predictions"
# Lundberg & Lee, NIPS 2017
# Dataset: Titanic (Classification)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
import xgboost
import xgboost
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from lime.lime_tabular import LimeTabularExplainer

# ============================================================
# SECTION 1: LOAD & PREPROCESS DATA
# ============================================================

print("=" * 60)
print("SECTION 1: Loading and Preprocessing Titanic Dataset")
print("=" * 60)

df = pd.read_csv('data/titanic.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# Drop columns not useful for prediction
df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=True)

# Fill missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# Encode categorical columns
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])        # male=1, female=0
df['Embarked'] = le.fit_transform(df['Embarked'])  # C=0, Q=1, S=2

print(f"\nPreprocessed shape: {df.shape}")
print(f"Class distribution:\n{df['Survived'].value_counts()}")

# ============================================================
# SECTION 2: TRAIN XGBOOST MODEL
# ============================================================

print("\n" + "=" * 60)
print("SECTION 2: Training XGBoost Classifier")
print("=" * 60)

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

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
# SECTION 3: SHAP VALUES (TreeExplainer)
# ============================================================

print("\n" + "=" * 60)
print("SECTION 3: Computing SHAP Values (TreeExplainer)")
print("=" * 60)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

print(f"SHAP values shape: {shap_values.shape}")
print(f"Base value (E[f(x)]): {explainer.expected_value:.4f}")
print(f"\nSHAP values for first passenger:\n")
for feat, val in zip(X_test.columns, shap_values[0]):
    print(f"  {feat:12s}: {val:+.4f}")

# ============================================================
# SECTION 4: VERIFY LOCAL ACCURACY PROPERTY (FROM PAPER)
# ============================================================

print("\n" + "=" * 60)
print("SECTION 4: Verifying Local Accuracy Property")
print("(Paper Property 1: base_value + sum(SHAP) = log-odds output)")
print("=" * 60)

for i in range(3):
    base_val = explainer.expected_value
    shap_sum = shap_values[i].sum()
    shap_prediction = base_val + shap_sum
    # XGBoost TreeExplainer works in log-odds space
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
# SECTION 5: VISUALIZATIONS
# ============================================================

print("\n" + "=" * 60)
print("SECTION 5: Generating SHAP Visualizations")
print("=" * 60)

# --- Plot 1: Summary Plot (Global Feature Importance) ---
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP Summary Plot - Global Feature Importance")
plt.tight_layout()
plt.savefig('outputs/summary_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/summary_plot.png")

# --- Plot 2: Bar Plot (Mean Absolute SHAP) ---
plt.figure()
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title("SHAP Feature Importance (Mean |SHAP|)")
plt.tight_layout()
plt.savefig('outputs/bar_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/bar_plot.png")

# --- Plot 3: Waterfall Plot (Single Prediction) ---
shap_explanation = shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
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

# --- Plot 4: SHAP Dependence Plot ---
plt.figure()
shap.dependence_plot('Age', shap_values, X_test, show=False)
plt.title("SHAP Dependence Plot - Age")
plt.tight_layout()
plt.savefig('outputs/dependence_plot_age.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/dependence_plot_age.png")

# ============================================================
# SECTION 6: SHAP vs LIME vs XGBoost Feature Importance
# ============================================================

print("\n" + "=" * 60)
print("SECTION 6: Comparing SHAP vs LIME vs Built-in Importance")
print("=" * 60)

# --- SHAP importance (mean absolute SHAP values) ---
shap_importance = pd.Series(
    np.abs(shap_values).mean(axis=0),
    index=X_test.columns
).sort_values(ascending=False)

# --- XGBoost built-in feature importance ---
xgb_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

# --- LIME importance (average over 20 test samples) ---
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

# --- Comparison Plot ---
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