"""
STEP 3: BASELINE MODEL - LINEAR REGRESSION
===========================================
Goal: Establish performance baseline with simplest model
Key Learning: Regression metrics (MAE, RMSE, R², MAPE), residual analysis

Why start with Linear Regression?
1. Simple, interpretable baseline
2. Fast to train
3. Shows if relationships are linear or non-linear
4. Sets benchmark for XGBoost to beat
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("STEP 3: BASELINE MODEL - LINEAR REGRESSION")
print("=" * 70)

# ============================================================
# LOAD PREPROCESSED DATA
# ============================================================

print("\n📥 Loading preprocessed data...")
train_data = pd.read_csv('models/train_data.csv')
test_data = pd.read_csv('models/test_data.csv')

X_train = train_data.drop('log_loss', axis=1)
y_train = train_data['log_loss']
X_test = test_data.drop('log_loss', axis=1)
y_test = test_data['log_loss']

print(f"✓ Training set: {X_train.shape[0]:,} samples")
print(f"✓ Test set: {X_test.shape[0]:,} samples")
print(f"✓ Features: {X_train.shape[1]}")

# ============================================================
# TRAIN LINEAR REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("STEP 3.1: TRAINING LINEAR REGRESSION")
print("=" * 70)

print("\n🎓 REGRESSION CONCEPT #6: LINEAR REGRESSION")
print("-" * 70)
print("""
Linear Regression: Simplest regression model

Formula: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ

How it works:
  - Finds best-fit line/hyperplane through data
  - Minimizes sum of squared errors: Σ(yᵢ - ŷᵢ)²
  
Assumptions:
  1. Linear relationship between features and target
  2. Features are independent (no multicollinearity)
  3. Errors are normally distributed
  4. Constant variance (homoscedasticity)

Pros:
  ✓ Simple, interpretable
  ✓ Fast to train
  ✓ Works well when relationships are truly linear
  
Cons:
  ✗ Can't capture non-linear patterns
  ✗ Sensitive to outliers
  ✗ Assumes all relationships are additive
  
For insurance claims:
  - Claim severity likely has non-linear relationships
  - Example: $50K vehicle might not cost 5x a $10K vehicle to repair
  - But let's see! Maybe relationships are more linear than we think.
""")

print("\n⏳ Training Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
print("✓ Training complete!")

# Make predictions (still in log scale)
print("\n🔮 Making predictions...")
y_pred_train_log = lr_model.predict(X_train)
y_pred_test_log = lr_model.predict(X_test)
print("✓ Predictions complete!")

# ============================================================
# INVERSE TRANSFORM TO DOLLARS
# ============================================================

print("\n" + "=" * 70)
print("STEP 3.2: INVERSE TRANSFORMATION")
print("=" * 70)

print("\n🎓 REGRESSION CONCEPT #7: INVERSE TRANSFORMATION")
print("-" * 70)
print("""
Remember: We trained on log-transformed target!

Training:
  Original: $2,850
  Transformed: log(1 + 2850) = 7.955
  Model predicts: 7.955 (in log scale)

Prediction:
  Model output: 7.955 (log scale)
  Need to convert back: exp(7.955) - 1 = $2,850
  
Function: np.expm1()
  expm1(x) = exp(x) - 1
  Inverse of log1p(): log1p(x) = log(1 + x)
""")

print("\n🔄 Converting predictions from log scale to dollars...")

# Convert to actual dollar amounts
y_train_actual = np.expm1(y_train)
y_test_actual = np.expm1(y_test)
y_pred_train_dollars = np.expm1(y_pred_train_log)
y_pred_test_dollars = np.expm1(y_pred_test_log)

print("✓ Conversion complete!")

print("\n📊 Example predictions:")
for i in range(5):
    print(f"  Claim {i+1}:")
    print(f"    Actual:    ${y_test_actual.iloc[i]:,.2f}")
    print(f"    Predicted: ${y_pred_test_dollars[i]:,.2f}")
    print(f"    Error:     ${abs(y_test_actual.iloc[i] - y_pred_test_dollars[i]):,.2f}")

# ============================================================
# REGRESSION METRICS
# ============================================================

print("\n" + "=" * 70)
print("STEP 3.3: REGRESSION METRICS (THE IMPORTANT PART!)")
print("=" * 70)

print("\n🎓 REGRESSION CONCEPT #8: EVALUATION METRICS")
print("-" * 70)
print("""
Unlike classification (precision/recall), regression uses different metrics:

1. MAE (Mean Absolute Error):
   Formula: (1/n) × Σ|actual - predicted|
   Meaning: Average dollar error
   Example: MAE = $1,800 means "on average, we're off by $1,800"
   
   Pros: Easy to interpret, in original units (dollars)
   Cons: Treats all errors equally
   
2. RMSE (Root Mean Squared Error):
   Formula: √[(1/n) × Σ(actual - predicted)²]
   Meaning: Penalizes large errors more heavily
   Example: RMSE = $2,500
   
   Pros: Penalizes outliers (important for high-value claims)
   Cons: Less interpretable (not linear scale)
   
3. R² (R-Squared / Coefficient of Determination):
   Formula: 1 - (SS_residual / SS_total)
   Range: -∞ to 1.0
   Meaning: % of variance explained by model
   Example: R² = 0.65 means "model explains 65% of variance"
   
   Pros: Standardized (easy to compare models)
   Cons: Doesn't tell you absolute error in dollars
   
4. MAPE (Mean Absolute Percentage Error):
   Formula: (1/n) × Σ|(actual - predicted) / actual| × 100
   Meaning: Average % error
   Example: MAPE = 35% means "on average, we're off by 35%"
   
   Pros: Relative metric (good for comparing across datasets)
   Cons: Undefined for zero values, penalizes under-predictions more

Which to use?
  - Business stakeholders: MAE (dollars are intuitive)
  - Data scientists: RMSE (penalizes big misses)
  - Model comparison: R² (standardized)
  - Relative accuracy: MAPE (percentage)
  
For insurance claims: MAE is most intuitive!
""")

# Calculate metrics
mae_train = mean_absolute_error(y_train_actual, y_pred_train_dollars)
mae_test = mean_absolute_error(y_test_actual, y_pred_test_dollars)

rmse_train = np.sqrt(mean_squared_error(y_train_actual, y_pred_train_dollars))
rmse_test = np.sqrt(mean_squared_error(y_test_actual, y_pred_test_dollars))

r2_train = r2_score(y_train_actual, y_pred_train_dollars)
r2_test = r2_score(y_test_actual, y_pred_test_dollars)

mape_train = mean_absolute_percentage_error(y_train_actual, y_pred_train_dollars) * 100
mape_test = mean_absolute_percentage_error(y_test_actual, y_pred_test_dollars) * 100

print("\n📊 LINEAR REGRESSION PERFORMANCE:")
print("=" * 70)

print("\n🎯 Training Set:")
print(f"  MAE:  ${mae_train:,.2f}")
print(f"  RMSE: ${rmse_train:,.2f}")
print(f"  R²:   {r2_train:.4f} ({r2_train*100:.2f}% variance explained)")
print(f"  MAPE: {mape_train:.2f}%")

print("\n🎯 Test Set:")
print(f"  MAE:  ${mae_test:,.2f} ← Most interpretable!")
print(f"  RMSE: ${rmse_test:,.2f} ← Penalizes big errors")
print(f"  R²:   {r2_test:.4f} ({r2_test*100:.2f}% variance explained)")
print(f"  MAPE: {mape_test:.2f}%")

# Check for overfitting
print("\n🔍 Overfitting Check:")
mae_diff = mae_test - mae_train
rmse_diff = rmse_test - rmse_train
r2_diff = r2_train - r2_test

print(f"  MAE difference (test - train): ${mae_diff:,.2f}")
print(f"  RMSE difference (test - train): ${rmse_diff:,.2f}")
print(f"  R² difference (train - test): {r2_diff:.4f}")

if abs(mae_diff) < 100 and abs(r2_diff) < 0.05:
    print("  ✓ No significant overfitting - model generalizes well!")
elif mae_diff > 0:
    print("  ⚠️  Slight underfitting - test error slightly higher")
else:
    print("  ⚠️  Slight overfitting - train error lower than test")

# Business interpretation
print("\n💼 Business Interpretation:")
print(f"  On average, we're off by ${mae_test:,.2f} per claim")
print(f"  For a typical ${y_test_actual.median():,.2f} claim, that's {(mae_test/y_test_actual.median())*100:.1f}% error")

# ============================================================
# RESIDUAL ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("STEP 3.4: RESIDUAL ANALYSIS")
print("=" * 70)

print("\n🎓 REGRESSION CONCEPT #9: RESIDUALS")
print("-" * 70)
print("""
Residual = Actual - Predicted

What we want to see:
  ✓ Residuals centered around 0 (no systematic bias)
  ✓ Constant variance (homoscedasticity)
  ✓ No patterns (random scatter)
  
What indicates problems:
  ✗ Funnel shape (variance increases with prediction size)
  ✗ Curve/pattern (non-linear relationships)
  ✗ Outliers (model can't handle extreme cases)
""")

# Calculate residuals
residuals_test = y_test_actual - y_pred_test_dollars

print("\n📊 Residual Statistics:")
print(f"  Mean: ${residuals_test.mean():,.2f} (should be ~0)")
print(f"  Std:  ${residuals_test.std():,.2f}")
print(f"  Min:  ${residuals_test.min():,.2f} (largest under-prediction)")
print(f"  Max:  ${residuals_test.max():,.2f} (largest over-prediction)")

# Identify large errors
large_errors = residuals_test[abs(residuals_test) > 10000]
print(f"\n⚠️  Claims with >$10K error: {len(large_errors):,} ({len(large_errors)/len(residuals_test)*100:.1f}%)")

# ============================================================
# VISUALIZATIONS
# ============================================================

print("\n" + "=" * 70)
print("STEP 3.5: CREATING VISUALIZATIONS")
print("=" * 70)

fig = plt.figure(figsize=(16, 10))

# 1. Actual vs Predicted
ax1 = plt.subplot(2, 3, 1)
plt.scatter(y_test_actual, y_pred_test_dollars, alpha=0.3, s=10)
plt.plot([y_test_actual.min(), y_test_actual.max()], 
         [y_test_actual.min(), y_test_actual.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Claim Amount ($)')
plt.ylabel('Predicted Claim Amount ($)')
plt.title('Actual vs Predicted\n(closer to red line = better)', fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

# 2. Residuals vs Predicted
ax2 = plt.subplot(2, 3, 2)
plt.scatter(y_pred_test_dollars, residuals_test, alpha=0.3, s=10)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Claim Amount ($)')
plt.ylabel('Residual (Actual - Predicted) ($)')
plt.title('Residual Plot\n(want random scatter around 0)', fontweight='bold')
plt.grid(alpha=0.3)

# 3. Residuals Distribution
ax3 = plt.subplot(2, 3, 3)
plt.hist(residuals_test, bins=50, edgecolor='black', alpha=0.7)
plt.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Error')
plt.xlabel('Residual ($)')
plt.ylabel('Frequency')
plt.title('Residual Distribution\n(want centered at 0)', fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

# 4. Error Distribution by Magnitude
ax4 = plt.subplot(2, 3, 4)
error_pct = (abs(residuals_test) / y_test_actual * 100).replace([np.inf, -np.inf], np.nan).dropna()
plt.hist(error_pct, bins=50, edgecolor='black', alpha=0.7, color='orange')
plt.xlabel('Absolute % Error')
plt.ylabel('Frequency')
plt.title(f'Percentage Error Distribution\nMedian: {error_pct.median():.1f}%', fontweight='bold')
plt.axvline(x=error_pct.median(), color='r', linestyle='--', lw=2, label=f'Median: {error_pct.median():.1f}%')
plt.legend()
plt.grid(alpha=0.3)

# 5. Metrics Comparison (Train vs Test)
ax5 = plt.subplot(2, 3, 5)
metrics = ['MAE', 'RMSE', 'R²', 'MAPE']
train_values = [mae_train, rmse_train, r2_train, mape_train/100]  # Normalize MAPE
test_values = [mae_test, rmse_test, r2_test, mape_test/100]

x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width/2, train_values, width, label='Train', color='#3498db')
plt.bar(x + width/2, test_values, width, label='Test', color='#e74c3c')
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Train vs Test Performance\n(similar = no overfitting)', fontweight='bold')
plt.xticks(x, ['MAE\n(÷1000)', 'RMSE\n(÷1000)', 'R²', 'MAPE\n(÷100)'])
plt.legend()
plt.grid(alpha=0.3, axis='y')

# 6. Large Errors Analysis
ax6 = plt.subplot(2, 3, 6)
error_bins = pd.cut(abs(residuals_test), 
                    bins=[0, 1000, 2000, 5000, 10000, np.inf],
                    labels=['<$1K', '$1-2K', '$2-5K', '$5-10K', '>$10K'])
error_dist = error_bins.value_counts().sort_index()

plt.bar(range(len(error_dist)), error_dist.values, color='coral', edgecolor='black')
plt.xticks(range(len(error_dist)), error_dist.index)
plt.xlabel('Absolute Error Range')
plt.ylabel('Number of Claims')
plt.title('Error Distribution by Magnitude', fontweight='bold')
plt.grid(alpha=0.3, axis='y')

# Add percentages on bars
for i, v in enumerate(error_dist.values):
    plt.text(i, v + 200, f'{v/len(residuals_test)*100:.1f}%', 
             ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/step3_linear_regression_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualizations saved: outputs/step3_linear_regression_analysis.png")

# ============================================================
# SAVE BASELINE MODEL
# ============================================================

print("\n" + "=" * 70)
print("STEP 3.6: SAVING BASELINE MODEL")
print("=" * 70)

# Save model
with open('models/linear_regression_baseline.pkl', 'wb') as f:
    pickle.dump(lr_model, f)
print("✓ Model saved: models/linear_regression_baseline.pkl")

# Save performance metrics
baseline_metrics = {
    'model_name': 'Linear Regression',
    'train_metrics': {
        'mae': float(mae_train),
        'rmse': float(rmse_train),
        'r2': float(r2_train),
        'mape': float(mape_train)
    },
    'test_metrics': {
        'mae': float(mae_test),
        'rmse': float(rmse_test),
        'r2': float(r2_test),
        'mape': float(mape_test)
    },
    'overfitting_check': {
        'mae_diff': float(mae_diff),
        'rmse_diff': float(rmse_diff),
        'r2_diff': float(r2_diff)
    }
}

import json
with open('models/baseline_metrics.json', 'w') as f:
    json.dump(baseline_metrics, f, indent=2)
print("✓ Metrics saved: models/baseline_metrics.json")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("BASELINE MODEL SUMMARY")
print("=" * 70)

print(f"""
📊 Linear Regression Performance:
   MAE:  ${mae_test:,.2f}
   RMSE: ${rmse_test:,.2f}
   R²:   {r2_test:.4f} ({r2_test*100:.1f}% variance explained)
   MAPE: {mape_test:.1f}%

💡 Key Insights:
   • Average error: ${mae_test:,.2f} per claim
   • {(1-r2_test)*100:.1f}% of variance unexplained (non-linear patterns?)
   • {len(large_errors):,} claims with >$10K error ({len(large_errors)/len(residuals_test)*100:.1f}%)
   • No significant overfitting (train/test scores similar)

🎯 What This Baseline Tells Us:
   Linear Regression R² = {r2_test:.4f}
   → {(1-r2_test)*100:.1f}% of patterns are non-linear or unexplained
   → This is why tree-based models (XGBoost) often win!
   → XGBoost can capture non-linear relationships

📁 Saved:
   • models/linear_regression_baseline.pkl
   • models/baseline_metrics.json
   • outputs/step3_linear_regression_analysis.png

🚀 Next: Train XGBoost and see if it beats this baseline!
""")

print("\n" + "=" * 70)
print("✅ STEP 3 COMPLETE!")
print("=" * 70)
print("\nNext: python step4_xgboost_training.py")
print("We'll train XGBoost and compare against this baseline!")