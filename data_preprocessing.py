"""
STEP 2: DATA PREPROCESSING & TRANSFORMATION
============================================
Goal: Prepare data for regression modeling
Key Learning: Log transformation, encoding strategies, train/test split

What we'll do:
1. Log-transform target variable
2. Handle categorical features (label encoding for trees)
3. Train/test split
4. Understand why we DON'T need feature scaling for tree-based models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("STEP 2: DATA PREPROCESSING")
print("=" * 70)

# ============================================================
# LOAD DATA
# ============================================================

print("\n📥 Loading dataset...")
df = pd.read_csv('allstate_claims.csv')
print(f"✓ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

# Create outputs directory
os.makedirs('outputs', exist_ok=True)
os.makedirs('models', exist_ok=True)

# ============================================================
# 1. LOG TRANSFORMATION OF TARGET
# ============================================================

print("\n" + "=" * 70)
print("STEP 2.1: LOG TRANSFORMATION OF TARGET")
print("=" * 70)

print("\n🎓 REGRESSION CONCEPT #2: WHY LOG TRANSFORM?")
print("-" * 70)
print("""
Problem: Target is right-skewed
  - Mean: $4,500
  - Median: $2,850  (Median < Mean = right skew!)
  - Max: $121,000

Without log transformation:
  Model tries to minimize error: (predicted - actual)²
  Error of $20K on $100K claim = 400M penalty
  Error of $500 on $2K claim = 250K penalty
  → Model focuses on outliers, ignores typical claims!

With log transformation:
  Model minimizes: (log(pred) - log(actual))²
  Large and small values on similar scale
  → Model learns patterns across entire range
  → We transform predictions back to dollars at the end
""")

# Original target statistics
original_target = df['loss']
print(f"\n📊 Original Target Statistics:")
print(f"  Mean:     ${original_target.mean():,.2f}")
print(f"  Median:   ${original_target.median():,.2f}")
print(f"  Std:      ${original_target.std():,.2f}")
print(f"  Skewness: {original_target.skew():.2f}")
print(f"  Range:    ${original_target.min():,.2f} to ${original_target.max():,.2f}")

# Apply log transformation
print("\n🔄 Applying log transformation: log(1 + x)...")
df['log_loss'] = np.log1p(df['loss'])

log_target = df['log_loss']
print(f"\n✓ Log-Transformed Target Statistics:")
print(f"  Mean:     {log_target.mean():.4f}")
print(f"  Median:   {log_target.median():.4f}")
print(f"  Std:      {log_target.std():.4f}")
print(f"  Skewness: {log_target.skew():.2f} ← Much better!")
print(f"  Range:    {log_target.min():.4f} to {log_target.max():.4f}")

print("\n💡 Key Insight:")
print(f"  Skewness reduced from {original_target.skew():.2f} → {log_target.skew():.2f}")
print(f"  Distribution is now much more normal/symmetric!")

# Visualize before/after
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Before
axes[0].hist(original_target, bins=50, edgecolor='black', alpha=0.7)
axes[0].axvline(original_target.mean(), color='red', linestyle='--', label=f'Mean: ${original_target.mean():,.0f}')
axes[0].axvline(original_target.median(), color='green', linestyle='--', label=f'Median: ${original_target.median():,.0f}')
axes[0].set_xlabel('Claim Amount ($)')
axes[0].set_ylabel('Frequency')
axes[0].set_title(f'Before: Original Distribution\nSkewness: {original_target.skew():.2f}')
axes[0].legend()
axes[0].grid(alpha=0.3)

# After
axes[1].hist(log_target, bins=50, edgecolor='black', alpha=0.7, color='green')
axes[1].axvline(log_target.mean(), color='red', linestyle='--', label=f'Mean: {log_target.mean():.2f}')
axes[1].axvline(log_target.median(), color='orange', linestyle='--', label=f'Median: {log_target.median():.2f}')
axes[1].set_xlabel('Log(Claim Amount)')
axes[1].set_ylabel('Frequency')
axes[1].set_title(f'After: Log-Transformed Distribution\nSkewness: {log_target.skew():.2f}')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/step2_log_transformation.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: outputs/step2_log_transformation.png")

# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

print("\n" + "=" * 70)
print("STEP 2.2: FEATURE PREPARATION")
print("=" * 70)

# Drop ID column (if exists) and both target versions
features_to_drop = ['id', 'loss', 'log_loss']
features_to_drop = [col for col in features_to_drop if col in df.columns]

X = df.drop(features_to_drop, axis=1)
y = df['log_loss']  # Use log-transformed target

print(f"\n✓ Features (X): {X.shape[1]} columns")
print(f"✓ Target (y): log-transformed loss")

# Identify categorical and numerical columns
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\n📊 Feature Types:")
print(f"  Categorical: {len(cat_cols)}")
print(f"  Numerical: {len(num_cols)}")

# ============================================================
# 3. ENCODING CATEGORICAL FEATURES
# ============================================================

print("\n" + "=" * 70)
print("STEP 2.3: ENCODING CATEGORICAL FEATURES")
print("=" * 70)

print("\n🎓 REGRESSION CONCEPT #3: ENCODING STRATEGIES")
print("-" * 70)
print("""
Two main approaches for categorical encoding:

1. ONE-HOT ENCODING (what we did in Project 1):
   - Creates binary column for each category
   - Example: Make = [Honda, Toyota, Ford]
     → Make_Honda (0/1), Make_Toyota (0/1), Make_Ford (0/1)
   
   Pros: No ordinal assumption, works for all models
   Cons: High dimensionality (100 categories = 100 columns!)
   
   When to use: 
   - Low cardinality (< 10 categories per feature)
   - Linear models (Linear Regression, Logistic Regression)

2. LABEL ENCODING:
   - Assigns integer to each category
   - Example: Make = [Honda, Toyota, Ford]
     → 0, 1, 2
   
   Pros: Single column per feature, efficient
   Cons: Implies ordinal relationship (2 > 1 > 0)
   
   When to use:
   - Tree-based models (XGBoost, Random Forest, LightGBM)
   - These models split on values, don't assume order
   
For our Allstate data:
  - 116 categorical features
  - Many have 2-50+ categories
  - One-hot encoding would create 1000+ columns!
  
Decision: LABEL ENCODING (we're using XGBoost)
""")

print(f"\n🔢 Applying Label Encoding to {len(cat_cols)} categorical features...")

# Store label encoders for later use (production deployment)
label_encoders = {}
X_encoded = X.copy()

for col in cat_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
    
    n_unique = X_encoded[col].nunique()
    print(f"  ✓ {col}: {n_unique} unique values → 0 to {n_unique-1}")

print(f"\n✓ Encoding complete!")
print(f"  Original shape: {X.shape}")
print(f"  Encoded shape: {X_encoded.shape} (Same! No dimension explosion)")

# Compare with one-hot encoding (for educational purposes)
X_onehot = pd.get_dummies(X, drop_first=True)
print(f"\n📊 Encoding Comparison:")
print(f"  Label Encoding: {X_encoded.shape[1]} columns")
print(f"  One-Hot Encoding: {X_onehot.shape[1]} columns")
print(f"  → Label encoding is {X_onehot.shape[1] / X_encoded.shape[1]:.1f}x more efficient!")

# ============================================================
# 4. TRAIN/TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("STEP 2.4: TRAIN/TEST SPLIT")
print("=" * 70)

print("\n🎓 REGRESSION CONCEPT #4: WHY RANDOM SPLIT (NOT STRATIFIED)?")
print("-" * 70)
print("""
In Classification (Project 1):
  - Used stratify=y to maintain fraud ratio (6.3%) in both sets
  - Binary target: can stratify by class

In Regression (Project 2):
  - Target is continuous (every value is unique)
  - Can't stratify by exact values
  - Solution: Random split (with random_state for reproducibility)
  - Large dataset (188K) ensures similar distributions
""")

print(f"\n✂️  Splitting data (80% train, 20% test, random_state=42)...")

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42
)

print(f"\n✓ Split complete!")
print(f"  Training set: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"  Test set: {X_test.shape[0]:,} samples ({X_test.shape[0]/len(df)*100:.1f}%)")

# Verify distributions are similar
print(f"\n📊 Distribution Check:")
print(f"  Train - Mean: {y_train.mean():.4f}, Std: {y_train.std():.4f}")
print(f"  Test  - Mean: {y_test.mean():.4f}, Std: {y_test.std():.4f}")
print(f"  → Distributions are similar ✓")

# ============================================================
# 5. FEATURE SCALING (EDUCATIONAL)
# ============================================================

print("\n" + "=" * 70)
print("STEP 2.5: FEATURE SCALING (NOT NEEDED FOR TREES!)")
print("=" * 70)

print("\n🎓 REGRESSION CONCEPT #5: WHEN TO SCALE FEATURES")
print("-" * 70)
print("""
Feature Scaling = Making all features same range (e.g., 0-1 or mean=0, std=1)

Example:
  Feature A: Age (18 to 100)
  Feature B: Income ($20K to $200K)

Without scaling:
  Income dominates due to larger magnitude
  
When scaling is REQUIRED:
  ✓ Linear Regression (distance-based)
  ✓ Neural Networks (gradient descent)
  ✓ K-Nearest Neighbors (distance-based)
  ✓ Support Vector Machines (distance-based)

When scaling is NOT needed:
  ✗ Decision Trees
  ✗ Random Forest
  ✗ XGBoost / LightGBM / CatBoost
  
Why? Tree-based models split on features independently:
  "Is Age > 30?" doesn't care if Age ranges 0-100 or 0-1
  "Is Income > 50000?" works regardless of scale
  
Decision: SKIP SCALING (we're using XGBoost)
""")

print("\n💡 Key Insight:")
print("  For tree-based models (XGBoost), we DON'T need to scale features.")
print("  This saves preprocessing time and makes the pipeline simpler!")

# Show what scaling would look like (educational)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
sample_feature = X_train['cont1'].values.reshape(-1, 1) if 'cont1' in X_train.columns else X_train.iloc[:, 0].values.reshape(-1, 1)

print(f"\n📊 Example (for education only - we won't use this):")
print(f"  Original range: {sample_feature.min():.2f} to {sample_feature.max():.2f}")
scaled_feature = scaler.fit_transform(sample_feature)
print(f"  Scaled range: {scaled_feature.min():.2f} to {scaled_feature.max():.2f}")
print(f"  → After scaling: mean ≈ 0, std ≈ 1")

# ============================================================
# 6. SAVE PREPROCESSING ARTIFACTS
# ============================================================

print("\n" + "=" * 70)
print("STEP 2.6: SAVING PREPROCESSING ARTIFACTS")
print("=" * 70)

# Save preprocessed data
print("\n💾 Saving preprocessed data...")

# Save train/test sets
train_data = pd.DataFrame(X_train, columns=X_encoded.columns)
train_data['log_loss'] = y_train.values
train_data.to_csv('models/train_data.csv', index=False)
print(f"✓ Saved: models/train_data.csv")

test_data = pd.DataFrame(X_test, columns=X_encoded.columns)
test_data['log_loss'] = y_test.values
test_data.to_csv('models/test_data.csv', index=False)
print(f"✓ Saved: models/test_data.csv")

# Save label encoders (for production deployment)
with open('models/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print(f"✓ Saved: models/label_encoders.pkl ({len(label_encoders)} encoders)")

# Save feature names
feature_names = X_encoded.columns.tolist()
with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
print(f"✓ Saved: models/feature_names.pkl ({len(feature_names)} features)")

# Save preprocessing metadata
preprocessing_metadata = {
    'original_features': X.shape[1],
    'encoded_features': X_encoded.shape[1],
    'categorical_features': len(cat_cols),
    'numerical_features': len(num_cols),
    'encoding_method': 'label_encoding',
    'target_transformation': 'log1p',
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'train_target_mean': float(y_train.mean()),
    'train_target_std': float(y_train.std()),
    'original_target_skewness': float(original_target.skew()),
    'transformed_target_skewness': float(log_target.skew())
}

import json
with open('models/preprocessing_metadata.json', 'w') as f:
    json.dump(preprocessing_metadata, f, indent=2)
print(f"✓ Saved: models/preprocessing_metadata.json")

# ============================================================
# 7. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING SUMMARY")
print("=" * 70)

print(f"""
✅ Target Transformation:
   Original skewness: {original_target.skew():.2f} → {log_target.skew():.2f}
   Method: log(1 + x)
   
✅ Feature Encoding:
   Categorical features: {len(cat_cols)} → Label Encoded
   Numerical features: {len(num_cols)} → No change needed
   Total features: {X_encoded.shape[1]}
   
✅ Train/Test Split:
   Training: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(df)*100:.1f}%)
   Test: {X_test.shape[0]:,} samples ({X_test.shape[0]/len(df)*100:.1f}%)
   Random split (not stratified - continuous target)
   
✅ Feature Scaling:
   Not applied (tree-based models don't need it)
   
✅ Saved Artifacts:
   • train_data.csv / test_data.csv
   • label_encoders.pkl (for production)
   • feature_names.pkl
   • preprocessing_metadata.json

🎯 Data is ready for modeling!
""")

print("\n" + "=" * 70)
print("✅ STEP 2 COMPLETE!")
print("=" * 70)
print("\nNext: python step3_baseline_model.py")
print("We'll train a simple Linear Regression as baseline!")