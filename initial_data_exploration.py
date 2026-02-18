"""
STEP 1: DATA LOADING AND EXPLORATION
=====================================
Goal: Understand the dataset structure and target distribution
Key Learning: Regression targets are continuous, not binary
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)

print("=" * 70)
print("STEP 1: DATA EXPLORATION")
print("=" * 70)

# ============================================================
# LOAD DATA
# ============================================================

print("\n📥 Loading dataset...")
df = pd.read_csv('allstate_claims.csv')
print(f"✓ Data loaded successfully!")
print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ============================================================
# BASIC INFO
# ============================================================

print("\n" + "=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)

print("\n🔍 First few rows:")
print(df.head())

print("\n📋 Data types:")
dtype_counts = df.dtypes.value_counts()
print(dtype_counts)

print(f"\n📊 Column breakdown:")
print(f"  Total columns: {len(df.columns)}")
print(f"  Categorical: {len(df.select_dtypes(include=['object']).columns)}")
print(f"  Numerical: {len(df.select_dtypes(include=['int64', 'float64']).columns) - 1}")  # -1 for target

# ============================================================
# TARGET VARIABLE ANALYSIS (THE MOST IMPORTANT PART!)
# ============================================================

print("\n" + "=" * 70)
print("TARGET VARIABLE: 'loss' (Claim Severity in Dollars)")
print("=" * 70)

target = df['loss']

print("\n📈 Basic Statistics:")
print(target.describe())

print("\n💡 Key Insights:")
print(f"  Min claim:     ${target.min():,.2f}")
print(f"  Max claim:     ${target.max():,.2f}")
print(f"  Mean claim:    ${target.mean():,.2f}")
print(f"  Median claim:  ${target.median():,.2f}")
print(f"  Std deviation: ${target.std():,.2f}")

# Check for skewness
skewness = target.skew()
print(f"\n⚠️  Skewness: {skewness:.2f}")
if skewness > 1:
    print("     → Heavily RIGHT-SKEWED (long tail of high values)")
    print("     → Most claims are small, few are very large")
    print("     → This will require LOG TRANSFORMATION!")
elif skewness < -1:
    print("     → Heavily LEFT-SKEWED")
else:
    print("     → Relatively symmetric")

# ============================================================
# WHY SKEWNESS MATTERS
# ============================================================

print("\n" + "=" * 70)
print("🎓 REGRESSION CONCEPT #1: WHY SKEWNESS MATTERS")
print("=" * 70)

print("""
In CLASSIFICATION (Project 1):
  Target was binary: 0 or 1 (fraud or not fraud)
  Skewness = class imbalance (14.8:1 ratio)
  Solution = scale_pos_weight

In REGRESSION (Project 2):
  Target is continuous: $0.65 to $121,000
  Skewness = distribution shape
  Problem: Model gets dominated by outliers
  
Example:
  - 1,000 claims at $2,000 each
  - 10 claims at $100,000 each
  
Without transformation:
  → Model tries to predict the $100K claims accurately
  → Ignores the $2K claims (majority!)
  → Poor performance overall
  
With log transformation:
  → All values compressed to similar scale
  → Model learns patterns across entire range
  → Better predictions when we transform back
""")

# ============================================================
# VISUALIZE TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("VISUALIZING TARGET DISTRIBUTION")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Original distribution
axes[0].hist(target, bins=50, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Claim Amount ($)')
axes[0].set_ylabel('Frequency')
axes[0].set_title(f'Original Distribution\nSkewness: {skewness:.2f} (Right-Skewed)')
axes[0].grid(alpha=0.3)

# Log-transformed distribution
log_target = np.log1p(target)  # log(1 + x) to handle zeros
log_skewness = log_target.skew()
axes[1].hist(log_target, bins=50, edgecolor='black', alpha=0.7, color='green')
axes[1].set_xlabel('Log(Claim Amount)')
axes[1].set_ylabel('Frequency')
axes[1].set_title(f'Log-Transformed Distribution\nSkewness: {log_skewness:.2f} (Much Better!)')
axes[1].grid(alpha=0.3)

# Box plot (shows outliers clearly)
axes[2].boxplot(target, vert=True)
axes[2].set_ylabel('Claim Amount ($)')
axes[2].set_title('Box Plot\n(Many outliers above whisker)')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/step1_target_distribution.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved: outputs/step1_target_distribution.png")

# ============================================================
# MISSING VALUES CHECK
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Column': missing.index,
    'Missing': missing.values,
    'Percentage': missing_pct.values
})
missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False)

if len(missing_df) > 0:
    print("\n⚠️  Columns with missing values:")
    print(missing_df.to_string(index=False))
else:
    print("\n✓ No missing values found!")

# ============================================================
# FEATURE TYPES
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ANALYSIS")
print("=" * 70)

# Separate features
cat_features = df.select_dtypes(include=['object']).columns.tolist()
num_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
num_features.remove('loss')  # Remove target
if 'id' in num_features:
    num_features.remove('id')  # Remove ID if present

print(f"\n📊 Feature Breakdown:")
print(f"  Categorical: {len(cat_features)} features")
if len(cat_features) > 0:
    print(f"    Examples: {cat_features[:5]}")
    
print(f"\n  Numerical: {len(num_features)} features")
if len(num_features) > 0:
    print(f"    Examples: {num_features[:5]}")

# Check cardinality of categorical features
if len(cat_features) > 0:
    print("\n  Categorical Feature Cardinality:")
    for col in cat_features[:10]:  # First 10
        n_unique = df[col].nunique()
        print(f"    {col}: {n_unique} unique values")
        if n_unique > 100:
            print(f"      ⚠️  High cardinality - may need special encoding")

# ============================================================
# KEY INSIGHTS SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("KEY INSIGHTS FOR REGRESSION")
print("=" * 70)

print(f"""
✓ Dataset Size: {df.shape[0]:,} claims
✓ Features: {df.shape[1] - 1} (after removing target and ID)
✓ Target: Continuous dollar amounts (${target.min():.2f} to ${target.max():,.2f})
✓ Distribution: Heavily right-skewed (skewness: {skewness:.2f})
✓ Missing Values: {"Yes - needs handling" if len(missing_df) > 0 else "None"}

🎯 Next Steps:
1. Log-transform target to normalize distribution
2. Encode categorical features
3. Split into train/test
4. Train baseline model (Linear Regression)
5. Train XGBoost and compare
""")

print("\n" + "=" * 70)
print("✅ STEP 1 COMPLETE!")
print("=" * 70)
print("\nRun: python step2_preprocessing.py")