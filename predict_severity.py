"""
PREDICT CLAIM SEVERITY - USING TRAINED XGBOOST MODEL
====================================================
Goal: Make predictions on new insurance claims
Shows: Complete prediction pipeline from raw input to dollar amount
"""

import pandas as pd
import numpy as np
import pickle
import json

print("=" * 70)
print("CLAIM SEVERITY PREDICTION - XGBOOST")
print("=" * 70)

# ============================================================
# LOAD MODEL AND PREPROCESSING ARTIFACTS
# ============================================================

print("\n📥 Loading model and preprocessing artifacts...")

# Load trained XGBoost model
with open('models/xgboost_regressor.pkl', 'rb') as f:
    xgb_model = pickle.load(f)
print("✓ XGBoost model loaded")

# Load label encoders (needed to encode categorical features)
with open('models/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)
print(f"✓ Label encoders loaded ({len(label_encoders)} encoders)")

# Load feature names (to ensure correct order)
with open('models/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)
print(f"✓ Feature names loaded ({len(feature_names)} features)")

# Load model metrics (for context)
with open('models/xgboost_metrics.json', 'r') as f:
    model_metrics = json.load(f)

print(f"\n📊 Model Performance:")
print(f"  MAE:  ${model_metrics['test_metrics']['mae']:,.2f}")
print(f"  RMSE: ${model_metrics['test_metrics']['rmse']:,.2f}")
print(f"  R²:   {model_metrics['test_metrics']['r2']:.4f}")

# ============================================================
# SAMPLE CLAIMS FOR PREDICTION
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE CLAIMS FOR PREDICTION")
print("=" * 70)

# Create sample claims (you can modify these!)
sample_claims = [
    {
        "name": "Low-Value Claim (Minor Fender Bender)",
        "description": "Small sedan, minor damage, no injury",
        # Note: In real Allstate data, features are like cat1, cat2, cont1, cont2
        # For this example, I'll create a synthetic claim based on typical patterns
        # You'll need to replace these with actual feature names from your dataset
    },
    {
        "name": "Medium-Value Claim (Moderate Collision)",
        "description": "Mid-size SUV, moderate damage, possible injury",
    },
    {
        "name": "High-Value Claim (Major Accident)",
        "description": "Luxury vehicle, severe damage, injury involved",
    }
]

# ============================================================
# HELPER FUNCTION: PREPROCESS AND PREDICT
# ============================================================

def predict_claim_severity(claim_data, model, encoders, feature_names):
    """
    Preprocess claim and predict severity
    
    Args:
        claim_data: Dictionary of claim features
        model: Trained XGBoost model
        encoders: Dictionary of label encoders
        feature_names: List of feature names in correct order
        
    Returns:
        predicted_amount: Predicted claim severity in dollars
    """
    # Convert to DataFrame
    df = pd.DataFrame([claim_data])
    
    # Apply label encoding to categorical features
    for col, encoder in encoders.items():
        if col in df.columns:
            # Handle unseen categories (use most frequent category)
            try:
                df[col] = encoder.transform(df[col].astype(str))
            except ValueError:
                # If category not seen during training, use mode
                df[col] = 0  # Default to first category
    
    # Ensure all features are present and in correct order
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0  # Add missing features as 0
    
    df = df[feature_names]  # Reorder columns
    
    # Make prediction (in log scale)
    log_prediction = model.predict(df)[0]
    
    # Convert back to dollars
    dollar_prediction = np.expm1(log_prediction)
    
    return dollar_prediction, log_prediction

# ============================================================
# LOAD ACTUAL TEST DATA FOR REAL PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("MAKING PREDICTIONS ON REAL TEST CLAIMS")
print("=" * 70)

# Load test data
test_data = pd.read_csv('models/test_data.csv')

# Separate features and actual values
X_test = test_data.drop('log_loss', axis=1)
y_test_log = test_data['log_loss']
y_test_actual = np.expm1(y_test_log)

# Select 5 random claims to predict
sample_indices = np.random.choice(len(X_test), 5, replace=False)

print("\n🔮 Predicting severity for 5 random test claims...\n")

for i, idx in enumerate(sample_indices, 1):
    # Get claim data
    claim_features = X_test.iloc[idx].to_dict()
    actual_amount = y_test_actual.iloc[idx]
    
    # Make prediction
    predicted_amount, log_pred = predict_claim_severity(
        claim_features, 
        xgb_model, 
        label_encoders, 
        feature_names
    )
    
    # Calculate error
    error = abs(actual_amount - predicted_amount)
    error_pct = (error / actual_amount) * 100
    
    print(f"{'='*70}")
    print(f"CLAIM #{i}")
    print(f"{'='*70}")
    print(f"  Actual Amount:    ${actual_amount:,.2f}")
    print(f"  Predicted Amount: ${predicted_amount:,.2f}")
    print(f"  Error:            ${error:,.2f} ({error_pct:.1f}%)")
    
    # Interpretation
    if error_pct < 20:
        print(f"  ✓ Excellent prediction (< 20% error)")
    elif error_pct < 40:
        print(f"  ✓ Good prediction (20-40% error)")
    elif error_pct < 60:
        print(f"  ⚠️  Moderate error (40-60%)")
    else:
        print(f"  ❌ High error (> 60%)")
    
    # Show log-scale prediction (for educational purposes)
    print(f"\n  [Technical] Log prediction: {log_pred:.4f}")
    print(f"  [Technical] Inverse transform: exp({log_pred:.4f}) - 1 = ${predicted_amount:,.2f}")
    print()

# ============================================================
# AGGREGATE STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION STATISTICS (ON ALL TEST CLAIMS)")
print("=" * 70)

# Predict on entire test set
y_pred_log = xgb_model.predict(X_test)
y_pred_dollars = np.expm1(y_pred_log)

# Calculate metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test_actual, y_pred_dollars)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_dollars))
r2 = r2_score(y_test_actual, y_pred_dollars)

print(f"\n📊 Performance Metrics:")
print(f"  MAE:  ${mae:,.2f}")
print(f"  RMSE: ${rmse:,.2f}")
print(f"  R²:   {r2:.4f} ({r2*100:.1f}% variance explained)")

# Error distribution
errors = np.abs(y_test_actual - y_pred_dollars)
error_pct = (errors / y_test_actual) * 100

print(f"\n📈 Error Distribution:")
print(f"  Mean error:   ${errors.mean():,.2f}")
print(f"  Median error: ${np.median(errors):,.2f}")
print(f"  25th percentile: ${np.percentile(errors, 25):,.2f}")
print(f"  75th percentile: ${np.percentile(errors, 75):,.2f}")

print(f"\n📊 Percentage Error Distribution:")
print(f"  < 20% error: {(error_pct < 20).sum():,} claims ({(error_pct < 20).sum()/len(error_pct)*100:.1f}%)")
print(f"  20-40% error: {((error_pct >= 20) & (error_pct < 40)).sum():,} claims ({((error_pct >= 20) & (error_pct < 40)).sum()/len(error_pct)*100:.1f}%)")
print(f"  40-60% error: {((error_pct >= 40) & (error_pct < 60)).sum():,} claims ({((error_pct >= 40) & (error_pct < 60)).sum()/len(error_pct)*100:.1f}%)")
print(f"  > 60% error:  {(error_pct >= 60).sum():,} claims ({(error_pct >= 60).sum()/len(error_pct)*100:.1f}%)")

# ============================================================
# PREDICT ON NEW CUSTOM CLAIM
# ============================================================

print("\n" + "=" * 70)
print("CUSTOM CLAIM PREDICTION")
print("=" * 70)

print("""
Want to predict severity for a custom claim?

To do this, you need to provide values for all 130 features.
For the Allstate dataset, features are named like:
  - cat1, cat2, cat3, ... cat116 (categorical)
  - cont1, cont2, cont3, ... cont14 (continuous)

Example custom claim:
  You can create a dictionary with all feature values and call:
  
  custom_claim = {
      'cat1': 'A',
      'cat2': 'B',
      'cont1': 0.5,
      'cont2': 0.3,
      ... (all 130 features)
  }
  
  predicted_amount, _ = predict_claim_severity(
      custom_claim, 
      xgb_model, 
      label_encoders, 
      feature_names
  )
  
  print(f"Predicted severity: ${predicted_amount:,.2f}")
""")

# ============================================================
# BUSINESS INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("BUSINESS USE CASES")
print("=" * 70)

print("""
💼 How This Model is Used in Production:

1. CLAIM ROUTING:
   - Claim submitted → Model predicts severity
   - Low (<$2K): Junior adjuster, standard queue
   - Medium ($2-10K): Mid-level adjuster, priority queue
   - High (>$10K): Senior adjuster, expedited handling
   
2. RESERVE ALLOCATION:
   - New claim comes in → Predict $8,450
   - Set aside $8,450 in reserves immediately
   - More accurate than flat-rate reserves
   - Better capital planning for CFO
   
3. COMBINED WITH FRAUD DETECTION:
   - Fraud probability: 67%
   - Predicted severity: $12,500
   - Risk score: HIGH (both high fraud + high value)
   - → Route to senior fraud investigator immediately
   
4. SLA MANAGEMENT:
   - Predict all claims for the week
   - Sort by severity (descending)
   - Ensure high-value claims get handled first
   - Customer satisfaction for major claims
   
5. CASHFLOW FORECASTING:
   - Predict severity for all pending claims
   - Sum up: $2.5M expected payouts this month
   - CFO can plan accordingly
   - Better liquidity management
""")

print("\n" + "=" * 70)
print("✅ PREDICTION COMPLETE!")
print("=" * 70)
print("\n💡 Next Steps:")
print("  • Integrate with fraud detection API")
print("  • Create combined /predict endpoint")
print("  • Deploy to production")