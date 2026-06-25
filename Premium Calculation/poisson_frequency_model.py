#!/usr/bin/env python3
"""
POISSON GLM FREQUENCY MODEL FOR CYBER & AI RISK
Actuarial Science Project - Premium Pricing Model
ML Techniques: Cross-validation, Regularization, Feature Selection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import PoissonRegressor, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_poisson_deviance
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. DATA LOADING & PREPARATION
# ============================================================================

print("=" * 80)
print("CYBER & AI RISK POISSON GLM FREQUENCY MODEL")
print("=" * 80)

# Load data
incidents = pd.read_csv('/mnt/project/incidents_master.csv')
financial = pd.read_csv('/mnt/project/financial_impact_1.csv')
market = pd.read_csv('/mnt/project/market_impact.csv')

print(f"\nLoaded {len(incidents)} incidents, {len(financial)} financial records, {len(market)} market records")

# Convert date columns to datetime
incidents['incident_date'] = pd.to_datetime(incidents['incident_date'])
incidents['disclosure_date'] = pd.to_datetime(incidents['disclosure_date'])
incidents['year'] = incidents['incident_date'].dt.year

print(f"Date range: {incidents['incident_date'].min().date()} to {incidents['incident_date'].max().date()}")
print(f"Year range: {incidents['year'].min()} to {incidents['year'].max()}")

# ============================================================================
# 2. FREQUENCY AGGREGATION
# ============================================================================

print("\n" + "=" * 80)
print("CREATING FREQUENCY AGGREGATIONS")
print("=" * 80)

# Aggregate by company (company-level frequency)
company_freq = incidents.groupby('company_name').agg({
    'incident_id': 'count',
    'company_revenue_usd': 'first',
    'employee_count': 'first',
    'country_hq': 'first',
    'industry_primary': 'first',
    'is_public_company': 'first',
    'attack_vector_primary': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown',
    'data_compromised_records': 'sum',
    'downtime_hours': 'mean'
}).reset_index()

company_freq.rename(columns={'incident_id': 'incident_count'}, inplace=True)
company_freq.fillna(0, inplace=True)

# Aggregate by industry (industry-level frequency)
industry_freq = incidents.groupby('industry_primary').agg({
    'incident_id': 'count',
    'company_revenue_usd': 'mean',
    'employee_count': 'mean',
    'data_compromised_records': 'mean'
}).reset_index()
industry_freq.rename(columns={'incident_id': 'incident_count'}, inplace=True)

# Aggregate by country (country-level frequency)
country_freq = incidents.groupby('country_hq').agg({
    'incident_id': 'count',
    'company_revenue_usd': 'mean',
    'employee_count': 'mean'
}).reset_index()
country_freq.rename(columns={'incident_id': 'incident_count'}, inplace=True)

# Aggregate by year (time trend)
year_freq = incidents.groupby('year').agg({
    'incident_id': 'count',
    'company_revenue_usd': 'mean',
    'data_compromised_records': 'mean'
}).reset_index()
year_freq.rename(columns={'incident_id': 'incident_count'}, inplace=True)

print(f"\nCompany-level aggregation: {len(company_freq)} unique companies")
print(f"Industry-level aggregation: {len(industry_freq)} industries")
print(f"Country-level aggregation: {len(country_freq)} countries")
print(f"Year range: {len(year_freq)} years")

print(f"\nIncident frequency statistics (by company):")
print(f"  Mean: {company_freq['incident_count'].mean():.2f}")
print(f"  Median: {company_freq['incident_count'].median():.2f}")
print(f"  Std Dev: {company_freq['incident_count'].std():.2f}")
print(f"  Min: {company_freq['incident_count'].min()}")
print(f"  Max: {company_freq['incident_count'].max()}")

# ============================================================================
# 3. EXPOSURE VARIABLE DEVELOPMENT
# ============================================================================

print("\n" + "=" * 80)
print("DEVELOPING EXPOSURE VARIABLES")
print("=" * 80)

# Use company-level data for main model
X = company_freq[['company_revenue_usd', 'employee_count', 'is_public_company']].copy()
y = company_freq['incident_count'].copy()

# Log transformation for revenue (avoid zero division)
X['log_revenue'] = np.log1p(X['company_revenue_usd'])
X['log_employees'] = np.log1p(X['employee_count'])
X['revenue_per_employee'] = X['company_revenue_usd'] / (X['employee_count'] + 1)
X['log_rev_per_emp'] = np.log1p(X['revenue_per_employee'])
X['is_public_company'] = X['is_public_company'].astype(int)

# Create size categories
X['company_size'] = pd.cut(X['company_revenue_usd'], 
                           bins=[0, 1e9, 10e9, 100e9, np.inf],
                           labels=['Small', 'Medium', 'Large', 'Enterprise'])

# Encode categorical variables
le_size = LabelEncoder()
X['company_size_encoded'] = le_size.fit_transform(X['company_size'])

# Select features for model
feature_cols = ['log_revenue', 'log_employees', 'is_public_company', 'company_size_encoded']
X_model = X[feature_cols].copy()
X_model = X_model.fillna(0)

print(f"\nFeatures selected: {feature_cols}")
print(f"\nFeature statistics:")
print(X_model.describe())

# Correlation analysis
print(f"\nFeature correlation with incident count:")
for col in feature_cols:
    corr = np.corrcoef(X_model[col], y)[0, 1]
    print(f"  {col}: {corr:.4f}")

# ============================================================================
# 4. POISSON GLM MODEL BUILDING
# ============================================================================

print("\n" + "=" * 80)
print("POISSON GLM MODEL DEVELOPMENT")
print("=" * 80)

# Split data for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_model, y, test_size=0.2, random_state=42)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Testing set: {len(X_test)} samples")

# Baseline Poisson GLM Model
poisson_model = PoissonRegressor(alpha=0, max_iter=1000)
poisson_model.fit(X_train, y_train)

print(f"\n--- BASELINE POISSON GLM ---")
print(f"Training deviance: {-2 * poisson_model.score(X_train, y_train):.4f}")
print(f"Testing deviance: {-2 * poisson_model.score(X_test, y_test):.4f}")
print(f"MAE (test): {mean_absolute_error(y_test, poisson_model.predict(X_test)):.4f}")
print(f"\nModel Coefficients:")
for feat, coef in zip(feature_cols, poisson_model.coef_):
    print(f"  {feat}: {coef:.4f}")
print(f"  Intercept: {poisson_model.intercept_:.4f}")

# ============================================================================
# 5. ML TECHNIQUES: REGULARIZATION & FEATURE SELECTION
# ============================================================================

print("\n" + "=" * 80)
print("REGULARIZATION & FEATURE SELECTION")
print("=" * 80)

# Ridge Regression (L2 regularization)
ridge_model = PoissonRegressor(alpha=0.1, max_iter=1000)
ridge_model.fit(X_train, y_train)
ridge_score = -2 * ridge_model.score(X_test, y_test)

# Lasso Regression (L1 regularization - feature selection)
lasso_model = PoissonRegressor(alpha=0.05, max_iter=1000)
lasso_model.fit(X_train, y_train)
lasso_score = -2 * lasso_model.score(X_test, y_test)

print(f"\nRidge (L2) Deviance (test): {ridge_score:.4f}")
print(f"Lasso (L1) Deviance (test): {lasso_score:.4f}")
print(f"Baseline Deviance (test): {-2 * poisson_model.score(X_test, y_test):.4f}")

# ============================================================================
# 6. CROSS-VALIDATION
# ============================================================================

print("\n" + "=" * 80)
print("K-FOLD CROSS-VALIDATION")
print("=" * 80)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Baseline model CV
baseline_cv_scores = -cross_val_score(PoissonRegressor(alpha=0, max_iter=1000), 
                                       X_model, y, cv=kfold, scoring='neg_mean_poisson_deviance')

# Ridge CV
ridge_cv_scores = -cross_val_score(PoissonRegressor(alpha=0.1, max_iter=1000), 
                                    X_model, y, cv=kfold, scoring='neg_mean_poisson_deviance')

# Lasso CV
lasso_cv_scores = -cross_val_score(PoissonRegressor(alpha=0.05, max_iter=1000), 
                                    X_model, y, cv=kfold, scoring='neg_mean_poisson_deviance')

print(f"\nBaseline Poisson GLM:")
print(f"  Mean CV Score: {baseline_cv_scores.mean():.4f} (+/- {baseline_cv_scores.std():.4f})")
print(f"  Fold scores: {baseline_cv_scores}")

print(f"\nRidge Regularization (α=0.1):")
print(f"  Mean CV Score: {ridge_cv_scores.mean():.4f} (+/- {ridge_cv_scores.std():.4f})")
print(f"  Fold scores: {ridge_cv_scores}")

print(f"\nLasso Regularization (α=0.05):")
print(f"  Mean CV Score: {lasso_cv_scores.mean():.4f} (+/- {lasso_cv_scores.std():.4f})")
print(f"  Fold scores: {lasso_cv_scores}")

# ============================================================================
# 7. HYPERPARAMETER TUNING
# ============================================================================

print("\n" + "=" * 80)
print("HYPERPARAMETER TUNING")
print("=" * 80)

alpha_values = np.logspace(-4, 0, 10)
cv_scores_by_alpha = []

for alpha in alpha_values:
    model = PoissonRegressor(alpha=alpha, max_iter=1000)
    scores = -cross_val_score(model, X_model, y, cv=5, scoring='neg_mean_poisson_deviance')
    cv_scores_by_alpha.append(scores.mean())

best_alpha_idx = np.argmin(cv_scores_by_alpha)
best_alpha = alpha_values[best_alpha_idx]

print(f"\nOptimal alpha (L2): {best_alpha:.6f}")
print(f"Optimal CV score: {cv_scores_by_alpha[best_alpha_idx]:.4f}")

# Fit final model with optimal alpha
final_model = PoissonRegressor(alpha=best_alpha, max_iter=1000)
final_model.fit(X_train, y_train)

print(f"\nFinal Model Performance:")
print(f"  Training deviance: {-2 * final_model.score(X_train, y_train):.4f}")
print(f"  Testing deviance: {-2 * final_model.score(X_test, y_test):.4f}")
print(f"  MAE: {mean_absolute_error(y_test, final_model.predict(X_test)):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, final_model.predict(X_test))):.4f}")

# ============================================================================
# 8. PREDICTIONS & RISK SCORING
# ============================================================================

print("\n" + "=" * 80)
print("PREDICTIONS & RISK SCORING")
print("=" * 80)

# Generate predictions on full dataset
y_pred_baseline = poisson_model.predict(X_model)
y_pred_final = final_model.predict(X_model)

# Add predictions to original data
company_freq['predicted_frequency_baseline'] = y_pred_baseline
company_freq['predicted_frequency_final'] = y_pred_final
company_freq['residuals'] = company_freq['incident_count'] - y_pred_final
company_freq['standardized_residuals'] = company_freq['residuals'] / np.sqrt(y_pred_final + 1)

# Risk score (predicted frequency normalized)
company_freq['risk_score'] = (y_pred_final / y_pred_final.mean()) * 100

print(f"\nPrediction Statistics (Final Model):")
print(f"  Mean predicted frequency: {y_pred_final.mean():.4f}")
print(f"  Std Dev: {y_pred_final.std():.4f}")
print(f"  Min: {y_pred_final.min():.4f}")
print(f"  Max: {y_pred_final.max():.4f}")

print(f"\nTop 10 Highest Risk Companies (by predicted frequency):")
top_risk = company_freq.nlargest(10, 'predicted_frequency_final')[
    ['company_name', 'incident_count', 'predicted_frequency_final', 'risk_score', 'industry_primary']
]
print(top_risk.to_string(index=False))

# ============================================================================
# 9. MODEL DIAGNOSTICS & VISUALIZATION
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Actual vs Predicted
ax1 = axes[0, 0]
ax1.scatter(company_freq['incident_count'], company_freq['predicted_frequency_final'], alpha=0.6)
ax1.plot([0, company_freq['incident_count'].max()], [0, company_freq['incident_count'].max()], 'r--')
ax1.set_xlabel('Actual Incident Frequency')
ax1.set_ylabel('Predicted Incident Frequency')
ax1.set_title('Actual vs Predicted Frequency')
ax1.grid(True, alpha=0.3)

# Plot 2: Residuals
ax2 = axes[0, 1]
ax2.scatter(company_freq['predicted_frequency_final'], company_freq['standardized_residuals'], alpha=0.6)
ax2.axhline(y=0, color='r', linestyle='--')
ax2.set_xlabel('Predicted Frequency')
ax2.set_ylabel('Standardized Residuals')
ax2.set_title('Residual Plot')
ax2.grid(True, alpha=0.3)

# Plot 3: Hyperparameter tuning
ax3 = axes[1, 0]
ax3.plot(alpha_values, cv_scores_by_alpha, 'b-o', linewidth=2)
ax3.set_xscale('log')
ax3.set_xlabel('Alpha (Regularization Parameter)')
ax3.set_ylabel('CV Deviance Score')
ax3.set_title('Hyperparameter Tuning')
ax3.grid(True, alpha=0.3)

# Plot 4: Frequency distribution
ax4 = axes[1, 1]
ax4.hist(company_freq['incident_count'], bins=20, alpha=0.7, label='Actual', edgecolor='black')
ax4.hist(company_freq['predicted_frequency_final'], bins=20, alpha=0.7, label='Predicted', edgecolor='black')
ax4.set_xlabel('Incident Frequency')
ax4.set_ylabel('Count')
ax4.set_title('Frequency Distribution')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/poisson_model_diagnostics.png', dpi=300, bbox_inches='tight')
print("Saved: poisson_model_diagnostics.png")

# ============================================================================
# 10. EXPORT RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("EXPORTING RESULTS")
print("=" * 80)

# Export company-level results
company_results = company_freq[[
    'company_name', 'incident_count', 'company_revenue_usd', 'employee_count',
    'industry_primary', 'is_public_company', 'predicted_frequency_baseline',
    'predicted_frequency_final', 'risk_score'
]].copy()
company_results = company_results.sort_values('risk_score', ascending=False)
company_results.to_csv('/mnt/user-data/outputs/company_frequency_predictions.csv', index=False)
print("Saved: company_frequency_predictions.csv")

# Export industry-level aggregations
industry_results = industry_freq.copy()
industry_results['incident_rate_per_1000'] = (industry_results['incident_count'] / len(incidents) * 1000)
industry_results = industry_results.sort_values('incident_count', ascending=False)
industry_results.to_csv('/mnt/user-data/outputs/industry_frequency_analysis.csv', index=False)
print("Saved: industry_frequency_analysis.csv")

# Export model coefficients
coef_df = pd.DataFrame({
    'Feature': feature_cols + ['Intercept'],
    'Baseline_Coefficient': list(poisson_model.coef_) + [poisson_model.intercept_],
    'Final_Model_Coefficient': list(final_model.coef_) + [final_model.intercept_],
    'Ridge_Coefficient': list(ridge_model.coef_) + [ridge_model.intercept_],
    'Lasso_Coefficient': list(lasso_model.coef_) + [lasso_model.intercept_]
})
coef_df.to_csv('/mnt/user-data/outputs/model_coefficients.csv', index=False)
print("Saved: model_coefficients.csv")

# ============================================================================
# 11. SUMMARY STATISTICS
# ============================================================================

print("\n" + "=" * 80)
print("MODEL SUMMARY")
print("=" * 80)

summary_stats = {
    'Total Companies': len(company_freq),
    'Total Incidents': y.sum(),
    'Training Samples': len(X_train),
    'Testing Samples': len(X_test),
    'Mean Frequency': y.mean(),
    'Std Dev Frequency': y.std(),
    'Baseline Test Deviance': -2 * poisson_model.score(X_test, y_test),
    'Final Model Test Deviance': -2 * final_model.score(X_test, y_test),
    'Optimal Alpha': best_alpha,
    'Final Model MAE': mean_absolute_error(y_test, final_model.predict(X_test)),
    'Final Model RMSE': np.sqrt(mean_squared_error(y_test, final_model.predict(X_test)))
}

for key, value in summary_stats.items():
    print(f"{key}: {value}")

print("\n" + "=" * 80)
print("MODEL DEVELOPMENT COMPLETE")
print("=" * 80)
print("\nNext Steps:")
print("1. Validate frequency model with domain experts")
print("2. Develop severity model (loss distribution)")
print("3. Calculate pure premium = E[frequency] × E[severity]")
print("4. Apply loading factors for expenses, profit, uncertainty")
print("5. Establish pricing by class and risk profile")
print("\nFiles generated:")
print("  - poisson_frequency_model.py (this script)")
print("  - poisson_model_diagnostics.png (model visualizations)")
print("  - company_frequency_predictions.csv (predictions by company)")
print("  - industry_frequency_analysis.csv (industry aggregation)")
print("  - model_coefficients.csv (model parameters)")
