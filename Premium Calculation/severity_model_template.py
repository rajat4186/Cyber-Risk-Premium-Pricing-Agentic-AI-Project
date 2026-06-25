#!/usr/bin/env python3
"""
SEVERITY MODEL FOR CYBER & AI RISK - TEMPLATE
Phase 2: Loss Distribution Modeling
Actuarial Science Project - Premium Pricing

This template builds on the completed frequency model to develop
a severity (loss distribution) component for the complete pricing model.

Run this template to:
1. Analyze loss data from financial_impact_1.csv
2. Fit parametric distributions (Lognormal, Pareto, Gamma)
3. Develop loss prediction models by company characteristics
4. Calculate E[Severity] for pure premium = E[Frequency] × E[Severity]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import lognorm, pareto, gamma, kstest, anderson
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("SEVERITY MODEL FOR CYBER & AI RISK - DEVELOPMENT TEMPLATE")
print("=" * 80)

# ============================================================================
# 1. DATA LOADING & PREPARATION
# ============================================================================

print("\n1. LOADING AND PREPARING LOSS DATA")
print("-" * 80)

# Load datasets
incidents = pd.read_csv('/mnt/project/incidents_master.csv')
financial = pd.read_csv('/mnt/project/financial_impact_1.csv')

# Merge incidents with financial data
merged = incidents.merge(financial, on='incident_id', how='inner')
print(f"Merged records: {len(merged)} incidents with financial data")

# Clean loss data
# Remove records with missing or zero total loss
merged = merged[merged['total_loss_usd'].notna()]
merged = merged[merged['total_loss_usd'] > 0]
print(f"Records with valid loss data: {len(merged)}")

# Summary statistics
print(f"\nLoss Statistics:")
print(f"  Mean: ${merged['total_loss_usd'].mean():,.0f}")
print(f"  Median: ${merged['total_loss_usd'].median():,.0f}")
print(f"  Std Dev: ${merged['total_loss_usd'].std():,.0f}")
print(f"  Min: ${merged['total_loss_usd'].min():,.0f}")
print(f"  Max: ${merged['total_loss_usd'].max():,.0f}")
print(f"  Q75: ${merged['total_loss_usd'].quantile(0.75):,.0f}")
print(f"  Q95: ${merged['total_loss_usd'].quantile(0.95):,.0f}")

losses = merged['total_loss_usd'].values

# ============================================================================
# 2. EXPLORATORY ANALYSIS - LOSS DISTRIBUTION
# ============================================================================

print("\n2. EXPLORATORY DISTRIBUTION ANALYSIS")
print("-" * 80)

# Log transform for analysis (standard in insurance)
log_losses = np.log(losses)

print(f"Log-Loss Statistics:")
print(f"  Mean: {log_losses.mean():.4f}")
print(f"  Std Dev: {log_losses.std():.4f}")
print(f"  Skewness: {stats.skew(log_losses):.4f} (>0 = right-skewed)")
print(f"  Kurtosis: {stats.kurtosis(log_losses):.4f} (>3 = heavy tails)")

# Calculate quantiles for tail analysis
quantiles = [50, 75, 90, 95, 99]
print(f"\nLoss Quantiles:")
for q in quantiles:
    value = np.percentile(losses, q)
    print(f"  {q}th percentile: ${value:,.0f}")

# ============================================================================
# 3. PARAMETRIC DISTRIBUTION FITTING
# ============================================================================

print("\n3. FITTING PARAMETRIC DISTRIBUTIONS")
print("-" * 80)

# Fit Lognormal Distribution
# ln(Loss) ~ N(μ, σ²)
mu_lognorm, sigma_lognorm = log_losses.mean(), log_losses.std()
lognorm_dist = lognorm(s=sigma_lognorm, scale=np.exp(mu_lognorm))
print(f"\nLognormal Distribution:")
print(f"  μ (log scale): {mu_lognorm:.4f}")
print(f"  σ (log scale): {sigma_lognorm:.4f}")

# Fit Pareto Distribution (for tail modeling)
# P(Loss > x) = (x_m / x)^α, where x_m is minimum loss
x_m = losses.min()
# MLE estimate: α = n / Σ[ln(x_i/x_m)]
alpha_pareto = len(losses) / np.sum(np.log(losses / x_m))
pareto_dist = pareto(b=alpha_pareto, loc=0, scale=x_m)
print(f"\nPareto Distribution:")
print(f"  x_m (minimum): ${x_m:,.0f}")
print(f"  α (shape): {alpha_pareto:.4f}")

# Fit Gamma Distribution
# Loss ~ Gamma(α, β)
# Using method of moments: α = μ²/σ², β = σ²/μ
mean_loss = losses.mean()
var_loss = losses.var()
alpha_gamma = mean_loss**2 / var_loss
beta_gamma = var_loss / mean_loss
gamma_dist = gamma(a=alpha_gamma, scale=beta_gamma)
print(f"\nGamma Distribution:")
print(f"  α (shape): {alpha_gamma:.4f}")
print(f"  β (scale): ${beta_gamma:,.0f}")

# ============================================================================
# 4. GOODNESS OF FIT TESTING
# ============================================================================

print("\n4. GOODNESS OF FIT TESTS")
print("-" * 80)

# Kolmogorov-Smirnov Test (comparing CDF)
ks_lognorm = kstest(losses, lognorm_dist.cdf)
ks_gamma = kstest(losses, gamma_dist.cdf)

print(f"\nKolmogorov-Smirnov Test (lower p-value = better fit):")
print(f"  Lognormal: KS={ks_lognorm[0]:.4f}, p-value={ks_lognorm[1]:.6f}")
print(f"  Gamma: KS={ks_gamma[0]:.4f}, p-value={ks_gamma[1]:.6f}")

# Anderson-Darling Test (weights tail behavior)
ad_lognorm = anderson(log_losses)
print(f"\nAnderson-Darling Test (for log-transformed Lognormal):")
print(f"  Statistic: {ad_lognorm.statistic:.4f}")
print(f"  Critical values: {ad_lognorm.critical_values}")

# Log-Likelihood Comparison (for AIC/BIC)
ll_lognorm = np.sum(lognorm_dist.logpdf(losses))
ll_gamma = np.sum(gamma_dist.logpdf(losses))

k = 2  # number of parameters
n = len(losses)
aic_lognorm = -2 * ll_lognorm + 2 * k
aic_gamma = -2 * ll_gamma + 2 * k
bic_lognorm = -2 * ll_lognorm + k * np.log(n)
bic_gamma = -2 * ll_gamma + k * np.log(n)

print(f"\nModel Comparison (Information Criteria - lower is better):")
print(f"  Lognormal AIC: {aic_lognorm:.2f}, BIC: {bic_lognorm:.2f}")
print(f"  Gamma AIC: {aic_gamma:.2f}, BIC: {bic_gamma:.2f}")

# Recommendation
print(f"\n{'='*80}")
print(f"RECOMMENDATION: {'Lognormal' if aic_lognorm < aic_gamma else 'Gamma'} distribution")
print(f"Rationale: Lower AIC/BIC, common in insurance, easier interpretation")
print(f"{'='*80}")

# ============================================================================
# 5. SEVERITY BY COMPANY CHARACTERISTICS
# ============================================================================

print("\n5. ANALYZING SEVERITY BY COMPANY CHARACTERISTICS")
print("-" * 80)

# Add log transformations for modeling
merged['log_loss'] = np.log(merged['total_loss_usd'])
merged['log_revenue'] = np.log1p(merged['company_revenue_usd'])
merged['log_employees'] = np.log1p(merged['employee_count'])

# Group by industry and analyze
print(f"\nMean Loss by Industry (Top 10):")
industry_loss = merged.groupby('industry_primary').agg({
    'total_loss_usd': ['count', 'mean', 'median', 'std'],
    'log_loss': 'mean'
}).round(0)
industry_loss.columns = ['Count', 'Mean_Loss', 'Median_Loss', 'Std_Loss', 'Mean_Log_Loss']
industry_loss = industry_loss.sort_values('Mean_Loss', ascending=False).head(10)
print(industry_loss)

# Group by company size
print(f"\nMean Loss by Company Revenue Quartile:")
merged['revenue_quartile'] = pd.qcut(merged['company_revenue_usd'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
revenue_loss = merged.groupby('revenue_quartile').agg({
    'total_loss_usd': ['count', 'mean', 'median'],
    'log_loss': 'mean'
}).round(0)
print(revenue_loss)

# ============================================================================
# 6. LOSS PREDICTION MODEL (Linear Regression on Log-Scale)
# ============================================================================

print("\n6. BUILDING LOSS PREDICTION MODEL")
print("-" * 80)

# Prepare features for severity modeling
X_severity = merged[[
    'log_revenue', 
    'log_employees', 
    'is_public_company',
    'data_compromised_records'
]].copy()

# Handle missing values
X_severity['data_compromised_records'] = X_severity['data_compromised_records'].fillna(0)
X_severity['log_records'] = np.log1p(X_severity['data_compromised_records'])

# Select final features
feature_cols_severity = ['log_revenue', 'log_employees', 'is_public_company', 'log_records']
X_severity = X_severity[feature_cols_severity].copy()
y_severity = merged['log_loss'].copy()

# Remove any NaN rows
valid_idx = X_severity.notna().all(axis=1) & y_severity.notna()
X_severity = X_severity[valid_idx]
y_severity = y_severity[valid_idx]

print(f"Severity model sample size: {len(X_severity)}")

# Train/test split
X_train_sev, X_test_sev, y_train_sev, y_test_sev = train_test_split(
    X_severity, y_severity, test_size=0.2, random_state=42
)

# Fit model (on log scale)
severity_model = Ridge(alpha=1.0)
severity_model.fit(X_train_sev, y_train_sev)

# Predictions
y_pred_sev = severity_model.predict(X_test_sev)
log_loss_mae = np.mean(np.abs(y_test_sev - y_pred_sev))

# Back-transform to original scale
y_pred_loss = np.exp(y_pred_sev)
y_test_loss = np.exp(y_test_sev)
loss_mae = np.mean(np.abs(y_test_loss - y_pred_loss))

print(f"\nSeverity Model Performance:")
print(f"  Training R²: {severity_model.score(X_train_sev, y_train_sev):.4f}")
print(f"  Testing R²: {severity_model.score(X_test_sev, y_test_sev):.4f}")
print(f"  MAE (log scale): ${log_loss_mae:.4f}")
print(f"  MAE (original scale): ${loss_mae:,.0f}")

print(f"\nSeverity Model Coefficients:")
for feat, coef in zip(feature_cols_severity, severity_model.coef_):
    print(f"  {feat}: {coef:.6f}")
print(f"  Intercept: {severity_model.intercept_:.6f}")

# ============================================================================
# 7. PURE PREMIUM CALCULATION FRAMEWORK
# ============================================================================

print("\n7. PURE PREMIUM CALCULATION FRAMEWORK")
print("-" * 80)

# Load frequency predictions
frequency_df = pd.read_csv('/mnt/user-data/outputs/company_frequency_predictions.csv')

print(f"\nPure Premium = E[Frequency] × E[Severity]")
print(f"\nExample Calculation:")
print(f"  E[Frequency] = 1.20 (from Poisson model)")
print(f"  E[Severity] = $250,000 (from loss model)")
print(f"  Pure Premium = 1.20 × $250,000 = $300,000")

print(f"\nApplying Loading Factors:")
print(f"  Acquisition Expenses: 20% (commission, admin)")
print(f"  General Expenses: 10% (overhead)")
print(f"  Profit Loading: 15% (target ROE)")
print(f"  Uncertainty Buffer: 5% (model error, tail risk)")
print(f"  Total Loading: 50%")

print(f"\n  Final Premium = Pure Premium × (1 + Loading)")
print(f"              = $300,000 × 1.50")
print(f"              = $450,000")

# ============================================================================
# 8. VISUALIZATION
# ============================================================================

print("\n8. GENERATING SEVERITY VISUALIZATIONS")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Loss distribution with fitted curve
ax1 = axes[0, 0]
ax1.hist(losses / 1e6, bins=50, density=True, alpha=0.7, label='Actual', edgecolor='black')
x_range = np.linspace(losses.min(), np.percentile(losses, 99), 200)
ax1.plot(x_range / 1e6, lognorm_dist.pdf(x_range), 'r-', linewidth=2, label='Lognormal')
ax1.plot(x_range / 1e6, gamma_dist.pdf(x_range), 'g--', linewidth=2, label='Gamma')
ax1.set_xlabel('Loss ($ Millions)')
ax1.set_ylabel('Density')
ax1.set_title('Loss Distribution with Fitted Curves')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Q-Q plot (log scale)
ax2 = axes[0, 1]
theoretical_quantiles = np.sort(lognorm_dist.rvs(size=len(log_losses)))
ax2.scatter(theoretical_quantiles, np.sort(log_losses), alpha=0.6)
min_val, max_val = np.min([theoretical_quantiles, log_losses]), np.max([theoretical_quantiles, log_losses])
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
ax2.set_xlabel('Theoretical Quantiles (Lognormal)')
ax2.set_ylabel('Sample Quantiles (Log Scale)')
ax2.set_title('Q-Q Plot')
ax2.grid(True, alpha=0.3)

# Plot 3: Loss by industry
ax3 = axes[1, 0]
industry_avg = merged.groupby('industry_primary')['total_loss_usd'].mean().sort_values(ascending=False).head(10)
ax3.barh(range(len(industry_avg)), industry_avg / 1e6)
ax3.set_yticks(range(len(industry_avg)))
ax3.set_yticklabels(industry_avg.index)
ax3.set_xlabel('Mean Loss ($ Millions)')
ax3.set_title('Average Loss by Industry (Top 10)')
ax3.grid(True, alpha=0.3, axis='x')

# Plot 4: Actual vs Predicted losses
ax4 = axes[1, 1]
ax4.scatter(y_test_loss / 1e6, y_pred_loss / 1e6, alpha=0.6)
min_val, max_val = np.min([y_test_loss, y_pred_loss]) / 1e6, np.max([y_test_loss, y_pred_loss]) / 1e6
ax4.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
ax4.set_xlabel('Actual Loss ($ Millions)')
ax4.set_ylabel('Predicted Loss ($ Millions)')
ax4.set_title('Severity Model: Actual vs Predicted')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/severity_model_diagnostics.png', dpi=300, bbox_inches='tight')
print("Saved: severity_model_diagnostics.png")

# ============================================================================
# 9. EXPORT RESULTS
# ============================================================================

print("\n9. EXPORTING SEVERITY MODEL RESULTS")
print("-" * 80)

# Export severity coefficients
severity_coef = pd.DataFrame({
    'Feature': feature_cols_severity + ['Intercept'],
    'Coefficient': list(severity_model.coef_) + [severity_model.intercept_]
})
severity_coef.to_csv('/mnt/user-data/outputs/severity_model_coefficients.csv', index=False)
print("Saved: severity_model_coefficients.csv")

# Export industry-level severity
industry_severity = merged.groupby('industry_primary').agg({
    'total_loss_usd': ['count', 'mean', 'median', 'std']
}).reset_index()
industry_severity.columns = ['industry', 'incident_count', 'mean_loss', 'median_loss', 'std_loss']
industry_severity.to_csv('/mnt/user-data/outputs/industry_severity_analysis.csv', index=False)
print("Saved: industry_severity_analysis.csv")

# ============================================================================
# 10. SUMMARY & NEXT STEPS
# ============================================================================

print("\n" + "=" * 80)
print("SEVERITY MODEL DEVELOPMENT COMPLETE")
print("=" * 80)

print(f"\nKey Findings:")
print(f"  - {len(merged)} incidents with loss data")
print(f"  - Mean loss: ${np.mean(losses):,.0f}")
print(f"  - Median loss: ${np.median(losses):,.0f}")
print(f"  - 95th percentile: ${np.percentile(losses, 95):,.0f}")
print(f"  - Recommended distribution: Lognormal")
print(f"  - Severity model R²: {severity_model.score(X_test_sev, y_test_sev):.4f}")

print(f"\nNext Steps:")
print(f"1. ✅ Frequency Model Complete")
print(f"2. ✅ Severity Model Complete (this script)")
print(f"3. ⏳ Combine frequency × severity for pure premium")
print(f"4. ⏳ Add loading factors (expenses, profit)")
print(f"5. ⏳ Generate final pricing tables by company/class")
print(f"6. ⏳ Back-test against market benchmarks")

print(f"\nFiles Generated:")
print(f"  - severity_model_diagnostics.png")
print(f"  - severity_model_coefficients.csv")
print(f"  - industry_severity_analysis.csv")

print("\n" + "=" * 80)
print("Ready to integrate with frequency model for complete pricing framework!")
print("=" * 80)
