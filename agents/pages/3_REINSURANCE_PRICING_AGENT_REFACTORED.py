# -*- coding: utf-8 -*-
"""
Cyber & AI Risk Reinsurance Quotation Agent (Refactored)
============================================================

ARCHITECTURE:
- Uses SAME frequency (Poisson GLM) & severity (Lognormal) models as Insurance Agent
- Dynamically calculates industry relativities from actual pricing data
- Reinsurance layers derived from insurance premium output
- No unit conversion errors; clean rate-based calculations

Project: Cyber & AI Risk Insurance Pricing (Actuarial)
Framework: Agno + Google Gemini
Data Source: 750 companies, 20 industries
Status: Production Ready
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Any, Dict
from dataclasses import dataclass
from datetime import datetime
import streamlit as st

np.random.seed(42)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 120)

# ============================================================================
# PHASE 1: LOAD & TRAIN FREQUENCY & SEVERITY MODELS (Aligned with Insurance Agent)
# ============================================================================

print("\n[PHASE 1] Loading incident & financial data from GitHub...")

try:
    incidents = pd.read_csv(
        "https://raw.githubusercontent.com/rajat4186/Cyber-Risk-Premium-Pricing-Agentic-AI-Project/refs/heads/main/data/incidents_master_cleaned.csv"
    )
    financial = pd.read_csv(
        "https://raw.githubusercontent.com/rajat4186/Cyber-Risk-Premium-Pricing-Agentic-AI-Project/refs/heads/main/data/financial_impact_cleaned.csv"
    )
    
    print(f"✓ Loaded incidents: {len(incidents)} records")
    print(f"✓ Loaded financial: {len(financial)} records")
    DATA_AVAILABLE = True
except Exception as e:
    print(f"⚠️  Data loading error: {e}")
    DATA_AVAILABLE = False
    incidents = None
    financial = None

# ============================================================================
# PHASE 1.5: TRAIN POISSON GLM FREQUENCY MODEL
# ============================================================================

print("\n[PHASE 1.5] Training Poisson GLM Frequency Model...")

def train_frequency_model(incidents_df):
    """Train Poisson GLM frequency model (same as Insurance Agent)"""
    try:
        from sklearn.linear_model import PoissonRegressor
        from sklearn.model_selection import train_test_split
        
        company_freq = (
            incidents_df.groupby("company_name")
            .agg({
                "incident_id": "count",
                "company_revenue_usd": "first",
                "employee_count": "first",
                "is_public_company": "first",
                "naics_code": "first",
            })
            .reset_index()
        )
        company_freq.rename(columns={"incident_id": "incident_count"}, inplace=True)
        company_freq = company_freq.dropna(subset=["incident_count", "company_revenue_usd"])

        X_freq = company_freq[["company_revenue_usd", "employee_count", "is_public_company"]].copy()
        X_freq["log_revenue"] = np.log1p(X_freq["company_revenue_usd"])
        X_freq["log_employees"] = np.log1p(X_freq["employee_count"])
        X_freq["revenue_tier"] = pd.cut(
            X_freq["company_revenue_usd"],
            bins=[0, 1e9, 10e9, 100e9, np.inf],
            labels=[0, 1, 2, 3],
        ).astype(int)

        feature_cols = ["log_revenue", "log_employees", "is_public_company", "revenue_tier"]
        X_freq_model = X_freq[feature_cols].fillna(0)
        y_freq = company_freq["incident_count"].copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X_freq_model, y_freq, test_size=0.2, random_state=42
        )

        freq_model = PoissonRegressor(alpha=0.1292, max_iter=1000)
        freq_model.fit(X_train, y_train)

        coefficients = {
            "intercept": float(freq_model.intercept_),
            "log_revenue": float(freq_model.coef_[0]),
            "log_employees": float(freq_model.coef_[1]),
            "is_public": float(freq_model.coef_[2]),
            "revenue_tier": float(freq_model.coef_[3]),
        }

        train_score = freq_model.score(X_train, y_train)
        test_score = freq_model.score(X_test, y_test)

        print(f"✓ Frequency Model Trained")
        print(f"  • Training R²: {train_score:.4f}")
        print(f"  • Testing R²: {test_score:.4f}")

        return coefficients

    except Exception as e:
        print(f"⚠️  Error training frequency model: {e}")
        return None

if DATA_AVAILABLE and incidents is not None:
    FREQ_COEFFICIENTS = train_frequency_model(incidents)
else:
    print("⚠️  Using fallback frequency coefficients")
    FREQ_COEFFICIENTS = {
        "intercept": -1.4502,
        "log_revenue": 0.0841,
        "log_employees": 0.0512,
        "is_public": 0.1292,
        "revenue_tier": 0.2145,
    }

# ============================================================================
# PHASE 2: TRAIN LOGNORMAL SEVERITY MODEL
# ============================================================================

print("\n[PHASE 2] Training Lognormal Severity Model...")

def train_severity_model(incidents_df, financial_df):
    """Train Lognormal severity model (same as Insurance Agent)"""
    try:
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import train_test_split
        
        losses = financial_df["total_loss_usd"].values
        log_losses = np.log(losses)

        mu = float(log_losses.mean())
        sigma = float(log_losses.std())

        print(f"  • Loss records: {len(losses)}")
        print(f"  • Mean loss: ${np.mean(losses) / 1e6:.1f}M")
        print(f"  • Lognormal: μ={mu:.4f}, σ={sigma:.4f}")

        merged = incidents_df.merge(financial_df, on="incident_id", how="inner")

        if len(merged) == 0:
            raise Exception("No merged data between incidents and financial")

        X_sev = merged[["company_revenue_usd", "employee_count", "is_public_company"]].copy()
        X_sev["log_revenue"] = np.log1p(X_sev["company_revenue_usd"])
        X_sev["log_employees"] = np.log1p(X_sev["employee_count"])
        X_sev["log_records"] = np.log1p(1000000)

        feature_cols_sev = ["log_revenue", "log_employees", "is_public_company", "log_records"]
        X_sev_model = X_sev[feature_cols_sev].fillna(0)
        y_sev = np.log(merged["total_loss_usd"])

        X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
            X_sev_model, y_sev, test_size=0.2, random_state=42
        )

        sev_model = Ridge(alpha=1.0)
        sev_model.fit(X_train_s, y_train_s)

        coefficients = {
            "intercept": float(sev_model.intercept_),
            "log_revenue": float(sev_model.coef_[0]),
            "log_employees": float(sev_model.coef_[1]),
            "is_public": float(sev_model.coef_[2]),
            "log_records": float(sev_model.coef_[3]),
        }

        test_score = sev_model.score(X_test_s, y_test_s)

        print(f"✓ Severity Model Trained")
        print(f"  • Testing R²: {test_score:.4f}")

        lognorm_params = {"mu": mu, "sigma": sigma}

        return coefficients, lognorm_params

    except Exception as e:
        print(f"⚠️  Error training severity model: {e}")
        return None, None

if DATA_AVAILABLE and incidents is not None and financial is not None:
    SEVER_COEFFICIENTS, LOGNORM_PARAMS = train_severity_model(incidents, financial)
else:
    print("⚠️  Using fallback severity coefficients")
    SEVER_COEFFICIENTS = {
        "intercept": 14.8504,
        "log_revenue": 0.2841,
        "log_employees": 0.1512,
        "is_public": 0.2292,
        "log_records": 0.0845,
    }
    LOGNORM_PARAMS = {"mu": 14.8504, "sigma": 1.2842}

# ============================================================================
# PHASE 3: DYNAMICALLY CALCULATE INDUSTRY RELATIVITIES FROM DATA
# ============================================================================

print("\n[PHASE 3] Calculating dynamic industry relativities from pricing data...")

def calculate_industry_relativities(incidents_df, financial_df):
    """
    Calculate industry relativity factors from actual premium data.
    Relativity = (Industry Mean Premium) / (Overall Mean Premium)
    """
    try:
        merged = incidents_df.merge(financial_df, on="incident_id", how="inner")
        
        if "naics_code" not in merged.columns:
            print("⚠️  NAICS code not in merged data; using fallback relativities")
            return None
        
        # Group by industry and calculate mean loss (proxy for premium)
        industry_losses = merged.groupby("naics_code")["total_loss_usd"].agg(["mean", "count"]).reset_index()
        industry_losses.columns = ["naics_code", "mean_loss", "incident_count"]
        
        overall_mean_loss = merged["total_loss_usd"].mean()
        
        # Relativity = industry_mean / overall_mean
        industry_losses["relativity"] = industry_losses["mean_loss"] / overall_mean_loss
        
        relativities_dict = {}
        for _, row in industry_losses.iterrows():
            naics = str(row["naics_code"]).strip()
            relativities_dict[naics] = {
                "relativity": round(float(row["relativity"]), 3),
                "count": int(row["incident_count"]),
                "mean_loss": float(row["mean_loss"])
            }
        
        print(f"✓ Calculated relativities for {len(relativities_dict)} industries")
        return relativities_dict
    
    except Exception as e:
        print(f"⚠️  Error calculating relativities: {e}")
        return None

if DATA_AVAILABLE and incidents is not None and financial is not None:
    INDUSTRY_RELATIVITIES_DYNAMIC = calculate_industry_relativities(incidents, financial)
else:
    INDUSTRY_RELATIVITIES_DYNAMIC = None

# Fallback hardcoded relativities if dynamic calculation failed
INDUSTRY_RELATIVITIES_FALLBACK = {
    '55': {'name': 'Management of Companies', 'relativity': 1.417, 'count': 2},
    '44-45': {'name': 'Retail Trade', 'relativity': 1.264, 'count': 65},
    '51': {'name': 'Information & Technology', 'relativity': 1.231, 'count': 102},
    '92': {'name': 'Public Administration', 'relativity': 1.221, 'count': 52},
    '52': {'name': 'Finance & Insurance', 'relativity': 1.181, 'count': 106},
    '31-33': {'name': 'Manufacturing', 'relativity': 1.023, 'count': 67},
    '21': {'name': 'Mining, Quarrying, Oil & Gas', 'relativity': 0.955, 'count': 21},
    '62': {'name': 'Health Care & Social Assistance', 'relativity': 0.914, 'count': 117},
    '22': {'name': 'Utilities', 'relativity': 0.885, 'count': 38},
    '42': {'name': 'Wholesale Trade', 'relativity': 0.804, 'count': 10},
}

# Use dynamic if available, otherwise fallback
INDUSTRY_RELATIVITIES = INDUSTRY_RELATIVITIES_DYNAMIC if INDUSTRY_RELATIVITIES_DYNAMIC else INDUSTRY_RELATIVITIES_FALLBACK

print(f"✓ Using {'dynamic' if INDUSTRY_RELATIVITIES_DYNAMIC else 'fallback'} industry relativities")

# ============================================================================
# PHASE 4: ACTUARIAL CONSTANTS (Loading Factors)
# ============================================================================

@dataclass
class ActuarialConstants:
    # Loading Factors (58% total) - aligned with Insurance Agent
    acquisition_expenses_pct: float = 0.20
    admin_expenses_pct: float = 0.10
    profit_margin_pct: float = 0.15
    uncertainty_buffer_pct: float = 0.08
    reinsurance_ceded_pct: float = 0.05
    total_loading_pct: float = 0.58

    # Proportional Reinsurance
    quota_share_pct: float = 0.30
    quota_share_commission: float = 0.30
    surplus_share_pct: float = 0.15
    surplus_commission: float = 0.30

    # XOL Parameters
    xol_profit_loading: float = 1.50

ACTUARIAL = ActuarialConstants()

print("✓ Actuarial constants loaded (aligned with Insurance Agent)")

# ============================================================================
# PHASE 5: PREDICTION FUNCTIONS (Same as Insurance Agent)
# ============================================================================

def predict_frequency(company_revenue: float, employee_count: int, is_public: bool) -> dict:
    """Predict frequency using Poisson GLM with extracted coefficients"""
    log_revenue = np.log1p(company_revenue)
    log_employees = np.log1p(employee_count)
    is_public_numeric = 1 if is_public else 0

    if company_revenue < 1e9:
        size_category = 0
    elif company_revenue < 10e9:
        size_category = 1
    elif company_revenue < 100e9:
        size_category = 2
    else:
        size_category = 3

    log_lambda = (
        FREQ_COEFFICIENTS["intercept"]
        + FREQ_COEFFICIENTS["log_revenue"] * log_revenue
        + FREQ_COEFFICIENTS["log_employees"] * log_employees
        + FREQ_COEFFICIENTS["is_public"] * is_public_numeric
        + FREQ_COEFFICIENTS["revenue_tier"] * size_category
    )

    predicted_frequency = np.exp(log_lambda)
    risk_score = (predicted_frequency / 1.1354) * 100

    return {
        "predicted_frequency": round(predicted_frequency, 4),
        "risk_score": round(risk_score, 1),
        "size_category": ["Small", "Medium", "Large", "Enterprise"][size_category],
    }

def predict_severity(
    company_revenue: float,
    employee_count: int,
    is_public: bool,
    records_at_risk: int = 1000000,
) -> dict:
    """Predict severity using Lognormal with extracted coefficients"""
    log_revenue = np.log1p(company_revenue)
    log_employees = np.log1p(employee_count)
    is_public_numeric = 1 if is_public else 0
    log_records = np.log1p(records_at_risk)

    log_loss = (
        SEVER_COEFFICIENTS["intercept"]
        + SEVER_COEFFICIENTS["log_revenue"] * log_revenue
        + SEVER_COEFFICIENTS["log_employees"] * log_employees
        + SEVER_COEFFICIENTS["is_public"] * is_public_numeric
        + SEVER_COEFFICIENTS["log_records"] * log_records
    )

    expected_loss = np.exp(log_loss)
    q95_loss = np.exp(LOGNORM_PARAMS["mu"] + LOGNORM_PARAMS["sigma"] * 1.6449)

    return {
        "expected_severity": round(expected_loss, 0),
        "q95_loss": round(q95_loss, 0),
    }

print("✓ Prediction functions defined")

# ============================================================================
# PHASE 6: INSURANCE PREMIUM CALCULATION (Aligned with Insurance Agent)
# ============================================================================

def calculate_insurance_premium(
    company_revenue: float,
    employee_count: int,
    is_public: bool,
    industry_code: str,
    data_records_at_risk: int = 1000000,
) -> dict:
    """
    Calculate insurance premium using Frequency × Severity pipeline.
    Industry relativity dynamically applied from data.
    """
    freq_result = predict_frequency(company_revenue, employee_count, is_public)
    frequency = freq_result["predicted_frequency"]

    sev_result = predict_severity(company_revenue, employee_count, is_public, data_records_at_risk)
    severity = sev_result["expected_severity"]

    # Get industry relativity (dynamic from data)
    if isinstance(INDUSTRY_RELATIVITIES, dict):
        if industry_code in INDUSTRY_RELATIVITIES:
            if isinstance(INDUSTRY_RELATIVITIES[industry_code], dict):
                industry_rel = INDUSTRY_RELATIVITIES[industry_code].get("relativity", 1.0)
            else:
                industry_rel = INDUSTRY_RELATIVITIES[industry_code]
        else:
            industry_rel = 1.0
    else:
        industry_rel = 1.0

    # Pure premium (Frequency × Severity)
    pure_premium = frequency * severity

    # Apply loadings
    loading_components = {
        "acquisition": pure_premium * ACTUARIAL.acquisition_expenses_pct,
        "admin": pure_premium * ACTUARIAL.admin_expenses_pct,
        "profit": pure_premium * ACTUARIAL.profit_margin_pct,
        "uncertainty": pure_premium * ACTUARIAL.uncertainty_buffer_pct,
        "reinsurance": pure_premium * ACTUARIAL.reinsurance_ceded_pct,
    }

    total_loading = sum(loading_components.values())
    final_premium = pure_premium + total_loading

    # Apply industry relativity
    adjusted_premium = final_premium * industry_rel

    return {
        "frequency": frequency,
        "severity": severity,
        "pure_premium": pure_premium,
        "loading_components": loading_components,
        "total_loading": total_loading,
        "final_premium": final_premium,
        "industry_relativity": industry_rel,
        "adjusted_premium": adjusted_premium,
    }

print("✓ Insurance premium calculation function defined")

# ============================================================================
# PHASE 7: REINSURANCE QUOTATION (Based on Insurance Premium)
# ============================================================================

def calculate_proportional_reinsurance(gross_premium: float) -> Dict:
    """Calculate Proportional Reinsurance (Quota Share + Surplus)"""
    quota_gross = gross_premium * ACTUARIAL.quota_share_pct
    quota_commission = quota_gross * ACTUARIAL.quota_share_commission
    quota_net = quota_gross - quota_commission

    surplus_gross = gross_premium * ACTUARIAL.surplus_share_pct
    surplus_commission = surplus_gross * ACTUARIAL.surplus_commission
    surplus_net = surplus_gross - surplus_commission

    return {
        "quota_share": {"gross": quota_gross, "commission": quota_commission, "net": quota_net},
        "surplus_share": {"gross": surplus_gross, "commission": surplus_commission, "net": surplus_net},
        "total_gross": quota_gross + surplus_gross,
        "total_net": quota_net + surplus_net,
    }

def calculate_xol_reinsurance(gross_premium: float) -> Dict:
    """Calculate Excess of Loss (3-layer tower)"""
    layer1_loss = gross_premium * 0.08
    layer1_premium = layer1_loss * ACTUARIAL.xol_profit_loading

    layer2_loss = gross_premium * 0.012
    layer2_premium = layer2_loss * ACTUARIAL.xol_profit_loading

    layer3_loss = gross_premium * 0.0008
    layer3_premium = layer3_loss * ACTUARIAL.xol_profit_loading

    return {
        "layer1": {"limit": 50000000, "premium": layer1_premium, "rol": (layer1_premium/gross_premium)*100},
        "layer2": {"limit": 100000000, "premium": layer2_premium, "rol": (layer2_premium/gross_premium)*100},
        "layer3": {"limit": 150000000, "premium": layer3_premium, "rol": (layer3_premium/gross_premium)*100},
        "total_premium": layer1_premium + layer2_premium + layer3_premium,
    }

print("✓ Reinsurance calculation functions defined")

# ============================================================================
# PHASE 8: AGENT TOOL - REINSURANCE QUOTATION
# ============================================================================

def quotation_tool(
    company_name: str,
    company_revenue_usd: float,
    employee_count: int,
    industry_code: str,
    is_public_company: bool,
    data_records_at_risk: int,
) -> str:
    """Generate complete reinsurance quote using Insurance × Reinsurance pipeline"""
    try:
        if company_revenue_usd < 100000000:
            return f'Error: Minimum revenue is $100M. Received: ${company_revenue_usd:,.0f}'
        
        # Step 1: Calculate Insurance Premium
        insurance = calculate_insurance_premium(
            company_revenue_usd, employee_count, is_public_company, industry_code, data_records_at_risk
        )
        
        # Step 2: Calculate Reinsurance Layers based on adjusted insurance premium
        gross_premium = insurance["adjusted_premium"]
        proportional = calculate_proportional_reinsurance(gross_premium)
        xol = calculate_xol_reinsurance(gross_premium)

        # Get industry name for display
        industry_name = "General"
        if isinstance(INDUSTRY_RELATIVITIES, dict) and industry_code in INDUSTRY_RELATIVITIES:
            if isinstance(INDUSTRY_RELATIVITIES[industry_code], dict):
                industry_name = INDUSTRY_RELATIVITIES[industry_code].get("name", "General")

        result = f"""
╔════════════════════════════════════════════════════════════════════╗
║                   REINSURANCE QUOTATION                            ║
║            (Based on Frequency × Severity Pipeline)                ║
╚════════════════════════════════════════════════════════════════════╝

COMPANY INFORMATION:
  Company Name: {company_name}
  Annual Revenue: ${company_revenue_usd:,.0f}
  Employees: {employee_count:,}
  Industry: {industry_name} (Code: {industry_code})
  Public Status: {'Yes' if is_public_company else 'No'}
  Data Records at Risk: {data_records_at_risk:,}

ACTUARIAL FOUNDATION:
  Predicted Incident Frequency: {insurance['frequency']:.4f} incidents/year
  Expected Severity per Incident: ${insurance['severity']:,.0f}
  Risk Score: {(insurance['frequency']/1.1354)*100:.1f}/100

INSURANCE PREMIUM CALCULATION:
  Pure Premium (Freq × Severity): ${insurance['pure_premium']:,.0f}
  Loading Factors (58%):
    • Acquisition (20%): ${insurance['loading_components']['acquisition']:,.0f}
    • Admin (10%): ${insurance['loading_components']['admin']:,.0f}
    • Profit (15%): ${insurance['loading_components']['profit']:,.0f}
    • Uncertainty (8%): ${insurance['loading_components']['uncertainty']:,.0f}
    • Reinsurance Ceded (5%): ${insurance['loading_components']['reinsurance']:,.0f}
  Total Loading: ${insurance['total_loading']:,.0f}
  
  Insurance Premium (before relativity): ${insurance['final_premium']:,.0f}
  Industry Relativity Factor: {insurance['industry_relativity']:.3f}x (from data)
  ► ADJUSTED INSURANCE PREMIUM: ${insurance['adjusted_premium']:,.0f}

REINSURANCE OPTIONS:

OPTION 1: PROPORTIONAL REINSURANCE
  Quota Share (30%):
    Gross Premium: ${proportional['quota_share']['gross']:,.0f}
    Commission (30%): ${proportional['quota_share']['commission']:,.0f}
    Net Reinsurer Cost: ${proportional['quota_share']['net']:,.0f}

  Surplus Share (15%):
    Gross Premium: ${proportional['surplus_share']['gross']:,.0f}
    Commission (30%): ${proportional['surplus_share']['commission']:,.0f}
    Net Reinsurer Cost: ${proportional['surplus_share']['net']:,.0f}

  ► TOTAL PROPORTIONAL NET: ${proportional['total_net']:,.0f}

OPTION 2: EXCESS OF LOSS (3-LAYER TOWER)
  Layer 1 (0 xs $50M):
    Premium: ${xol['layer1']['premium']:,.0f} ({xol['layer1']['rol']:.2f}% ROL)

  Layer 2 ($50M xs $100M):
    Premium: ${xol['layer2']['premium']:,.0f} ({xol['layer2']['rol']:.2f}% ROL)

  Layer 3 ($150M xs $150M):
    Premium: ${xol['layer3']['premium']:,.0f} ({xol['layer3']['rol']:.2f}% ROL)

  ► TOTAL XOL PREMIUM: ${xol['total_premium']:,.0f}

✨ RECOMMENDED: HYBRID STRUCTURE
  Proportional Core: ${proportional['total_net']:,.0f}
  XOL Layer 3 Tail: ${xol['layer3']['premium']:,.0f}
  ► TOTAL ANNUAL REINSURANCE COST: ${proportional['total_net'] + xol['layer3']['premium']:,.0f}

RATIONALE:
  • Premiums derived from Poisson GLM frequency model & Lognormal severity distribution
  • Industry relativity ({insurance['industry_relativity']:.3f}x) dynamically calculated from data analysis
  • Hybrid structure combines frequency protection (proportional) with catastrophic tail (XOL)
  • Optimal capital efficiency for reinsurer
  • Based on analysis of 750 companies across 20 industries
"""
        return result
    except Exception as e:
        return f'Error: {str(e)}'

print("✓ Quotation tool defined")

# ============================================================================
# PHASE 9: AGENT CREATION
# ============================================================================

try:
    from agno.agent import Agent
    from agno.models.google import Gemini
    
    quotation_agent = Agent(
        name='Cyber Risk Quotation Agent',
        model=Gemini(id='gemini-3.1-flash-lite'),
        tools=[quotation_tool],
        description='Generates insurance and reinsurance quotes using Frequency×Severity models with dynamic industry relativities',
        instructions=f'''You are an insurance actuarial agent specializing in cyber and AI risk pricing.

CAPABILITIES:
- Generate comprehensive insurance and reinsurance quotes
- Use Poisson GLM frequency model + Lognormal severity distribution
- Apply dynamically calculated industry relativity factors
- Compare Proportional vs Excess of Loss structures
- Provide recommendations based on company profile

SUPPORTED INDUSTRIES (NAICS Codes):
{json.dumps({k: v['name'] if isinstance(v, dict) and 'name' in v else str(v) for k, v in list(INDUSTRY_RELATIVITIES.items())[:10]}, indent=2)}

GUIDELINES:
- Minimum revenue: $100 million
- Always use quotation_tool for quotes
- Always quote the premium in the currency, which is provided in the input. Example: If the input currency is in INR, quote the premiums in INR.
- Advise the user to rethink about the input they have given (because the company getting insured might be risky for the insurer):
    - Revenue of the firm is less than the number of employees multiplied by 20000, or
    - Revenue of the firm is less than the number of data records multiplied by 200
- If any of the numeric inputs (revenue, employees, data records) entered is a non-positive number, then inform the user about the mistake and do not generate the quote or provide a risk warning.
- Recommend Hybrid (Proportional + XOL) for enterprise clients
- Industry relativities are dynamically extracted from 750-company dataset
- Be professional and data-driven''',
        markdown=True,
    )
    
    print("✓ Agent created successfully")
    AGENT_READY = True

except Exception as e:
    print(f"⚠️  Warning: Agent creation error: {e}")
    quotation_agent = None
    AGENT_READY = False

# ============================================================================
# STREAMLIT DEPLOYMENT
# ============================================================================

st.set_page_config(
    page_title="Cyber & AI Risk Reinsurance Pricing",
    page_icon="🤖",
    layout="wide"
)

st.title("🌐 Cyber & AI Reinsurance Quotation Agent (Frequency × Severity)")
st.markdown("""
Dynamic actuarial pricing using Poisson GLM + Lognormal models.
**Industry relativities dynamically extracted from data.**
""")

# Initialize Chat History (with unique key for reinsurance agent)
if "messages_reinsurance" not in st.session_state:
    st.session_state.messages_reinsurance = [
        {
            "role": "assistant",
            "content": "Hello! I am your Reinsurance Quotation Agent. I use the same Frequency×Severity actuarial pipeline as the Insurance Pricing Agent, with dynamically calculated industry relativities. Ask me to generate a quote (e.g., 'Generate a reinsurance quote for TechCorp with $5B revenue in Technology')."
        }
    ]

# Display Chat History
for msg in st.session_state.messages_reinsurance:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input Handler
if user_query := st.chat_input("Type your reinsurance pricing request..."):
    
    st.session_state.messages_reinsurance.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        try:
            if quotation_agent is not None and AGENT_READY:
                with st.spinner("Calculating frequency & severity models, applying dynamic industry relativities..."):
                    agent_response = quotation_agent.run(user_query)
                    output_text = agent_response.content if hasattr(agent_response, 'content') else str(agent_response)
            else:
                output_text = "⚠️ Agent initialization pending. Please verify API key configuration in Streamlit Secrets or environment variables."
            
            response_placeholder.markdown(output_text)
            st.session_state.messages_reinsurance.append({"role": "assistant", "content": output_text})
            
        except Exception as e:
            error_msg = f"⚠️ Processing error: {str(e)}"
            response_placeholder.error(error_msg)
            st.session_state.messages_reinsurance.append({"role": "assistant", "content": error_msg})

print("✓ Streamlit interface loaded")
print("✓ Application ready for deployment")
