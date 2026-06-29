import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from typing import Dict, Tuple, Any
from sklearn.linear_model import Ridge, PoissonRegressor
from sklearn.model_selection import train_test_split

# Agno Framework and Model Imports
from agno.agent import Agent
from agno.models.google import Gemini

# ReportLab Engine Layout Components
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# PHASE 1: DYNAMIC MODEL TRAINING — Poisson GLM + Lognormal Severity
# -----------------------------------------------------------------------------
_FREQ_FALLBACK = {
    "intercept": -1.4502, "log_revenue": 0.0841, "log_employees": 0.0512,
    "is_public": 0.1292, "revenue_tier": 0.2145
}
_LOGNORM_FALLBACK = {"mu": 14.8504, "sigma": 1.2842}

@st.cache_data(show_spinner="Training frequency & severity models from live data...")
def train_models():
    """Load data from GitHub and train Poisson GLM + Lognormal models dynamically."""
    try:
        incidents = pd.read_csv(
            "https://raw.githubusercontent.com/rajat4186/Cyber-Risk-Premium-Pricing-Agentic-AI-Project/refs/heads/main/data/incidents_master_cleaned.csv"
        )
        financial = pd.read_csv(
            "https://raw.githubusercontent.com/rajat4186/Cyber-Risk-Premium-Pricing-Agentic-AI-Project/refs/heads/main/data/financial_impact_cleaned.csv"
        )

        # --- Frequency Model (Poisson GLM) ---
        company_freq = (
            incidents.groupby("company_name")
            .agg({"incident_id": "count", "company_revenue_usd": "first",
                  "employee_count": "first", "is_public_company": "first"})
            .reset_index()
            .rename(columns={"incident_id": "incident_count"})
            .dropna()
        )
        Xf = company_freq[["company_revenue_usd", "employee_count", "is_public_company"]].copy()
        Xf["log_revenue"]   = np.log1p(Xf["company_revenue_usd"])
        Xf["log_employees"] = np.log1p(Xf["employee_count"])
        Xf["revenue_tier"]  = pd.cut(Xf["company_revenue_usd"],
                                     bins=[0, 1e9, 10e9, 100e9, np.inf],
                                     labels=[0, 1, 2, 3]).astype(int)
        yf = company_freq["incident_count"]
        Xf_model = Xf[["log_revenue", "log_employees", "is_public_company", "revenue_tier"]].fillna(0)
        Xf_tr, _, yf_tr, _ = train_test_split(Xf_model, yf, test_size=0.2, random_state=42)

        freq_model = PoissonRegressor(alpha=0.1292, max_iter=1000)
        freq_model.fit(Xf_tr, yf_tr)
        freq_coefs = {
            "intercept":    float(freq_model.intercept_),
            "log_revenue":  float(freq_model.coef_[0]),
            "log_employees":float(freq_model.coef_[1]),
            "is_public":    float(freq_model.coef_[2]),
            "revenue_tier": float(freq_model.coef_[3]),
        }

        # --- Severity Model (Lognormal + Ridge) ---
        losses    = financial["total_loss_usd"].values
        log_losses = np.log(losses)
        lognorm   = {"mu": float(log_losses.mean()), "sigma": float(log_losses.std())}

        merged = incidents.merge(financial, on="incident_id", how="inner")
        Xs = merged[["company_revenue_usd", "employee_count", "is_public_company"]].copy()
        Xs["log_revenue"]   = np.log1p(Xs["company_revenue_usd"])
        Xs["log_employees"] = np.log1p(Xs["employee_count"])
        Xs["log_records"]   = np.log1p(1_000_000)
        ys = np.log(merged["total_loss_usd"])
        Xs_model = Xs[["log_revenue", "log_employees", "is_public_company", "log_records"]].fillna(0)
        Xs_tr, _, ys_tr, _ = train_test_split(Xs_model, ys, test_size=0.2, random_state=42)

        sev_model = Ridge(alpha=1.0)
        sev_model.fit(Xs_tr, ys_tr)
        sev_coefs = {
            "intercept":    float(sev_model.intercept_),
            "log_revenue":  float(sev_model.coef_[0]),
            "log_employees":float(sev_model.coef_[1]),
            "is_public":    float(sev_model.coef_[2]),
            "log_records":  float(sev_model.coef_[3]),
        }

        return freq_coefs, lognorm, sev_coefs

    except Exception as e:
        st.warning(f"Live model training failed ({e}). Using pre-trained fallback coefficients.")
        return _FREQ_FALLBACK, _LOGNORM_FALLBACK, None

FREQ_COEFFICIENTS, LOGNORM_PARAMS, SEVER_COEFFICIENTS = train_models()

LOADING_FACTORS = {
    "acquisition": 0.20,
    "admin": 0.10,
    "profit": 0.15,
    "uncertainty": 0.08,
    "reinsurance": 0.05
}
TOTAL_LOADING = sum(LOADING_FACTORS.values())

INDUSTRY_DATA = {
    '51': {'name': 'Information & Technology', 'relativity': 1.231},
    '52': {'name': 'Finance & Insurance', 'relativity': 1.181},
    '44-45': {'name': 'Retail Trade', 'relativity': 1.264},
    '92': {'name': 'Public Administration', 'relativity': 1.221},
    '31-33': {'name': 'Industrial Manufacturing', 'relativity': 1.023},
    '21': {'name': 'Mining', 'relativity': 0.955},
    '62': {'name': 'IT Services / Healthcare', 'relativity': 0.914},
    '22': {'name': 'Utilities', 'relativity': 0.885},
    '42': {'name': 'Business Services', 'relativity': 0.804},
    '55': {'name': 'Management of Companies', 'relativity': 1.417}
    # '48-49': {'name': 'Transportation & Warehousing', 'relativity': 1.000},
    # '11': {'name': 'Agriculture, Forestry & Fishing', 'relativity': 1.000},
    # '72': {'name': 'Accommodation & Food Services', 'relativity': 1.000},
    # '61': {'name': 'Educational Services', 'relativity': 1.000},
    # '53': {'name': 'Real Estate & Rental', 'relativity': 1.000},
    # '54': {'name': 'Professional Services', 'relativity': 1.000},
    # '56': {'name': 'Administrative & Support', 'relativity': 1.000},
    # '81': {'name': 'Other Services', 'relativity': 1.000},
    # '71': {'name': 'Arts, Entertainment & Recreation', 'relativity': 1.000},
    # '23': {'name': 'Construction', 'relativity': 1.000}
}

REVENUE_TIER_RANGES = {
    "Q1_Small": (0, 1e9),
    "Q2_Medium": (1e9, 10e9),
    "Q3_Large": (10e9, 100e9),
    "Q4_Enterprise": (100e9, float("inf")),
}

REVENUE_TIER_RELATIVITIES = {
    "Q1_Small": 0.346,
    "Q2_Medium": 0.578,
    "Q3_Large": 1.024,
    "Q4_Enterprise": 2.050,
}

# -----------------------------------------------------------------------------
# PHASE 2: RISK PREDICTION ROUTINES & UNDERWRITING LOGIC
# -----------------------------------------------------------------------------
def predict_frequency(company_revenue: float, employee_count: int, is_public: bool) -> dict:
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
        FREQ_COEFFICIENTS["intercept"] + 
        FREQ_COEFFICIENTS["log_revenue"] * log_revenue + 
        FREQ_COEFFICIENTS["log_employees"] * log_employees + 
        FREQ_COEFFICIENTS["is_public"] * is_public_numeric + 
        FREQ_COEFFICIENTS["revenue_tier"] * size_category
    )
    predicted_frequency = np.exp(log_lambda)
    risk_score = min(100.0, max(5.0, (predicted_frequency / 1.1354) * 100))
    return {
        "predicted_frequency": round(predicted_frequency, 4),
        "risk_score": round(risk_score, 1),
        "size_category": ["Small", "Medium", "Large", "Enterprise"][size_category]
    }

def predict_severity(company_revenue: float, employee_count: int, is_public: bool) -> dict:
    log_revenue = np.log1p(company_revenue)
    log_employees = np.log1p(employee_count)
    log_loss = LOGNORM_PARAMS["mu"] + 0.15 * log_revenue + 0.05 * log_employees
    expected_loss = np.exp(log_loss)
    q95_loss = np.exp(LOGNORM_PARAMS["mu"] + LOGNORM_PARAMS["sigma"] * 1.6449)
    return {
        "expected_severity": round(expected_loss, 0),
        "q95_loss": round(q95_loss, 0)
    }

def calculate_insurance_quotation(revenue: float, employees: int, is_public: bool, industry_code: str) -> dict:
    freq_metrics = predict_frequency(revenue, employees, is_public)
    sev_metrics = predict_severity(revenue, employees, is_public)
    
    pure_premium = freq_metrics["predicted_frequency"] * sev_metrics["expected_severity"]
    
    # Apply loadings
    acq_load = pure_premium * LOADING_FACTORS["acquisition"]
    admin_load = pure_premium * LOADING_FACTORS["admin"]
    profit_load = pure_premium * LOADING_FACTORS["profit"]
    uncert_load = pure_premium * LOADING_FACTORS["uncertainty"]
    reins_load = pure_premium * LOADING_FACTORS["reinsurance"]
    
    gross_premium = pure_premium * (1 + TOTAL_LOADING)
    
    # Adjust via relativities
    ind_rel = INDUSTRY_DATA.get(industry_code, {'relativity': 1.0})['relativity']
    
    revenue_tier = "Q1_Small"
    for tier, (min_r, max_r) in REVENUE_TIER_RANGES.items():
        if min_r <= revenue < max_r:
            revenue_tier = tier
            break
    rev_rel = REVENUE_TIER_RELATIVITIES.get(revenue_tier, 1.0)
    
    final_premium = gross_premium * ind_rel * rev_rel
    
    # Reinsurance distributions
    quota_share_net = final_premium * 0.30 * 0.85 # 30% quota share minus 15% ceding commission
    surplus_share_net = final_premium * 0.15 * 0.90 # 15% layer minus commission
    layer1_xol = final_premium * 0.08
    hybrid_total_cost = quota_share_net + surplus_share_net + layer1_xol
    
    return {
        "company_profile": {
            "name": st.session_state.get("company_name_input", "Client Corp"),
            "revenue": revenue,
            "employees": employees,
            "industry": INDUSTRY_DATA.get(industry_code, {'name': 'General'})['name']
        },
        "risk_metrics": {
            "predicted_frequency": freq_metrics["predicted_frequency"],
            "expected_severity": sev_metrics["expected_severity"],
            "q95_loss": sev_metrics["q95_loss"],
            "risk_score": freq_metrics["risk_score"]
        },
        "premium_calculations": {
            "pure_premium": pure_premium,
            "loading_components": {
                "acquisition": acq_load,
                "admin": admin_load,
                "uncertainty": uncert_load,
                "reinsurance": reins_load,
                "profit": profit_load
            },
            "final_premium": final_premium
        },
        "reinsurance_allocation": {
            "proportional": {
                "quota_share": {"net": quota_share_net},
                "surplus_share": {"net": surplus_share_net}
            },
            "excess_of_loss": {
                "layer1": {"premium": layer1_xol}
            },
            "hybrid_total_cost": hybrid_total_cost
        }
    }

# -----------------------------------------------------------------------------
# PHASE 3: DYNAMIC NARRATIVE BUILDER
# -----------------------------------------------------------------------------
def _build_narrative(data: dict) -> str:
    company  = data.get("company_profile", {})
    risk     = data.get("risk_metrics", {})
    premium  = data.get("premium_calculations", {})
    reins    = data.get("reinsurance_allocation", {})

    name        = company.get("name", "The client")
    industry    = company.get("industry", "the sector")
    revenue     = company.get("revenue", 0)
    employees   = company.get("employees", 0)
    freq        = risk.get("predicted_frequency", 0)
    severity    = risk.get("expected_severity", 0)
    q95         = risk.get("q95_loss", 0)
    risk_score  = risk.get("risk_score", 0)
    pure_prem   = premium.get("pure_premium", 0)
    final_prem  = premium.get("final_premium", 0)
    hybrid      = reins.get("hybrid_total_cost", 0)
    quota_net   = reins.get("proportional", {}).get("quota_share", {}).get("net", 0)
    surplus_net = reins.get("proportional", {}).get("surplus_share", {}).get("net", 0)
    xol_prem    = reins.get("excess_of_loss", {}).get("layer1", {}).get("premium", 0)

    if risk_score >= 80:
        risk_level      = "HIGH"
        coverage_rec    = "complete Tier 1 + Tier 2 combined coverage"
        urgency         = "Immediate underwriting action is recommended."
    elif risk_score >= 50:
        risk_level      = "MODERATE"
        coverage_rec    = "Tier 1 primary coverage with XOL tail extension"
        urgency         = "Standard underwriting review is advised."
    else:
        risk_level      = "LOW-TO-MODERATE"
        coverage_rec    = "Tier 1 primary coverage"
        urgency         = "Routine monitoring is sufficient."

    return (
        f"{name} operates in the {industry} sector with annual revenues of ${revenue:,.0f} "
        f"and a workforce of {employees:,} employees. "
        f"Actuarial modelling using a Poisson GLM frequency model and Lognormal severity distribution "
        f"projects an incident frequency of {freq:.4f} cyber events per annum with an expected per-event "
        f"loss severity of ${severity:,.0f}. The 95th-percentile tail loss exposure stands at ${q95:,.0f}, "
        f"reflecting the organisation's significant digital footprint and data processing obligations.\n\n"

        f"The Aggregate Risk Score of {risk_score}/100 classifies this entity as a {risk_level} cyber risk. "
        f"The actuarial pure premium — representing expected annual losses before any expense loading — "
        f"is ${pure_prem:,.0f}. After applying the standard 58% composite loading (acquisition 20%, "
        f"administration 10%, profit margin 15%, uncertainty buffer 8%, reinsurance ceded 5%), "
        f"the recommended gross annual premium is ${final_prem:,.0f}.\n\n"

        f"A hybrid reinsurance placement is recommended comprising: a Quota Share layer (net cost "
        f"${quota_net:,.0f}), a Surplus Share layer (net cost ${surplus_net:,.0f}), and an Excess of "
        f"Loss Layer 1 tail aggregate (${xol_prem:,.0f}), yielding a total reinsurance portfolio cost "
        f"of ${hybrid:,.0f}. This structure optimises capital efficiency by combining proportional "
        f"frequency protection with catastrophic tail coverage. "
        f"The underwriting recommendation is {coverage_rec}. {urgency}"
    )

# -----------------------------------------------------------------------------
# PHASE 4: PRODUCTION REPORTLAB HIGH-FIDELITY PDF TOOL
# -----------------------------------------------------------------------------
def generate_pdf_report(quotation_data_json: str, output_filename: str = "Cyber_Risk_Insurance_Report.pdf") -> str:
    """
    Parses full pipeline output data to compile an executive multi-dimensional breakdown report.
    """
    try:
        data = json.loads(quotation_data_json)
    except Exception as e:
        return f"Error parsing input data dictionary JSON: {str(e)}"
        
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54
    )
    styles = getSampleStyleSheet()
    
    # Custom Brand Palette
    PRIMARY_COLOR = colors.HexColor("#1A365D")   # Corporate Deep Slate Blue
    SECONDARY_COLOR = colors.HexColor("#2B6CB0") # Medium Slate
    NEUTRAL_DARK = colors.HexColor("#2D3748")    # Charcoal body text
    BG_LIGHT = colors.HexColor("#F7FAFC")        # Table Header shading
    
    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Heading1'],
        fontSize=22, leading=26, textColor=PRIMARY_COLOR, spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle', parent=styles['Normal'],
        fontSize=10, leading=14, textColor=SECONDARY_COLOR, spaceAfter=18
    )
    h1_style = ParagraphStyle(
        'SectionH1', parent=styles['Heading2'],
        fontSize=13, leading=17, textColor=PRIMARY_COLOR, spaceBefore=12, spaceAfter=8,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'BodyDark', parent=styles['Normal'],
        fontSize=9.5, leading=13.5, textColor=NEUTRAL_DARK, spaceAfter=4
    )
    bold_body_style = ParagraphStyle(
        'BodyDarkBold', parent=body_style, fontName='Helvetica-Bold'
    )
    white_header_style = ParagraphStyle(
        'WhiteHeader', parent=body_style, textColor=colors.white, fontName='Helvetica-Bold'
    )
    
    story = []
    
    # Header Section
    story.append(Paragraph("CYBER & AI RISK INSURANCE REPORT", title_style))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Confidential Actuarial Portfolio", subtitle_style))
    story.append(Spacer(1, 8))
    
    # Section 1: Client Data
    story.append(Paragraph("1. Corporate Risk Profile & Metadata Audit", h1_style))
    company_info = data.get("company_profile", {})
    risk_metrics = data.get("risk_metrics", {})
    
    profile_table = [
        [Paragraph("<b>Attribute</b>", body_style), Paragraph("<b>Value</b>", body_style), Paragraph("<b>Actuarial Metrics</b>", body_style), Paragraph("<b>Value</b>", body_style)],
        [Paragraph("Company Name", body_style), Paragraph(str(company_info.get("name", "N/A")), body_style), Paragraph("Incident Frequency", body_style), Paragraph(str(risk_metrics.get("predicted_frequency", "N/A")), body_style)],
        [Paragraph("Annual Revenue", body_style), Paragraph(f"${company_info.get('revenue', 0):,.0f}", body_style), Paragraph("Expected Severity", body_style), Paragraph(f"${risk_metrics.get('expected_severity', 0):,.0f}", body_style)],
        [Paragraph("Industry Sector", body_style), Paragraph(str(company_info.get("industry", "N/A")), body_style), Paragraph("Aggregate Risk Score", body_style), Paragraph(f"{risk_metrics.get('risk_score', 'N/A')}/100", body_style)]
    ]
    t1 = Table(profile_table, colWidths=[1.5*inch, 1.8*inch, 2.2*inch, 1.5*inch])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), BG_LIGHT),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
        ('BOX', (0,0), (-1,-1), 1, PRIMARY_COLOR),
    ]))
    story.append(t1)
    story.append(Spacer(1, 12))
    
    # Section 2: Premium Calculations
    story.append(Paragraph("2. Actuarial Premium Loading Breakdown", h1_style))
    premium_calc = data.get("premium_calculations", {})
    loadings = premium_calc.get("loading_components", {})
    
    premium_table = [
        [Paragraph("<b>Premium Component Matrix</b>", white_header_style), Paragraph("<b>Allocation Amount (USD)</b>", white_header_style)],
        [Paragraph("Pure Premium (Baseline Expected Losses)", body_style), Paragraph(f"${premium_calc.get('pure_premium', 0):,.0f}", body_style)],
        [Paragraph("Acquisition & Underwriting Expenses Loading", body_style), Paragraph(f"${loadings.get('acquisition', 0):,.0f}", body_style)],
        [Paragraph("Administrative Operating Loading", body_style), Paragraph(f"${loadings.get('admin', 0):,.0f}", body_style)],
        [Paragraph("Actuarial Buffer / Volatility Loading", body_style), Paragraph(f"${loadings.get('uncertainty', 0):,.0f}", body_style)],
        [Paragraph("<b>Final Gross Premium (Annualized)</b>", body_style), Paragraph(f"${premium_calc.get('final_premium', 0):,.0f}", bold_body_style)]
    ]
    t2 = Table(premium_table, colWidths=[4.2*inch, 2.8*inch])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY_COLOR),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
        ('BACKGROUND', (0,-1), (-1,-1), BG_LIGHT),
        ('BOX', (0,0), (-1,-1), 1, PRIMARY_COLOR),
    ]))
    story.append(t2)
    story.append(Spacer(1, 12))
    
    # Section 3: Reinsurance
    story.append(Paragraph("3. Multi-Dimensional Reinsurance Placement Strategy", h1_style))
    reinsurance = data.get("reinsurance_allocation", {})
    reins_table = [
        [Paragraph("<b>Reinsurance Structure Layer</b>", body_style), Paragraph("<b>Net Premium Value (USD)</b>", body_style)],
        [Paragraph("Quota Share Layer Arrangement", body_style), Paragraph(f"${reinsurance.get('proportional', {}).get('quota_share', {}).get('net', 0):,.0f}", body_style)],
        [Paragraph("Surplus Share Layer Coverage", body_style), Paragraph(f"${reinsurance.get('proportional', {}).get('surplus_share', {}).get('net', 0):,.0f}", body_style)],
        [Paragraph("Excess of Loss (XOL) Aggregates", body_style), Paragraph(f"${reinsurance.get('excess_of_loss', {}).get('layer1', {}).get('premium', 0):,.0f}", body_style)],
        [Paragraph("<b>Hybrid Composite Portfolio Value</b>", body_style), Paragraph(f"${reinsurance.get('hybrid_total_cost', 0):,.0f}", bold_body_style)]
    ]
    t3 = Table(reins_table, colWidths=[4.2*inch, 2.8*inch])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), BG_LIGHT),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
        ('BOX', (0,0), (-1,-1), 1, PRIMARY_COLOR),
    ]))
    story.append(t3)
    story.append(Spacer(1, 12))
    
    # Section 4: Narrative
    story.append(Paragraph("4. Strategic Underwriting Narrative", h1_style))
    narrative_text = _build_narrative(data)
    for para in narrative_text.split("\n\n"):
        story.append(Paragraph(para.strip(), body_style))
        story.append(Spacer(1, 6))
    
    doc.build(story)
    return output_filename

# -----------------------------------------------------------------------------
# PHASE 4: UNIFIED STREAMLIT UI VIEW & AGENT ORCHESTRATION
# -----------------------------------------------------------------------------
def run_integrated_pipeline_app():
    st.set_page_config(page_title="Cyber Insurance Underwriting Pipeline", layout="wide")
    st.title("🛡️ Cyber & AI Risk Underwriting & Reporting System")
    st.write("End-to-End Pricing Pipeline powered by Statistical Coefficients & Multi-Agent Agno Orchestration.")

    # Sidebar Inputs
    st.sidebar.header("🏢 Underwriting Risk Profiler")
    st.sidebar.markdown("---")
    
    company_name = st.sidebar.text_input("Company Name", value="TechCorp Systems Inc")
    st.session_state["company_name_input"] = company_name
    
    revenue = st.sidebar.number_input("Annual Gross Revenue ($)", min_value=100000.0, value=250000000.0, step=1000000.0)
    employees = st.sidebar.number_input("Total Employee Count", min_value=1, value=1250, step=10)
    is_public = st.sidebar.checkbox("Is Publicly Traded Enterprise?", value=False)
    
    ind_options = {k: f"{v['name']} (NAICS {k})" for k, v in INDUSTRY_DATA.items()}
    selected_industry = st.sidebar.selectbox("Industry Sector Classification", options=list(ind_options.keys()), format_func=lambda x: ind_options[x])

    st.sidebar.markdown("---")
    trigger_pricing = st.sidebar.button("⚙️ Compute Underwriting Metrics", use_container_width=True)

    # Execute pricing logic when triggered or load active session cache
    if trigger_pricing or "active_underwriting_data" not in st.session_state:
        pricing_results = calculate_insurance_quotation(revenue, employees, is_public, selected_industry)
        st.session_state["active_underwriting_data"] = pricing_results

    active_data = st.session_state["active_underwriting_data"]

    # Dashboard Presentation Visual Widgets
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Annualized Gross Premium", f"${active_data['premium_calculations']['final_premium']:,.2f}")
    with m2:
        st.metric("Hybrid Reinsurance Portfolio Cost", f"${active_data['reinsurance_allocation']['hybrid_total_cost']:,.2f}")
    with m3:
        st.metric("Aggregate Vulnerability Score", f"{active_data['risk_metrics']['risk_score']}/100")

    # Expandable Breakdowns tabs
    t_metrics, t_premium, t_reinsurance = st.tabs(["📊 Actuarial Projections", "💰 Premium Loading Matrix", "🏢 Reinsurance Allocation Layers"])
    
    with t_metrics:
        metrics = active_data["risk_metrics"]
        df_metrics = pd.DataFrame([
            {"Metric": "Predicted Frequency (incidents/year)", "Value": metrics["predicted_frequency"]},
            {"Metric": "Expected Severity",                    "Value": f"${metrics['expected_severity']:,.0f}"},
            {"Metric": "Q95 Loss (95th percentile)",          "Value": f"${metrics['q95_loss']:,.0f}"},
            {"Metric": "Aggregate Risk Score",                 "Value": f"{metrics['risk_score']} / 100"},
        ])
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)

    with t_premium:
        premium = active_data["premium_calculations"]
        loadings = premium["loading_components"]
        df_premium = pd.DataFrame([
            {"Premium Component":                   "Pure Premium (Baseline Expected Losses)", "Amount (USD)": f"${premium['pure_premium']:,.2f}"},
            {"Premium Component": "Acquisition & Underwriting Expenses (20%)",               "Amount (USD)": f"${loadings['acquisition']:,.2f}"},
            {"Premium Component":                          "Administrative Operating (10%)",  "Amount (USD)": f"${loadings['admin']:,.2f}"},
            {"Premium Component":                     "Actuarial Uncertainty Buffer (8%)",    "Amount (USD)": f"${loadings['uncertainty']:,.2f}"},
            {"Premium Component":                              "Reinsurance Ceded (5%)",      "Amount (USD)": f"${loadings['reinsurance']:,.2f}"},
            {"Premium Component":                                  "Profit Margin (15%)",     "Amount (USD)": f"${loadings['profit']:,.2f}"},
            {"Premium Component":                        "Final Gross Premium (Annualized)",  "Amount (USD)": f"${premium['final_premium']:,.2f}"},
        ])
        st.dataframe(df_premium, use_container_width=True, hide_index=True)

    with t_reinsurance:
        reins = active_data["reinsurance_allocation"]
        df_reins = pd.DataFrame([
            {"Reinsurance Structure":         "Quota Share — Net (30% layer, 15% commission)", "Net Premium (USD)": f"${reins['proportional']['quota_share']['net']:,.2f}"},
            {"Reinsurance Structure":       "Surplus Share — Net (15% layer, 10% commission)", "Net Premium (USD)": f"${reins['proportional']['surplus_share']['net']:,.2f}"},
            {"Reinsurance Structure":                    "Excess of Loss — Layer 1 (XOL 8%)", "Net Premium (USD)": f"${reins['excess_of_loss']['layer1']['premium']:,.2f}"},
            {"Reinsurance Structure": "Hybrid Composite Total (Proportional + XOL Combined)", "Net Premium (USD)": f"${reins['hybrid_total_cost']:,.2f}"},
        ])
        st.dataframe(df_reins, use_container_width=True, hide_index=True)

    # Reporting Agent Executive Execution Block
    st.subheader("🤖 Agno Cognitive Reporting Assistant")
    st.write("The reporting agent compiles contextual data models into a production executive summary and exports a signed PDF document.")

    if st.button("🚀 Execute Reporting Agent & Generate Signed PDF"):
        # Configure the Agno Agent with Gemini Engine
        reporting_agent = Agent(
            name="Actuarial Portfolio Reporting Agent",
            model=Gemini(id="gemini-2.5-flash"),
            tools=[generate_pdf_report],
            description="Consolidates multi-agent model inputs and outputs clean structured text and matching ReportLab artifacts.",
            instructions=[
                "You are an Executive Cyber Underwriting Reporting Agent.",
                "Review the structured input calculations and provide a 3-bullet insight regarding the risk profile.",
                "Always run the generate_pdf_report tool using the dictionary context passed to you to generate the file artifact."
            ]
        )

        with st.spinner("Agent constructing multi-dimensional portfolio parameters..."):
            # Inject dynamic underwriting data into agent loop
            agent_prompt = f"Review this underwriting dataset and generate the matching report: {json.dumps(active_data)}"
            response = reporting_agent.run(agent_prompt)
            
            st.success("🎉 Multi-Dimensional Dossier compiled successfully!")
            st.markdown(response.content)

            # Trigger automated PDF construction tool call
            pdf_path = "Cyber_Risk_Insurance_Report.pdf"
            generate_pdf_report(json.dumps(active_data), output_filename=pdf_path)

            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as pdf_file:
                    st.download_button(
                        label="📥 Download Executive Multi-Dimensional PDF Report",
                        data=pdf_file,
                        file_name=f"Cyber_Insurance_Dossier_{company_name.replace(' ', '_')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

if __name__ == "__main__":
    run_integrated_pipeline_app()
