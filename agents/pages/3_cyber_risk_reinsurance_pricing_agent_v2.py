# -*- coding: utf-8 -*-
"""
Cyber & AI Risk Reinsurance Quotation Agent
============================================================

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

np.random.seed(42)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 120)

# ============================================================================
# DYNAMIC PREMIUM STATISTICS CALCULATION
# ============================================================================

def calculate_premium_statistics(pricing_data: pd.DataFrame = None) -> Dict[str, float]:
    """
    Dynamically calculate premium statistics from incident data.
    Falls back to reference values if data unavailable.
    """
    try:
        if pricing_data is None or len(pricing_data) == 0:
            # Load from GitHub if not provided
            pricing_data = pd.read_csv(
                "https://raw.githubusercontent.com/rajat4186/Cyber-Risk-Premium-Pricing-Agentic-AI-Project/refs/heads/main/data/financial_impact_cleaned.csv"
            )
        
        # Calculate statistics
        final_premiums = pricing_data.get('total_loss_usd', [])
        
        if len(final_premiums) > 0:
            stats = {
                'mean_pure_premium': float(final_premiums.mean() * 0.63),  # 63% is pure premium
                'mean_final_premium': float(final_premiums.mean()),
                'median_final_premium': float(final_premiums.median()),
                'min_final_premium': float(final_premiums.min()),
                'max_final_premium': float(final_premiums.max()),
                'std_dev_premium': float(final_premiums.std()),
            }
            return stats
    except Exception as e:
        pass
    
    # Fallback to reference values (from 750-company analysis)
    return {
        'mean_pure_premium': 26.4,
        'mean_final_premium': 41.8,
        'median_final_premium': 30.7,
        'min_final_premium': 8.7,
        'max_final_premium': 163.5,
        'std_dev_premium': 30.0,
    }

# Load premium statistics dynamically
PREMIUM_STATS = calculate_premium_statistics()

# ============================================================================
# ACTUARIAL CONSTANTS (Dynamic)
# ============================================================================

@dataclass
class ActuarialConstants:
    # Dynamic premium statistics from data analysis
    mean_pure_premium: float = PREMIUM_STATS['mean_pure_premium']
    mean_final_premium: float = PREMIUM_STATS['mean_final_premium']
    median_final_premium: float = PREMIUM_STATS['median_final_premium']
    min_final_premium: float = PREMIUM_STATS['min_final_premium']
    max_final_premium: float = PREMIUM_STATS['max_final_premium']
    std_dev_premium: float = PREMIUM_STATS['std_dev_premium']

    # Loading Factors (58% total)
    acquisition_expenses_pct: float = 0.20  # 20%
    admin_expenses_pct: float = 0.10  # 10%
    profit_margin_pct: float = 0.15  # 15%
    uncertainty_buffer_pct: float = 0.08  # 8%
    reinsurance_ceded_pct: float = 0.05  # 5%
    total_loading_pct: float = 0.58  # 58% total

    # Proportional Reinsurance
    quota_share_pct: float = 0.30
    quota_share_commission: float = 0.30
    surplus_share_pct: float = 0.15
    surplus_commission: float = 0.30

    # XOL Parameters
    xol_profit_loading: float = 1.50

ACTUARIAL = ActuarialConstants()

print("✓ Actuarial constants loaded (dynamic from data analysis)")
print(f"  Mean pure premium: ${ACTUARIAL.mean_pure_premium:.1f}M")
print(f"  Mean final premium: ${ACTUARIAL.mean_final_premium:.1f}M")

# ============================================================================
# INDUSTRY CLASSIFICATION & RISK MULTIPLIERS
# ============================================================================

INDUSTRY_DATA = {
    '55': {'name': 'Management of Companies', 'relativity': 1.417, 'count': 2, 'mean_premium': 59.2},
    '44-45': {'name': 'Retail Trade', 'relativity': 1.264, 'count': 65, 'mean_premium': 52.8},
    '51': {'name': 'Information & Technology', 'relativity': 1.231, 'count': 102, 'mean_premium': 51.4},
    '92': {'name': 'Public Administration', 'relativity': 1.221, 'count': 52, 'mean_premium': 51.0},
    '52': {'name': 'Finance & Insurance', 'relativity': 1.181, 'count': 106, 'mean_premium': 49.3},
    '31-33': {'name': 'Manufacturing', 'relativity': 1.023, 'count': 67, 'mean_premium': 42.7},
    '21': {'name': 'Mining, Quarrying, Oil & Gas', 'relativity': 0.955, 'count': 21, 'mean_premium': 39.9},
    '62': {'name': 'Health Care & Social Assistance', 'relativity': 0.914, 'count': 117, 'mean_premium': 38.1},
    '22': {'name': 'Utilities', 'relativity': 0.885, 'count': 38, 'mean_premium': 36.9},
    '42': {'name': 'Wholesale Trade', 'relativity': 0.804, 'count': 10, 'mean_premium': 33.6},
    '48-49': {'name': 'Transportation & Warehousing', 'relativity': 0.800, 'count': 23, 'mean_premium': 33.4},
    '11': {'name': 'Agriculture, Forestry & Fishing', 'relativity': 0.702, 'count': 4, 'mean_premium': 29.3},
    '72': {'name': 'Accommodation & Food Services', 'relativity': 0.690, 'count': 16, 'mean_premium': 28.8},
    '61': {'name': 'Educational Services', 'relativity': 0.663, 'count': 37, 'mean_premium': 27.7},
    '53': {'name': 'Real Estate & Rental', 'relativity': 0.663, 'count': 8, 'mean_premium': 27.7},
    '54': {'name': 'Professional Services', 'relativity': 0.650, 'count': 36, 'mean_premium': 27.1},
    '56': {'name': 'Administrative & Support', 'relativity': 0.620, 'count': 16, 'mean_premium': 25.9},
    '81': {'name': 'Other Services', 'relativity': 0.617, 'count': 3, 'mean_premium': 25.8},
    '71': {'name': 'Arts, Entertainment & Recreation', 'relativity': 0.615, 'count': 14, 'mean_premium': 25.7},
    '23': {'name': 'Construction', 'relativity': 0.598, 'count': 13, 'mean_premium': 25.0},
}

print(f"✓ Supported industries: {len(INDUSTRY_DATA)}")

# ============================================================================
# PREMIUM CALCULATION FUNCTIONS
# ============================================================================

def calculate_gross_premium(company_revenue: float, industry_code: str) -> Dict:
    """Calculate insurance premium using industry relativity"""
    if industry_code not in INDUSTRY_DATA:
        raise ValueError(f'Unknown industry: {industry_code}')

    relativity = INDUSTRY_DATA[industry_code]['relativity']
    base_rate = (ACTUARIAL.mean_final_premium / 1000) * relativity

    pure_premium_rate = base_rate / (1 + ACTUARIAL.total_loading_pct)
    pure_premium_usd = company_revenue * pure_premium_rate
    gross_premium_usd = company_revenue * base_rate

    return {
        'pure_premium_rate': pure_premium_rate,
        'gross_premium_rate': base_rate,
        'pure_premium_usd': pure_premium_usd,
        'gross_premium_usd': gross_premium_usd,
        'industry_relativity': relativity,
        'loading_pct': ACTUARIAL.total_loading_pct,
    }

def calculate_proportional_reinsurance(gross_premium: float) -> Dict:
    """Calculate Proportional Reinsurance (Quota Share + Surplus)"""
    quota_gross = gross_premium * ACTUARIAL.quota_share_pct
    quota_commission = quota_gross * ACTUARIAL.quota_share_commission
    quota_net = quota_gross - quota_commission

    surplus_gross = gross_premium * ACTUARIAL.surplus_share_pct
    surplus_commission = surplus_gross * ACTUARIAL.surplus_commission
    surplus_net = surplus_gross - surplus_commission

    return {
        'quota_share': {'gross': quota_gross, 'commission': quota_commission, 'net': quota_net},
        'surplus_share': {'gross': surplus_gross, 'commission': surplus_commission, 'net': surplus_net},
        'total_gross': quota_gross + surplus_gross,
        'total_net': quota_net + surplus_net,
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
        'layer1': {'limit': 50000000, 'premium': layer1_premium, 'rol': (layer1_premium/gross_premium)*100},
        'layer2': {'limit': 100000000, 'premium': layer2_premium, 'rol': (layer2_premium/gross_premium)*100},
        'layer3': {'limit': 150000000, 'premium': layer3_premium, 'rol': (layer3_premium/gross_premium)*100},
        'total_premium': layer1_premium + layer2_premium + layer3_premium,
    }

print("✓ Calculation functions defined")

# ============================================================================
# AGENT TOOL: REINSURANCE QUOTATION
# ============================================================================

def quotation_tool(company_name: str, company_revenue_usd: float, industry_code: str) -> str:
    """Generate complete reinsurance quote"""
    try:
        if company_revenue_usd < 100000000:
            return f'Error: Minimum revenue is $100M. Received: ${company_revenue_usd:,.0f}'
        if industry_code not in INDUSTRY_DATA:
            return f'Error: Unknown industry {industry_code}. Supported: {list(INDUSTRY_DATA.keys())}'

        insurance = calculate_gross_premium(company_revenue_usd, industry_code)
        proportional = calculate_proportional_reinsurance(insurance['gross_premium_usd'])
        xol = calculate_xol_reinsurance(insurance['gross_premium_usd'])

        result = f"""
╔════════════════════════════════════════════════════════════════════╗
║                   REINSURANCE QUOTATION                            ║
╚════════════════════════════════════════════════════════════════════╝

COMPANY INFORMATION:
  Company Name: {company_name}
  Annual Revenue: ${company_revenue_usd:,.0f}
  Industry: {INDUSTRY_DATA[industry_code]['name']} (Code: {industry_code})
  Industry Relativity: {insurance['industry_relativity']:.3f}x

PRIMARY INSURANCE PREMIUM:
  Gross Premium Rate: {insurance['gross_premium_rate']*100:.2f}%
  Loading Factors: {insurance['loading_pct']*100:.0f}% (Acq 20%, Admin 10%, Profit 15%, Uncertainty 8%, Reinsurance 5%)
  Annual Premium: ${insurance['gross_premium_usd']:,.0f}

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
  ► TOTAL ANNUAL COST: ${proportional['total_net'] + xol['layer3']['premium']:,.0f}

RATIONALE:
  • Hybrid combines frequency protection (proportional) with catastrophic tail (XOL)
  • Optimal capital efficiency for reinsurer
  • Based on analysis of 750 companies across 20 industries
"""
        return result
    except Exception as e:
        return f'Error: {str(e)}'

print("✓ Quotation tool defined")

# ============================================================================
# IMPORTS & AGENT CREATION
# ============================================================================

try:
    from agno.agent import Agent
    from agno.models.google import Gemini
    
    # API Key Setup with multi-path fallback
    if "GOOGLE_API_KEY" not in os.environ:
        try:
            from google.colab import userdata
            os.environ["GOOGLE_API_KEY"] = userdata.get("Final_Project_Key")
        except (ImportError, Exception):
            # Fallback: Check environment or allow graceful degradation
            if "GOOGLE_API_KEY" not in os.environ:
                print("⚠️  Warning: GOOGLE_API_KEY not configured")
                print("    Set via environment: export GOOGLE_API_KEY='your-key'")
                print("    Or Streamlit secrets: .streamlit/secrets.toml")
    
    quotation_agent = Agent(
        name='Cyber Risk Quotation Agent',
        model=Gemini(id='gemini-2.5-flash'),
        tools=[quotation_tool],
        description='Generates insurance and reinsurance quotes for cyber/AI risk',
        instructions=f'''You are an insurance actuarial agent specializing in cyber and AI risk pricing.

CAPABILITIES:
- Generate comprehensive insurance and reinsurance quotes
- Compare Proportional vs Excess of Loss structures
- Provide recommendations based on company profile
- Explain actuarial assumptions and methodology

SUPPORTED INDUSTRIES (NAICS Codes):
{json.dumps({k: v['name'] for k, v in INDUSTRY_DATA.items()}, indent=2)}

GUIDELINES:
- Minimum revenue: $100 million
- Always use quotation_tool for quotes
- Recommend Hybrid (Proportional + XOL) for most cases
- Data based on 750 companies analyzed
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

import streamlit as st
import os

# Configure Streamlit
st.set_page_config(
    page_title="Cyber & AI Risk Reinsurance Pricing",
    page_icon="🤖",
    layout="wide"
)

st.title("🌐 Cyber & AI Reinsurance Quotation Agent")
st.markdown("Dynamic actuarial pricing for commercial cyber and AI risk insurance.")

# API Key Management
if "GOOGLE_API_KEY" not in os.environ:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    elif "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    elif "Final_Project_Key" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["Final_Project_Key"]

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your Reinsurance Quotation Agent. Ask me to generate a quote for your company (e.g., 'Generate a reinsurance quote for TechCorp with $5B revenue in Technology'). Minimum revenue: $100M."
        }
    ]

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input Handler
if user_query := st.chat_input("Type your reinsurance pricing request..."):
    
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        try:
            if quotation_agent is not None and AGENT_READY:
                with st.spinner("Analyzing risk profile and generating quote..."):
                    agent_response = quotation_agent.run(user_query)
                    output_text = agent_response.content if hasattr(agent_response, 'content') else str(agent_response)
            else:
                output_text = "⚠️ Agent initialization pending. Please verify API key configuration in Streamlit Secrets or environment variables."
            
            response_placeholder.markdown(output_text)
            st.session_state.messages.append({"role": "assistant", "content": output_text})
            
        except Exception as e:
            error_msg = f"⚠️ Processing error: {str(e)}"
            response_placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

print("✓ Streamlit interface loaded")
print("✓ Application ready for deployment")
