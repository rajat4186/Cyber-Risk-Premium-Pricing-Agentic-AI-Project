import streamlit as st

# Configure the unified dashboard layout
st.set_page_config(
    page_title="Corporate Cyber Risk & Actuarial Intelligence Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main Dashboard Header
st.title("🛡️ Corporate Cyber Risk & Actuarial Intelligence Suite")
st.subheader("An Agentic AI Framework for Commercial Underwriting & Portfolio Risk Management")
st.markdown("---")

# Executive Overview Summary
st.markdown("""
### 📊 Platform Architecture Overview
This unified platform bridges advanced predictive modeling with generative agentic frameworks to streamline commercial lines underwriting. 
By translating historical breach vectors into quantitative metrics, the ecosystem systematically removes manual database queries, automates schema validations, and yields mathematically sound, loaded risk premiums.

**Select an operational agent module from the sidebar menu to get started.**
""")

st.markdown("<br>", unsafe_allow_html=True)

# Layout Columns to Highlight the First Two Agents Side-by-Side
col1, col2 = st.columns(2)

with col1:
    st.image("https://img.icons8.com/fluent/96/000000/checked-user-male.png", width=64)  # Placeholder Icon
    st.markdown("## 📑 Agent 1: Data Validation & Cleaning")
    st.markdown("""
    **Objective:** Streamlines multi-format file ingestion, resolves formatting syntax anomalies, and validates data integrity before modeling phases.
    
    ### ⚙️ Core Capabilities:
    * **Polymorphic Ingestion:** Automatically parses structural configurations from messy files.
    * **Statistical Sanitization:** Programmatically tracks down missing variables and strips out currency/bracket string noise.
    * **Self-Correcting Guardrails:** Wraps exceptions inside a ReAct reasoning loop to fix schema mismatches without throwing execution halts.
    """)
    st.markdown("👉 *Navigate to **Data Validation** in the sidebar to clean new files.*")

with col2:
    st.image("https://img.icons8.com/fluent/96/000000/file-insurance.png", width=64)  # Placeholder Icon
    st.markdown("## 📊 Agent 2: Cyber Insurance Pricing")
    st.markdown("""
    **Objective:** Evaluates unique corporate risk parameters to calculate granular, actuarially loaded premium brackets.
    
    ### ⚙️ Core Capabilities:
    * **Poisson GLM Frequency Engine:** Calculates company-specific annualized breach event probability ($\lambda$) based on cohort exposure metrics.
    * **Lognormal Severity Model:** Computes expected financial impact conditional upon an event occurring.
    * **Risk-Based Tier Allocation:** Automatically segments coverage lines:
        * **Tier 1 (Primary):** Ransomware, Data Breaches, and Supply Chain Risk.
        * **Tier 2 (Secondary):** DDoS, Malware, Trojans, Backdoors, and APTs.
    """)
    st.markdown("👉 *Navigate to **Cyber Pricing** in the sidebar to generate instant insurance quotes.*")

# --- Footer Area detailing Upcoming Enhancements ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888888; font-size: 0.9em;'>
    <strong>Upcoming Framework Extensions:</strong> Module 3 (Reinsurance Treaty Optimization Agent) is currently in development to handle capital tail-risk, stop-loss triggers, and Excess of Loss (XOL) allocation structures.
</div>
""", unsafe_allow_html=True)
