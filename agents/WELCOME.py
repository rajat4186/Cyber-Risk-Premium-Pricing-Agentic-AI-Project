import streamlit as st

# ==============================================================================
# PAGE CONFIGURATION & THEME
# ==============================================================================
st.set_page_config(
    page_title="Cyber & AI Risk Actuarial Suite",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# MULTI-PAGE DECLARATION (Using modern st.navigation Architecture)
# ==============================================================================
# Defining pages explicitly ensures strict order, custom icons, and proper rendering
homepage = st.Page(
    page="WELCOME.py", 
    title="Dashboard Home", 
    icon="💻", 
    default=True
)

data_validation_agent = st.Page(
    page="pages/1_DATA_VALIDATION_AGENT.py", 
    title="Data Validation Agent", 
    icon="✅"
)

insurance_pricing_agent = st.Page(
    page="pages/2_INSURANCE_PRICING_AGENT.py", 
    title="Insurance Pricing Agent", 
    icon="📊"
)

reinsurance_pricing_agent = st.Page(
    page="pages/3_REINSURANCE_PRICING_AGENT.py", 
    title="Reinsurance Pricing Agent", 
    icon="🧱"
)

reporting_agent = st.Page(
    page="pages/Reporting_agent_new.py", 
    title="Reporting Agent", 
    icon="📋"
)

# Initialize the navigation system
pg = st.navigation(
    {
        "Overview": [homepage],
        "Agentic Core Modules": [
            data_validation_agent, 
            insurance_pricing_agent, 
            reinsurance_pricing_agent, 
            reporting_agent
        ]
    }
)

# Run navigation route guard logic
if pg != homepage:
    pg.run()
else:
    # ==============================================================================
    # HOMEPAGE / DASHBOARD INTERFACE
    # ==============================================================================
    
    # Header Banner Section
    st.title("🛡️ Cyber & AI Risk Premium Pricing Framework")
    st.subheader("An Agentic AI End-to-End Actuarial Ingestion & Quotation Architecture")
    st.markdown("---")

    # Metrics Summary Bar
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric(label="System Status", value="Operational", delta="Stable")
    with col_m2:
        st.metric(label="Ingested Historical Records", value="750 Companies", delta="Verified")
    with col_m3:
        st.metric(label="Industry Coverage Nodes", value="20 Verticals", delta="Synchronized")
    with col_m4:
        st.metric(label="Underwriting Framework", value="NIST CSF 2.0 / FAIR", delta="Compliant")

    st.markdown("### 🚀 Module Framework Navigator")
    st.write("Welcome to the predictive pricing control deck. Select a module from the sidebar or explore the specific functions of our decentralized agent networks below:")

    # Grid Layout for Agent Descriptions
    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("#### 🔍 1. Data Validation Agent")
            st.markdown(
                """
                Handles raw corporate metadata logs and unstructured system feeds. 
                Executes automated polymorphic data type validation, missing record imputations, 
                and regex cleaning rules to generate contextually sanitized datasets.
                """
            )
            # st.caption("Primary Artifacts: `incidents_cleaned.csv` | `financial_impact_cleaned.csv`")

        st.markdown(" ") # Spacer

        with st.container(border=True):
            st.markdown("#### ⛓️ 3. Reinsurance Pricing Agent")
            st.markdown(
                """
                Evaluates aggregate loss distribution models and capital market vulnerabilities. 
                Utilizes specialized sub-routines to structure Risk-Transfer boundaries, calculating 
                Quota Share percentages, Excess of Loss (XOL) attachment triggers, and systemic accumulation layers.
                """
            )
            # st.caption("Primary Artifacts: `reinsurance_treaty_structures.json`")

    with col2:
        with st.container(border=True):
            st.markdown("#### 📊 2. Insurance Pricing Agent")
            st.markdown(
                """
                Houses the core predictive mathematical modeling stack. Marries the GLM frequency model with empirical severity loss estimations to calculate risk premiums. 
                Applies risk loading factors across Primary (Tier 1) and Secondary (Tier 2) attack profiles.
                """
            )
            # st.caption("Primary Artifacts: `poisson_frequency_model.py` | `premium_calculation_engine.py`")

        st.markdown(" ") # Spacer

        with st.container(border=True):
            st.markdown("#### 📋 4. Reporting Agent")
            st.markdown(
                """
                Synthesizes technical underwriting outputs and structures executive briefing books. 
                Compiles multi-page PDF performance manuals, visualizes residual risk distributions, 
                and outputs production-ready compliance reports for auditing and regulatory oversight.
                """
            )
            # st.caption("Primary Artifacts: `Executive_Risk_Briefing.pdf`")

    # Footer Information
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: gray; font-size: 0.85em;">
            © 2026 Agentic AI Cyber Insurance Consortium | Actuarial Lines & Reinsurance Pricing Engine v1.0
        </div>
        """,
        unsafe_allow_html=True
    )
