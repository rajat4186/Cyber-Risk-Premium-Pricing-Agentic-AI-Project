# 🛡️ Cyber Risk Premium Pricing - Agentic AI Project

**Guiding the Future of Cyber Insurance Pricing with AI**

An end-to-end, multi-agent agentic AI system for pricing cyber risk across insurance and reinsurance lines, built on actuarial GLM/Lognormal frequency-severity models and orchestrated with Agno + Google Gemini.

Submitted under the **SSSIA AI Internship Program**

**Authors:** Rajat Chhabra, Suvitha Nagarajan, Sarah Comfort Samson

---

## 📌 Overview

Cyber risk is notoriously hard to price — sparse loss data, fat-tailed severity distributions, and fast-evolving attack vectors make traditional actuarial pricing difficult to scale. This project explores whether a coordinated set of AI agents, each responsible for a distinct stage of the underwriting lifecycle, can produce transparent, auditable, and actuarially grounded premium quotations for commercial cyber and AI risk — for both **insurance** and **reinsurance** lines.

The system is calibrated on a dataset of **750 companies across 20 industries** and prices coverage across:

- Ransomware
- Data breaches
- Supply chain attacks
- DDoS
- Malware, trojans, backdoors
- Advanced Persistent Threats (APTs)

organized around four underlying risk pillars:

1. **Data / Model Integrity**
2. **Operational Interfacing**
3. **Data Privacy & IP Exfiltration**
4. **Governance & Supply Chain**

---

## 🧠 System Architecture

The application is a multi-page **Streamlit** app, with each page powered by a dedicated **Agno** agent running on **Google Gemini**.

```
WELCOME.py                                        → Landing dashboard & navigation
agents/pages/
 ├── 1_DATA_VALIDATION_AGENT.py                    → Data Validation Agent
 ├── 2_INSURANCE_PRICING_AGENT_GUARDRAIL.py         → Insurance Pricing Agent
 ├── 3_REINSURANCE_PRICING_AGENT_REFACTORED_FIXED.py → Reinsurance Pricing Agent
 └── 4_REPORTING_AGENT.py                          → Executive Reporting Agent
```

| Agent | Role |
|---|---|
| **Data Validation Agent** | Agentic schema audit, adaptive imputation, and AI-assisted text/entity cleaning with a reflection loop to reject unsafe transformations |
| **Insurance Pricing Agent** | Frequency × Severity premium quotation with a 5%-of-revenue guardrail, industry relativities, and tiered coverage |
| **Reinsurance Pricing Agent** | Quota Share, Surplus Share, and 3-layer Excess-of-Loss (XOL) reinsurance structuring, with bidirectional industry name/code resolution |
| **Executive Reporting Agent** | Consolidates outputs into an executive narrative and a formatted PDF underwriting report (ReportLab) |

---

## 📐 Actuarial Methodology

- **Frequency Model:** Poisson GLM with Ridge regularization (`sklearn.linear_model.PoissonRegressor`), trained on `log_revenue`, `log_employees`, `is_public_company`.
- **Severity Model:** Lognormal loss distribution + Ridge regression (`sklearn.linear_model.Ridge`), trained on `log_revenue`, `log_employees`, `is_public_company`, `log_records`.
- **Loading Stack (58% composite):**
  - Acquisition — 20%
  - Administration — 10%
  - Profit Margin — 15%
  - Uncertainty Buffer — 8%
  - Reinsurance Ceded — 5%
- **Revenue Guardrail:** Final premium capped at a maximum of 5% of annual company revenue; all loading components scale down proportionally if triggered.
- **Industry Relativities:** Dynamically calculated as (industry mean loss / overall mean loss) from NAICS-coded incident data where volume permits; falls back to static reference relativities for industries with insufficient data (10 of 20 industries).
- **Reinsurance Structures:**
  - Quota Share (30% cession, 30% commission)
  - Surplus Share (15% cession, 30% commission)
  - Excess of Loss — 3-layer tower (ROL-based pricing)
  - Recommended hybrid: Proportional core + XOL tail layer

---

## 🗂️ Data

Data is loaded dynamically at runtime from GitHub-hosted CSVs, with hardcoded fallback coefficients if the pipeline is unreachable:

- `incidents_master.csv` — incident-level records (company, industry, attack type)
- `financial_impact.csv` — loss amounts per incident
- `market_impact.csv` — market/reputational impact data
- `complete_pricing_framework.xlsx` — empirical industry relativity reference table

---

## ⚙️ Tech Stack

| Layer | Tools |
|---|---|
| Agent Orchestration | [Agno](https://github.com/agno-agi/agno) |
| LLM | Google Gemini (2.0 Flash / 3.1 Flash Lite) |
| UI | Streamlit (multi-page) |
| Modeling | scikit-learn (Poisson GLM, Ridge regression) |
| Reporting | ReportLab (PDF generation) |
| Data | Pandas, NumPy, GitHub-hosted CSVs |

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install streamlit agno google-genai scipy scikit-learn pandas numpy reportlab
```

### API Key Setup
The app looks for a Gemini API key via a three-path fallback:
1. `GOOGLE_API_KEY` environment variable
2. `st.secrets["GOOGLE_API_KEY"]`
3. `st.secrets["Final_Project_Key"]`

Create a `.streamlit/secrets.toml`:
```toml
GOOGLE_API_KEY = "your-api-key-here"
```

### Run the app
```bash
streamlit run agents/WELCOME.py
```

---

## 📁 Repository Structure

```
Cyber-Risk-Premium-Pricing-Agentic-AI-Project/
├── agents/
│   ├── WELCOME.py
│   └── pages/
│       ├── 1_DATA_VALIDATION_AGENT.py
│       ├── 2_INSURANCE_PRICING_AGENT_GUARDRAIL.py
│       ├── 3_REINSURANCE_PRICING_AGENT_REFACTORED_FIXED.py
│       └── 4_REPORTING_AGENT.py
├── data/
│   ├── incidents_master_cleaned.csv
│   ├── financial_impact_cleaned.csv
│   └── market_impact_cleaned.csv
└── README.md
```

---

## 🧭 Roadmap / Known Issues

- Reconcile severity scaling logic between the Insurance Pricing Agent and the Reporting Agent's `predict_severity()` function so both stay consistent as the underlying models are refined.
- Extend empirical industry relativity calculation to the remaining data-constrained industries as more incident data becomes available.
- Expand reinsurance guardrail coverage in `calculate_insurance_premium()`.
- Add automated regression tests comparing agent-level premium outputs for consistency.

---

## ⚠️ Disclaimer

This is an academic/practical capstone project built for the SSSIA AI Internship Program. Pricing outputs are illustrative and based on a synthetic/sample dataset of 750 companies — they are **not** intended for actual underwriting or commercial use without independent actuarial validation.

---

## 📄 License

N/A
