import os
import json
from datetime import datetime
from typing import Dict, Any
from agno.agent import Agent
from agno.models.google import Gemini

# ReportLab imports for professional multi-page document layout
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_pdf_report_tool(
    company_data_json: str, 
    data_validation_json: str,
    primary_quote_json: str, 
    reinsurance_quote_json: str
) -> str:
    """
    Compiles structured telemetry outputs from Data Validation, Premium Pricing, 
    and Reinsurance agents into a single multi-dimensional board-ready PDF report.
    Uses an ultra-premium Black and Gold executive palette.
    """
    try:
        # Safely parse JSON inputs passed from upstream agents
        comp = json.loads(company_data_json)
        val = json.loads(data_validation_json)
        p_quote = json.loads(primary_quote_json)
        r_quote = json.loads(reinsurance_quote_json)
        
        pdf_filename = f"Cyber_Risk_Executive_Report_{comp.get('name', 'Company')}.pdf"
        doc = SimpleDocTemplate(
            pdf_filename,
            pagesize=letter,
            rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
        )
        
        styles = getSampleStyleSheet()
        
        # ----------------------------------------------------
        # THEME ACCENTS: EXECUTIVE MATTE BLACK & SECTOR GOLD
        # ----------------------------------------------------
        PRIMARY_COLOR = colors.HexColor("#1A1A1A")   # Rich Off-Black / Obsidian
        SECONDARY_COLOR = colors.HexColor("#C5A059") # Matte Premium Metallic Gold
        ACCENT_COLOR = colors.HexColor("#8C6D31")    # Deep Muted Gold
        TEXT_COLOR = colors.HexColor("#2D2D2D")      # Crisp Charcoal for body text
        LIGHT_BG = colors.HexColor("#F9F8F6")        # Soft Warm Off-White Cream
        
        # Safe style overriding to avoid collisions
        styles['Normal'].textColor = TEXT_COLOR
        styles['Normal'].fontSize = 10
        styles['Normal'].leading = 14
        
        title_style = ParagraphStyle(
            'DocTitle', parent=styles['Normal'],
            fontName='Helvetica-Bold', fontSize=24, leading=28,
            textColor=PRIMARY_COLOR, spaceAfter=15
        )
        h1_style = ParagraphStyle(
            'SectionH1', parent=styles['Normal'],
            fontName='Helvetica-Bold', fontSize=14, leading=18,
            textColor=PRIMARY_COLOR, spaceBefore=14, spaceAfter=8,
            keepWithNext=True
        )
        body_style = styles['Normal']
        
        story = []
        
        # ====================================================
        # PAGE 1: TITLE & EXECUTIVE PROFILE + DATA VALIDATION
        # ====================================================
        story.append(Paragraph("CYBER & AI RISK PLACEMENT REPORT", title_style))
        story.append(Paragraph(f"<b>Prepared For:</b> {comp.get('name')} | <b>Date:</b> {datetime.now().strftime('%B %d, %Y')}", body_style))
        story.append(Spacer(1, 12))
        
        story.append(Paragraph("1. Executive Risk Summary & Data Integrity Profile", h1_style))
        summary_text = (
            f"This executive risk placement report provides a holistic, multi-dimensional overview of the cyber risk profile, "
            f"underwriting data validation audit benchmarks, and advanced capital-efficient risk transfer structures built for <b>{comp.get('name')}</b>."
        )
        story.append(Paragraph(summary_text, body_style))
        story.append(Spacer(1, 10))
        
        # Core Parameters & Data Validation Matrix
        info_data = [
            [Paragraph("<b>Risk & Governance Metric</b>", body_style), Paragraph("<b>Assigned Profile</b>", body_style), Paragraph("<b>Actuarial / Audit Relativity</b>", body_style)],
            ["Annual Gross Revenue", comp.get('revenue', 'N/A'), p_quote.get('premium_adjustments', {}).get('company_size_adjustment', 'Baseline')],
            ["Industry Domain Sector", comp.get('industry', 'N/A'), p_quote.get('premium_adjustments', {}).get('industry_adjustment', 'Baseline')],
            ["Calculated Cyber Risk Score", f"{p_quote.get('actuarial_analysis', {}).get('risk_score', 'N/A')} / 100", "Normalized frequency distribution"],
            ["Data Hygiene Audit Status", f"<b>{val.get('status', 'PASSED')}</b>", "Verified via Data Validation Agent"],
            ["Remediated Record Volatilities", val.get('cleaned_records_count', '0 Records'), f"Missing fields isolated: {val.get('missing_fields_remediated', 0)}"]
        ]
        
        t_info = Table(info_data, colWidths=[160, 150, 220])
        t_info.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#EAEAEA")),
            ('TEXTCOLOR', (0,0), (-1,0), PRIMARY_COLOR),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG]),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#D1D1D1")),
            ('LINEBELOW', (0,0), (-1,0), 1.5, SECONDARY_COLOR) # Gold line accent under table header
        ]))
        story.append(t_info)
        story.append(Spacer(1, 15))
        
        # ----------------------------------------------------
        # PRIMARY UNDERWRITING DETAILS
        # ----------------------------------------------------
        story.append(Paragraph("2. Primary Insurance Architecture & Actuarial Loadings", h1_style))
        actuarial_intro = (
            f"The primary loss engine calibrated a pure underlying actuarial premium of "
            f"<b>{p_quote.get('actuarial_analysis', {}).get('pure_premium', 'N/A')}</b>. "
            f"The comprehensive base-loaded exposure costs are distributed according to statutory expense models:"
        )
        story.append(Paragraph(actuarial_intro, body_style))
        story.append(Spacer(1, 10))
        
        # Loading Factor Grid
        loadings = p_quote.get('loading_breakdown', {})
        loading_data = [
            [Paragraph("<b>Statutory Expense Allocation Type</b>", body_style), Paragraph("<b>Loaded Cost Matrix</b>", body_style)],
            ["Acquisition Costs & Distribution Fees (20%)", loadings.get('acquisition_expenses_20%', 'N/A')],
            ["Administrative Expenses & Operational Overhead (10%)", loadings.get('administrative_10%', 'N/A')],
            ["Underwriting Underlying Margin & Expected Profit (15%)", loadings.get('profit_margin_15%', 'N/A')],
            ["Model/Volatility Uncertainty Risk Buffer (8%)", loadings.get('uncertainty_buffer_8%', 'N/A')],
            ["Reinsurance Operational Friction Allocation (5%)", loadings.get('reinsurance_5%', 'N/A')],
            ["Total Loaded Base Annual Premium", p_quote.get('premium_adjustments', {}).get('adjusted_premium', 'N/A')]
        ]
        
        t_load = Table(loading_data, colWidths=[320, 210])
        t_load.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), PRIMARY_COLOR), # Black Header
            ('TEXTCOLOR', (0,0), (-1,0), SECONDARY_COLOR), # Gold Header Text
            ('FONTNAME', (0,-1), (-1,-1), 'Helvetica-Bold'),
            ('BACKGROUND', (0,-1), (-1,-1), LIGHT_BG),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
            ('LINEBELOW', (0,-1), (-1,-1), 1.5, SECONDARY_COLOR) # Gold Highlight Accent on total summary row
        ]))
        story.append(t_load)
        
        story.append(PageBreak()) # Force break to separate Reinsurance and Strategy onto Page 2 cleanly
        
        # ====================================================
        # PAGE 2: REINSURANCE STRUCTURES & HYBRID TRANSFER STRATEGY
        # ====================================================
        story.append(Paragraph("3. Global Capital Reinsurance Structures", h1_style))
        story.append(Paragraph("To shield underlying balance sheets from extreme cyber accumulation claims, risk structures are parsed below:", body_style))
        story.append(Spacer(1, 10))
        
        prop = r_quote.get('proportional', {})
        xol = r_quote.get('xol', {})
        
        reins_data = [
            [Paragraph("<b>Reinsurance Structure Option</b>", body_style), Paragraph("<b>Coverage Parameters</b>", body_style), Paragraph("<b>Net Cost Framework</b>", body_style)],
            ["Proportional Quota Share (30%)", "30% Net Ceded Exposure Pool", prop.get('quota_share', {}).get('net', 'N/A')],
            ["Proportional Surplus Share (15%)", "15% High-Capacity Retention Line", prop.get('surplus_share', {}).get('net', 'N/A')],
            ["Excess of Loss (XOL) - Layer 1", "Limit: $50M Attachment Base", xol.get('layer1', {}).get('premium', 'N/A')],
            ["Excess of Loss (XOL) - Layer 2", "Limit: $100M Medium Volatility Layer", xol.get('layer2', {}).get('premium', 'N/A')],
            ["Excess of Loss (XOL) - Layer 3", "Limit: $150M Systemic Catastrophe Tail", xol.get('layer3', {}).get('premium', 'N/A')]
        ]
        
        t_reins = Table(reins_data, colWidths=[180, 200, 160])
        t_reins.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), PRIMARY_COLOR), # Black Header
            ('TEXTCOLOR', (0,0), (-1,0), SECONDARY_COLOR), # Gold Text
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG]),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#D1D1D1"))
        ]))
        story.append(t_reins)
        story.append(Spacer(1, 15))
        
        story.append(Paragraph("4. Recommended Hybrid Placement & Next Steps", h1_style))
        rec_text = (
            f"<b>Structural Placement Recommendation:</b> For optimized capital retention, we advise deploying the <b>Hybrid Strategy</b>. "
            f"This mechanism merges Proportional mechanisms to handle expected predictable loss ratios while retaining an XOL Layer 3 structure "
            f"to cover high-impact systemic claims. "
            f"<br/><br/><b>Compliance & Binding Requirements:</b> Ensure all data pipelines match the confirmed data validation parameters "
            f"and return the signed binding paperwork within the designated 30-day corporate window."
        )
        story.append(Paragraph(rec_text, body_style))
        
        doc.build(story)
        return os.path.abspath(pdf_filename)
        
    except Exception as e:
        return f"Error compiling layout: {str(e)}"

# ----------------------------------------------------
# Instantiating the Master Reporting Agent via Agno
# ----------------------------------------------------
reporting_agent = Agent(
    name="Cyber Risk Multi-Dimensional Reporting Agent",
    model=Gemini(id="gemini-2.0-flash"),
    tools=[generate_pdf_report_tool],
    instructions="""
    You are a professional Cyber Insurance Director and Report Orchestrator.
    Your objective is to accept parameters representing the company profile, data validation status, primary underwriting data, and reinsurance matrix data.
    You must call the `generate_pdf_report_tool` with stringified representations of these parameters to generate a clean, professional PDF.
    After running the tool successfully, provide a clear text overview summary stating that the Black and Gold premium report artifact has been securely generated and print its absolute local folder destination path.
    """,
    markdown=True
)

# ----------------------------------------------------
# PIPELINE PLAYGROUND / LOCAL DEMO SIMULATION
# ----------------------------------------------------
if __name__ == "__main__":
    # 1. Setup API key validation to protect against engine crashes
    if "GEMINI_API_KEY" not in os.environ:
        print("[WARNING]: Environment variable 'GEMINI_API_KEY' not found. Please assign it before execution.")
        # os.environ["GEMINI_API_KEY"] = "AIzaSy..."

    # 2. Mocking outputs calculated by Upstream Agents 1 & 2 + Data Validation Check
    mock_company = {
        "name": "Nexus Global Cloud Technologies",
        "revenue": "$12.4 Billion",
        "employees": "34,000",
        "industry": "Information Technologies (Code 51)"
    }
    
    mock_data_validation = {
        "status": "PASSED / REGULATORY CLEAN",
        "cleaned_records_count": "758,200 Records Checked",
        "missing_fields_remediated": 42
    }

    mock_premium_pricing = {
        "actuarial_analysis": {
            "pure_premium": "$42,100,000",
            "risk_score": 84.1
        },
        "premium_adjustments": {
            "company_size_adjustment": "1.082 x Enterprise Scale Modifier",
            "industry_adjustment": "1.250 x High Risk Code 51",
            "adjusted_premium": "$54,800,000"
        },
        "loading_breakdown": {
            "acquisition_expenses_20%": "$10,960,000",
            "administrative_10%": "$5,480,000",
            "profit_margin_15%": "$8,220,000",
            "uncertainty_buffer_8%": "$4,384,000",
            "reinsurance_5%": "$2,740,000"
        }
    }

    mock_reinsurance = {
        "proportional": {
            "quota_share": {"net": "$11,508,000"},
            "surplus_share": {"net": "$5,754,000"}
        },
        "xol": {
            "layer1": {"premium": "$4,384,000"},
            "layer2": {"premium": "$657,600"},
            "layer3": {"premium": "$43,840"}
        }
    }

    # 3. Triggering Orchestration Run Execution
    print("Initiating Multi-Dimensional Report Compilation Workflow Pipeline...")
    reporting_agent.print_response(
        f"Generate our multi-dimensional executive board-level placement summary document using these data frames:\n\n"
        f"Company Baseline Information: {json.dumps(mock_company)}\n"
        f"Data Validation Diagnostics: {json.dumps(mock_data_validation)}\n"
        f"Primary Premium Calculations: {json.dumps(mock_premium_pricing)}\n"
        f"Reinsurance Quotation Parameters: {json.dumps(mock_reinsurance)}"
    )
