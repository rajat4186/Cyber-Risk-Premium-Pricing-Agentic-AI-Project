import os
import ast
import glob
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from google.genai.errors import APIError
from agno.agent import Agent
from agno.models.google import Gemini

# ----------------------------------------------------
# API Key Setup
# ----------------------------------------------------
if "GOOGLE_API_KEY" not in os.environ:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    elif "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    elif "Capstone_Project" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["Capstone_Project"]

if "GEMINI_API_KEY" not in os.environ:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    elif "GOOGLE_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    elif "Capstone_Project" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["Capstone_Project"]

try:
    from google.colab import userdata
    colab_key = userdata.get('Capstone_Project') or userdata.get('GEMINI_API_KEY') or userdata.get('GOOGLE_API_KEY')
    if colab_key:
        if "GOOGLE_API_KEY" not in os.environ:
            os.environ["GOOGLE_API_KEY"] = colab_key
        if "GEMINI_API_KEY" not in os.environ:
            os.environ["GEMINI_API_KEY"] = colab_key
except (ImportError, Exception):
    pass

# Initialize the Gemini Client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Initialize the Agno Agent for UI Validation Audit
data_cleaning_agent = Agent(
    name="Data Schema Auditor",
    model=Gemini(id="gemini-2.5-flash"),
    instructions=[
        "You are an expert data science agent.",
        "Audit and validate the data mapping structure of the provided sample matrix.",
        "Provide a clear breakdown of recommendations, structural issues, and potential data quality anomalies."
    ]
)

class FullyAgenticValidator:
    def __init__(self, model="gemini-2.5-flash"):
        self.model = model

    def _call_llm_with_backoff(self, prompt, system_prompt="You are an expert data science agent.", max_retries=3):
        """Helper to guarantee clean string responses from Gemini with dynamic backoff if rate limits occur."""
        wait_time = 2
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.0
                    )
                )
                return response.text.strip()
            except APIError as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    if attempt < max_retries - 1:
                        print(f"⚠️ Rate limit hit. Backing off for {wait_time}s (Attempt {attempt+1}/{max_retries})...")
                        time.sleep(wait_time)
                        wait_time *= 2
                        continue
                print(f"❌ API Error: {e}")
                return None
            except Exception as e:
                print(f"❌ Unexpected Error: {e}")
                return None
        return None

    def profile_and_plan(self, df, file_name="Dataset"):
        """STAGE 1: COGNITIVE INSPECTION & STRATEGY PLANNING"""
        print(f"\n🕵️‍♂️ Analyzing structural anomalies for: {file_name}...")

        data_profile = {}
        for col in df.columns:
            missing_count = int(df[col].isnull().sum())
            missing_pct = (missing_count / len(df)) * 100
            unique_count = int(df[col].nunique())
            sample_vals = df[col].dropna().head(3).astype(str).tolist()

            if pd.api.types.is_numeric_dtype(df[col]):
                col_type = "numeric"
                skew = df[col].skew() if len(df[col].dropna()) > 2 else 0
                stats = {"skew": round(float(skew), 2) if not pd.isna(skew) else 0, "min": float(df[col].min()), "max": float(df[col].max())}
            else:
                col_type = "text/categorical"
                stats = {}

            data_profile[col] = {
                "type": col_type,
                "missing_pct": round(missing_pct, 1),
                "unique_values_count": unique_count,
                "sample_values": sample_vals,
                "metrics": stats
            }

        prompt = f"""
        You are a principal data scientist. Inspect this dataset schema and profiling data:
        {json.dumps(data_profile, indent=2)}

        Tasks:
        1. Classify each column strictly into 'text_processing' or 'numeric_processing'.
        2. For numeric columns with missing data (missing_pct > 0), choose the mathematically superior imputation strategy ('mean', 'median', or 'zero') based on the context, samples, and skewness metrics.
        3. For text columns, determine if they look like categorical/name data that requires spelling standardisation ('spell_check': true/false).

        Return strictly a valid JSON object matching this structure exactly. Do not include markdown wraps.
        {{
          "columns": {{
             "column_name_1": {{"action": "numeric_processing", "impute_strategy": "median"}},
             "column_name_2": {{"action": "text_processing", "spell_check": true}}
          }}
        }}
        """
        
        raw_plan = self._call_llm_with_backoff(prompt, "You output raw, valid JSON maps only. No markdown formatting.")
        if raw_plan and raw_plan.startswith("```"):
            raw_plan = raw_plan.strip("```json").strip("```").strip()
            
        try:
            plan = json.loads(raw_plan)["columns"]
            return plan
        except Exception:
            print("⚠️ Profiling parser failed; applying standard fallback rules.")
            return {col: {"action": "numeric_processing" if pd.api.types.is_numeric_dtype(df[col]) else "text_processing", "impute_strategy": "median", "spell_check": True} for col in df.columns}

    def clean_numeric(self, df, col, strategy):
        """STAGE 2A: DETERMINISTIC NUMERIC CLEANING & STRATEGIC IMPUTATION"""
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(r'[^\d\.]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')

        if df[col].isnull().any():
            if strategy == "mean":
                fill_value = df[col].mean()
            elif strategy == "median":
                fill_value = df[col].median()
            else:
                fill_value = 0
            
            if pd.isna(fill_value): fill_value = 0
            df[col] = df[col].fillna(fill_value)
            print(f"   🔢 Filled missing rows in '{col}' using calculated {strategy} ({round(fill_value, 2)})")
        return df

    def clean_text_with_reflection(self, df, col):
        """STAGE 2B: AGENTIC TEXT STANDARDIZATION WITH RESILIENT JSON PARSING"""
        df[col] = df[col].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
        unique_dirty = df[col].dropna().unique().tolist()

        if not unique_dirty or len(unique_dirty) > 80:
            return df

        gen_prompt = f"""
        Analyze these unique raw string terms from the database column '{col}':
        {unique_dirty}

        Instructions:
        - Correct typos and spelling errors.
        - Convert all records to strict Title Case capitalization.
        - Merge duplicate variations of identical business entities.

        Return strictly a valid JSON object mapping old dirty terms to new clean terms. 
        Do not include markdown wraps or code block syntax.
        Format: {{"old_string": "New Clean String"}}
        """
        try:
            raw_map = self._call_llm_with_backoff(gen_prompt, "You output raw, valid JSON structures only. No formatting wraps.")
            
            if not raw_map:
                print(f"   ⚠️ API token pool empty or server busy. Skipping text mapping for '{col}'.")
                return df

            if raw_map.startswith("```"):
                raw_map = raw_map.strip("```json").strip("```python").strip("```").strip()
            
            cleaning_map = json.loads(raw_map)

            reflect_prompt = f"""
            Review this proposed translation dictionary built for data cleaning:
            {json.dumps(cleaning_map)}

            Act as an unyielding data auditor. Ensure the keys did not have their core semantic business identities swapped mistakenly.
            Identify any invalid or highly dangerous keys. Return strictly a valid JSON list of strings containing the keys that must be REJECTED.
            Format: ["bad_key_1"]
            If all changes look perfectly safe, return an empty list: []
            """
            raw_reflection = self._call_llm_with_backoff(reflect_prompt, "You output raw, valid JSON lists only.")
            
            if raw_reflection:
                if raw_reflection.startswith("```"):
                    raw_reflection = raw_reflection.strip("```json").strip("```python").strip("```").strip()
                rejected_keys = json.loads(raw_reflection)

                if rejected_keys:
                    print(f"   🚫 Reflection Loop intercepted {len(rejected_keys)} unsafe alterations! Overruling: {rejected_keys}")
                    for key in rejected_keys:
                        if key in cleaning_map: del cleaning_map[key]

            df[col] = df[col].map(cleaning_map).fillna(df[col])
        except Exception as e:
            print(f"   ⚠️ Skipping LLM syntax map for '{col}' due to formatting parsing error: {e}")
        return df

    def execute_agent_pipeline(self, df, file_name="Dataset"):
        """Main Orchestrator loop for processing a dataframe directly."""
        cleaning_blueprint = self.profile_and_plan(df, file_name)

        for col, specifications in cleaning_blueprint.items():
            if col not in df.columns: continue
            
            action = specifications.get("action")
            if action == "numeric_processing":
                df = self.clean_numeric(df, col, specifications.get("impute_strategy", "median"))
            elif action == "text_processing":
                if specifications.get("spell_check", True):
                    print(f"   🔤 Deploying spellcheck and deduplication loop on '{col}'...")
                    df = self.clean_text_with_reflection(df, col)
        return df

# ----------------------------------------------------
# Streamlit Cache Functions for Ultimate Speed
# ----------------------------------------------------
@st.cache_data
def get_cached_audit(sample_data_str):
    """Caches the heavy audit report to prevent repeated API calls on structural re-runs."""
    query = f"Audit and validate the data mapping structure of this sample matrix:\n{sample_data_str}"
    response = data_cleaning_agent.run(query)
    return response.content

# ----------------------------------------------------
# UI FOR STREAMLIT APPLICATION
# ----------------------------------------------------
st.title("🛡️ Fast Agentic Data Validation Auditor")
st.subheader("📁 Upload Actuarial Dataset")
uploaded_file = st.file_uploader("Choose a CSV or text file to analyze:", type="csv")

if uploaded_file is not None:
    df_preview = pd.read_csv(uploaded_file)
    st.write("📊 **Raw File Preview (First 5 rows):**")
    st.dataframe(df_preview.head())
    
    # --- Action Section 1: UI Audit Report ---
    if st.button("Execute Agent Validation Audit"):
        with st.spinner("Agent parsing schema strings and conducting fast sanity checks..."):
            try:
                sample_data = df_preview.head(10).to_string()
                report_content = get_cached_audit(sample_data)
                st.success("✓ Data Schema Audit Completed!")
                st.markdown(report_content)
            except Exception as e:
                st.error(f"Validation workflow pipeline hit an execution error: {e}")
                
    # --- Action Section 2: In-App Clean & Impute Pipeline ---
    if st.button("Run Adaptive Cleaning Pipeline"):
        with st.spinner("Executing structural code corrections and processing data schema..."):
            try:
                validator = FullyAgenticValidator()
                cleaned_df = validator.execute_agent_pipeline(df_preview.copy(), uploaded_file.name)
                
                st.success("🎉 Processing completed successfully with zero hardcoded delay drops!")
                st.write("📊 **Cleaned Data Preview:**")
                st.dataframe(cleaned_df.head())
                
                csv_bytes = cleaned_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Cleaned Dataset",
                    data=csv_bytes,
                    file_name=uploaded_file.name.replace(".csv", "_cleaned.csv"),
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Cleaning execution pipeline error: {e}")

# ----------------------------------------------------
# Optional Local Folder Batch Tooling (Safely Encapsulated)
# ----------------------------------------------------
st.markdown("---")
st.subheader("🤖 Batch Local Validation Tool")
st.caption("Clean raw files stored in your project's data directory locally without halting your browser interface.")

if st.button("Scan & Batch Clean Local Folder"):
    with st.spinner("Locating datasets in data/ directory..."):
        agent = FullyAgenticValidator()
        data_folder = "data"
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
            
        csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
        files_to_clean = [f for f in csv_files if "_cleaned" not in f]
        
        if not files_to_clean:
            st.warning(f"No raw CSV files found to process in local folder '{data_folder}'!")
        else:
            st.info(f"Found {len(files_to_clean)} local targets. Beginning direct execution processing...")
            for file_path in files_to_clean:
                st.write(f"🔄 Currently processing: `{os.path.basename(file_path)}`")
                raw_df = pd.read_csv(file_path)
                processed_df = agent.execute_agent_pipeline(raw_df, os.path.basename(file_path))
                
                output_path = file_path.replace(".csv", "_cleaned.csv")
                processed_df.to_csv(output_path, index=False)
                st.success(f"💾 Saved fully validated asset to: `{output_path}`")
            st.success("🎉 All local datasets have been successfully processed!")
