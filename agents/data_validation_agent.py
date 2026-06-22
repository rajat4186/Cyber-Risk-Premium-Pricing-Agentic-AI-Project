"""
Data Cleaning and Validation Agent
Capstone Project

Purpose:
- Load CSV files
- Validate data quality
- Clean common issues
- Generate validation report
- Save cleaned dataset
"""

import pandas as pd
import numpy as np
import os


class DataValidationAgent:

    def __init__(self):
        self.report = []

    # --------------------------------------------------
    # Load Dataset
    # --------------------------------------------------

    def load_data(self, file_path):

        try:

            df = pd.read_csv(file_path)

            self.report.append(
                f"\nDataset loaded successfully: {file_path}"
            )

            return df

        except Exception as e:

            self.report.append(
                f"Error loading dataset {file_path}: {e}"
            )

            return None

    # --------------------------------------------------
    # Missing Values Check
    # --------------------------------------------------

    def check_missing_values(self, df):

        missing = df.isnull().sum()

        self.report.append("=== Missing Values ===")

        for col, count in missing.items():

            if count > 0:

                self.report.append(
                    f"{col}: {count}"
                )

        return missing

    # --------------------------------------------------
    # Handle Missing Values
    # --------------------------------------------------

    def clean_missing_values(self, df):

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns

        for col in numeric_columns:

            df[col] = df[col].fillna(
                df[col].median()
            )

        text_columns = df.select_dtypes(
            include="object"
        ).columns

        for col in text_columns:

            df[col] = df[col].fillna(
                "Unknown"
            )

        self.report.append(
            "Missing values cleaned."
        )

        return df

    # --------------------------------------------------
    # Duplicate Check
    # --------------------------------------------------

    def check_duplicates(self, df):

        duplicates = df.duplicated().sum()

        self.report.append(
            f"Duplicate Rows Found: {duplicates}"
        )

        return duplicates

    # --------------------------------------------------
    # Remove Duplicates
    # --------------------------------------------------

    def remove_duplicates(self, df):

        before = len(df)

        df = df.drop_duplicates()

        after = len(df)

        removed = before - after

        self.report.append(
            f"Duplicate Rows Removed: {removed}"
        )

        return df

    # --------------------------------------------------
    # Numeric Validation
    # --------------------------------------------------

    def validate_numeric_columns(
        self,
        df,
        numeric_columns
    ):

        self.report.append(
            "=== Numeric Validation ==="
        )

        for col in numeric_columns:

            if col in df.columns:

                invalid = pd.to_numeric(
                    df[col],
                    errors="coerce"
                ).isna().sum()

                self.report.append(
                    f"{col}: {invalid} invalid values"
                )

                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )

        return df

    # --------------------------------------------------
    # Range Checks
    # --------------------------------------------------

    def range_checks(
        self,
        df,
        numeric_columns
    ):

        self.report.append(
            "=== Range Checks ==="
        )

        for col in numeric_columns:

            if col in df.columns:

                negatives = (
                    df[col] < 0
                ).sum()

                self.report.append(
                    f"{col}: {negatives} negative values"
                )

    # --------------------------------------------------
    # Date Validation
    # --------------------------------------------------

    def validate_dates(
        self,
        df,
        date_columns
    ):

        self.report.append(
            "=== Date Validation ==="
        )

        for col in date_columns:

            if col in df.columns:

                converted = pd.to_datetime(
                    df[col],
                    errors="coerce"
                )

                invalid = (
                    converted.isna().sum()
                )

                self.report.append(
                    f"{col}: {invalid} invalid dates"
                )

                df[col] = converted

        return df

    # --------------------------------------------------
    # Business Rules
    # --------------------------------------------------

    def business_rules(self, df):

        self.report.append(
            "=== Business Rules ==="
        )

        if (
            "incident_date" in df.columns
            and "disclosure_date" in df.columns
        ):

            invalid_dates = (
                df["disclosure_date"]
                < df["incident_date"]
            ).sum()

            self.report.append(
                f"Disclosure before Incident: {invalid_dates}"
            )

        if "employee_count" in df.columns:

            invalid_emp = (
                df["employee_count"] <= 0
            ).sum()

            self.report.append(
                f"Invalid Employee Counts: {invalid_emp}"
            )

    # --------------------------------------------------
    # Data Quality Score
    # --------------------------------------------------

    def quality_score(self, df):

        score = 100

        missing = df.isnull().sum().sum()

        duplicates = df.duplicated().sum()

        score -= min(missing, 20)

        score -= min(duplicates, 10)

        score = max(score, 0)

        self.report.append(
            f"Data Quality Score: {score}/100"
        )

    # --------------------------------------------------
    # Save Clean Dataset
    # --------------------------------------------------

    def save_clean_dataset(
        self,
        df,
        output_path
    ):

        # Safely generate target folder structure if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df.to_csv(
            output_path,
            index=False
        )

        self.report.append(
            f"Clean dataset saved to: {output_path}"
        )

    # --------------------------------------------------
    # Generate Report
    # --------------------------------------------------

    def generate_report(self):

        print("\n".join(self.report))

    # --------------------------------------------------
    # Main Run Function
    # --------------------------------------------------

    def run(
        self,
        input_file,
        output_file,
        numeric_columns=None,
        date_columns=None
    ):

        df = self.load_data(input_file)

        if df is None:
            return

        # Default to empty lists if nothing is supplied
        numeric_columns = numeric_columns if numeric_columns else []
        date_columns = date_columns if date_columns else []

        # 1. Identify raw issues and duplicates first
        self.check_missing_values(df)
        self.check_duplicates(df)
        df = self.remove_duplicates(df)

        # 2. Re-format and validate metrics (coerces string errors into NaNs)
        df = self.validate_numeric_columns(df, numeric_columns)
        self.range_checks(df, numeric_columns)
        df = self.validate_dates(df, date_columns)

        # 3. Check custom contextual business logic
        self.business_rules(df)

        # 4. Clean up all missing values (including structural ones generated above)
        df = self.clean_missing_values(df)

        # 5. Conclude evaluation metrics and serialize out
        self.quality_score(df)
        self.save_clean_dataset(df, output_file)


# --------------------------------------------------
# Execution Target Block
# --------------------------------------------------

if __name__ == "__main__":

    agent = DataValidationAgent()

    # --- 1. Process financial_impact.csv ---
    agent.run(
        input_file="data/financial_impact.csv",
        output_file="data/cleaned_financial_impact.csv",
        numeric_columns=[
            "direct_loss_usd", "ransom_demanded_usd", "ransom_paid_usd", 
            "recovery_cost_usd", "legal_fees_usd", "regulatory_fine_usd", 
            "insurance_payout_usd", "total_loss_usd", "total_loss_lower_bound", 
            "total_loss_upper_bound", "inflation_adjusted_usd", "cpi_index_used"
        ],
        date_columns=["created_at", "updated_at"]
    )

    # --- 2. Process incidents_master.csv ---
    agent.run(
        input_file="data/incidents_master.csv",
        output_file="data/cleaned_incidents_master.csv",
        numeric_columns=[
            "company_revenue_usd", "employee_count", "data_compromised_records", 
            "downtime_hours", "quality_score"
        ],
        date_columns=[
            "incident_date", "discovery_date", "disclosure_date", 
            "created_at", "updated_at"
        ]
    )

    # --- 3. Process market_impact.csv ---
    agent.run(
        input_file="data/market_impact.csv",
        output_file="data/cleaned_market_impact.csv",
        numeric_columns=[
            "price_7d_before", "price_disclosure_day", "price_1d_after", 
            "price_7d_after", "price_30d_after", "volume_avg_30d_baseline", 
            "volume_disclosure_day", "sector_return_same_period", "abnormal_return_1d", 
            "abnormal_return_7d", "abnormal_return_30d", "car_neg1_to_pos1", 
            "car_0_to_7", "car_0_to_30", "car_0_to_90", "t_statistic_1d", 
            "p_value_1d", "t_statistic_30d", "p_value_30d", "market_cap_at_disclosure", 
            "volume_ratio_disclosure", "pre_incident_volatility_30d", 
            "post_incident_volatility_30d", "days_to_price_recovery"
        ],
        date_columns=["created_at", "updated_at"]
    )

    # Print the aggregated report across all processing cycles
    agent.generate_report()
