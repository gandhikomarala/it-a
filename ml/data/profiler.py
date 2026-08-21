# Automated dataset profiling, schema inspection, and data quality scoring.
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from packages.schemas.dataset import (
    DatasetProfileSchema, ColumnDistributionSchema, DataQualityReportSchema
)
from packages.shared.enums import QualityScoreTier
from packages.utilities.math_stats import compute_percentiles
from packages.logging.logger import get_logger

logger = get_logger(__name__)

class DatasetProfiler:
    # Inspects data structure, calculates distributions, and computes quality score.

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.row_count = len(df)
        self.column_count = len(df.columns)

    def generate_profile(self) -> DatasetProfileSchema:
        # Generate comprehensive distribution profile across all columns.
        column_profiles: List[ColumnDistributionSchema] = []

        for col in self.df.columns:
            series = self.df[col]
            null_count = int(series.isnull().sum())
            null_pct = round((null_count / self.row_count) * 100, 2) if self.row_count > 0 else 0.0
            unique_count = int(series.nunique(dropna=True))

            is_numeric = pd.api.types.is_numeric_dtype(series)
            
            mean_val = None
            std_val = None
            min_val = None
            max_val = None
            median_val = None
            quantiles_dict = None
            top_cats = None
            outlier_cnt = 0

            if is_numeric:
                clean_series = series.dropna()
                if len(clean_series) > 0:
                    mean_val = float(clean_series.mean())
                    std_val = float(clean_series.std()) if len(clean_series) > 1 else 0.0
                    min_val = float(clean_series.min())
                    max_val = float(clean_series.max())
                    median_val = float(clean_series.median())
                    quantiles_dict = compute_percentiles(clean_series.tolist())

                    q25 = quantiles_dict.get("p25", min_val)
                    q75 = quantiles_dict.get("p75", max_val)
                    iqr = q75 - q25
                    lower_bound = q25 - 1.5 * iqr
                    upper_bound = q75 + 1.5 * iqr
                    outlier_cnt = int(((clean_series < lower_bound) | (clean_series > upper_bound)).sum())
            else:
                value_counts = series.value_counts(dropna=True).head(5)
                top_cats = {str(k): int(v) for k, v in value_counts.items()}

            column_profiles.append(ColumnDistributionSchema(
                column_name=col,
                data_type=str(series.dtype),
                null_count=null_count,
                null_percentage=null_pct,
                unique_count=unique_count,
                mean=mean_val,
                std=std_val,
                min_value=min_val,
                max_value=max_val,
                median=median_val,
                quantiles=quantiles_dict,
                top_categories=top_cats,
                outlier_count=outlier_cnt
            ))

        duplicate_rows = int(self.df.duplicated().sum())
        duplicate_pct = round((duplicate_rows / self.row_count) * 100, 2) if self.row_count > 0 else 0.0
        memory_bytes = int(self.df.memory_usage(deep=True).sum())

        return DatasetProfileSchema(
            row_count=self.row_count,
            column_count=self.column_count,
            memory_usage_bytes=memory_bytes,
            columns=column_profiles,
            duplicate_rows_count=duplicate_rows,
            duplicate_rows_percentage=duplicate_pct
        )

    def evaluate_quality(self) -> DataQualityReportSchema:
        # Evaluate dataset quality across 4 pillars: Completeness, Validity, Uniqueness, Consistency.
        if self.row_count == 0:
            return DataQualityReportSchema(
                quality_score=0.0,
                quality_tier=QualityScoreTier.CRITICAL,
                completeness_score=0.0,
                validity_score=0.0,
                uniqueness_score=0.0,
                consistency_score=0.0,
                issues_detected=[{"severity": "CRITICAL", "message": "Dataset is completely empty"}],
                recommendations=["Provide a non-empty dataset with customer records"],
                is_approved=False
            )

        issues: List[Dict[str, Any]] = []
        recommendations: List[str] = []

        total_cells = self.row_count * self.column_count
        total_nulls = int(self.df.isnull().sum().sum())
        completeness_pct = max(0.0, 100.0 - (total_nulls / total_cells * 100.0))

        for col in self.df.columns:
            null_pct = (self.df[col].isnull().sum() / self.row_count) * 100.0
            if null_pct > 30.0:
                issues.append({"column": col, "severity": "HIGH", "message": f"High missing value rate: {null_pct:.1f}%"})
                recommendations.append(f"Consider imputing or dropping column '{col}' due to >30% missing values.")

        dup_count = int(self.df.duplicated().sum())
        uniqueness_pct = max(0.0, 100.0 - (dup_count / self.row_count * 100.0))
        if dup_count > 0:
            issues.append({"severity": "MEDIUM", "message": f"Detected {dup_count} duplicate customer rows."})
            recommendations.append("Deduplicate dataset rows before running model training.")

        validity_score = 100.0
        for col in self.df.select_dtypes(include=[np.number]).columns:
            if (col.lower() == "age" or col.lower().startswith("age_") or col.lower().endswith("_age")) and (self.df[col] < 0).any():
                validity_score -= 10.0
                issues.append({"column": col, "severity": "CRITICAL", "message": "Invalid negative ages detected."})
            elif "charge" in col.lower() and (self.df[col] < 0).any():
                validity_score -= 5.0
                issues.append({"column": col, "severity": "HIGH", "message": "Negative financial charges detected."})
            elif "spend" in col.lower() and (self.df[col] < 0).any():
                validity_score -= 5.0
                issues.append({"column": col, "severity": "HIGH", "message": "Negative spend values detected."})
        validity_score = max(0.0, validity_score)

        consistency_score = 100.0
        if "customer_id" in self.df.columns:
            cid_dups = int(self.df["customer_id"].duplicated().sum())
            if cid_dups > 0:
                consistency_score -= 20.0
                issues.append({"column": "customer_id", "severity": "HIGH", "message": f"{cid_dups} non-unique customer IDs found."})
                recommendations.append("Ensure 'customer_id' is uniquely assigned per customer.")

        overall_score = round(
            (completeness_pct * 0.35) +
            (validity_score * 0.30) +
            (uniqueness_pct * 0.20) +
            (consistency_score * 0.15),
            2
        )

        if overall_score >= 90.0:
            tier = QualityScoreTier.EXCELLENT
        elif overall_score >= 80.0:
            tier = QualityScoreTier.GOOD
        elif overall_score >= 70.0:
            tier = QualityScoreTier.ACCEPTABLE
        elif overall_score >= 50.0:
            tier = QualityScoreTier.POOR
        else:
            tier = QualityScoreTier.CRITICAL

        is_approved = overall_score >= 75.0 and len([i for i in issues if i.get("severity") == "CRITICAL"]) == 0

        return DataQualityReportSchema(
            quality_score=overall_score,
            quality_tier=tier,
            completeness_score=round(completeness_pct, 2),
            validity_score=round(validity_score, 2),
            uniqueness_score=round(uniqueness_pct, 2),
            consistency_score=round(consistency_score, 2),
            issues_detected=issues,
            recommendations=recommendations or ["Dataset meets enterprise quality guidelines."],
            is_approved=is_approved
        )
