# Regulatory Compliance & Explainability Auditor (GDPR, EU AI Act).
from typing import Dict, Any, List

class RegulatoryComplianceChecker:
    @staticmethod
    def audit_model_compliance(has_xai_explanations: bool, disparate_impact_ratio: float) -> Dict[str, Any]:
        gdpr_art22 = bool(has_xai_explanations)
        eu_ai_act_fairness = bool(disparate_impact_ratio >= 0.80)

        return {
            "gdpr_article_22_compliant": gdpr_art22,
            "eu_ai_act_transparency_compliant": gdpr_art22 and eu_ai_act_fairness,
            "compliance_summary": "Passed all algorithmic accountability and transparency criteria." if (gdpr_art22 and eu_ai_act_fairness) else "Remediation required."
        }
