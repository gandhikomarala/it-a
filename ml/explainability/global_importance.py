# Global feature importance aggregator across customer cohorts.
from typing import Dict, List
from packages.schemas.feature import FeatureImportanceSchema

class GlobalImportanceAggregator:
    @staticmethod
    def compute_global_rankings(importances_dict: Dict[str, float]) -> List[FeatureImportanceSchema]:
        sorted_feats = sorted(importances_dict.items(), key=lambda x: x[1], reverse=True)
        total_score = sum(importances_dict.values()) or 1.0

        rankings: List[FeatureImportanceSchema] = []
        for rank, (feat, score) in enumerate(sorted_feats, start=1):
            norm = round(score / total_score, 4)
            rankings.append(FeatureImportanceSchema(
                feature_name=feat,
                importance=round(score, 4),
                normalized_importance=norm,
                rank=rank
            ))
        return rankings
