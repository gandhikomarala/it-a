# Feature registry managing domain feature versions and metadata.
from typing import Dict, List, Optional
from packages.schemas.feature import FeatureDefinitionSchema, FeatureSetResponse
from .definitions import FEATURE_DEFINITIONS

class FeatureRegistry:
    def __init__(self):
        self._features: Dict[str, FeatureDefinitionSchema] = {
            f.name: f for f in FEATURE_DEFINITIONS
        }

    def register_feature(self, feature: FeatureDefinitionSchema) -> None:
        self._features[feature.name] = feature

    def get_feature(self, name: str) -> Optional[FeatureDefinitionSchema]:
        return self._features.get(name)

    def list_features(self, category: Optional[str] = None) -> List[FeatureDefinitionSchema]:
        if category:
            return [f for f in self._features.values() if f.category == category]
        return list(self._features.values())
