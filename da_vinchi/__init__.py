"""
Da Vinchi - Feature Engineering Pipeline for Financial Markets

An 8-stage modular pipeline for comprehensive feature engineering:
- Stage 0: Data hygiene and preprocessing
- Stage 1: OHLCV technical features 
- Stage 2: Cross-sectional risk and market context
- Stage 3: Regime and seasonal structure
- Stage 4: Instrument relationships and correlations
- Stage 5: Cross-instrument enriched features
- Stage 6: Alternative data (news, sentiment, macro)
- Stage 7: Target labeling and forward returns
- Stage 8: Model-ready dataset assembly

Each stage is modular and configurable, with middleware for data routing,
model selection, and configuration management.
"""

from .interfaces.driver_api import DaVinchiDriver
from .core.pipeline_manager import PipelineManager
from .core.stage_base import StageBase, FeatureStage, ValidationStage, StageData, StageMetadata

__version__ = "1.0.0"
__author__ = "Da Vinchi Pipeline Team"

# Main API exports
__all__ = [
    "DaVinchiDriver",
    "PipelineManager", 
    "StageBase",
    "FeatureStage",
    "ValidationStage",
    "StageData",
    "StageMetadata"
]