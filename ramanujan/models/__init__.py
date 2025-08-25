"""
Model implementations for the Ramanujan ML framework
"""

# Prediction Models (Supervised)
from .prediction import (
    XGBoostModel,
    LightGBMModel,
    CatBoostModel,
    RandomForestModel,
    LinearRegressionModel,
    RidgeRegressionModel,
    LSTMModel,
    GARCHModel
)

# Clustering Models (Unsupervised)
from .clustering import (
    KMeansModel,
    GMMModel,
    HMMModel
)

# Correlation Models
from .correlation import (
    PearsonCorrelationModel,
    SpearmanCorrelationModel,
    KendallCorrelationModel,
    TailDependenceModel,
    MutualInformationModel
)

# Sentiment Analysis Models
from .sentiment import (
    FinBERTSentimentModel,
    VaderSentimentModel,
    TextBlobSentimentModel,
    CustomFinancialSentimentModel,
    NewsAggregationModel
)

__all__ = [
    # Prediction Models
    "XGBoostModel", "LightGBMModel", "CatBoostModel", "RandomForestModel",
    "LinearRegressionModel", "RidgeRegressionModel", "LSTMModel", "GARCHModel",
    # Clustering Models
    "KMeansModel", "GMMModel", "HMMModel", 
    # Correlation Models
    "PearsonCorrelationModel", "SpearmanCorrelationModel", "KendallCorrelationModel",
    "TailDependenceModel", "MutualInformationModel",
    # Sentiment Analysis Models
    "FinBERTSentimentModel", "VaderSentimentModel", "TextBlobSentimentModel",
    "CustomFinancialSentimentModel", "NewsAggregationModel"
]