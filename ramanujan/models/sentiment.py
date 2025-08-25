"""
Sentiment Analysis Models for Financial Text Processing

This module provides various sentiment analysis models specialized for financial text,
including news articles, earnings calls, social media posts, and analyst reports.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
import warnings
from datetime import datetime

from ..base import SupervisedModel
from ..config import ModelConfig
from ..exceptions import ModelError, TrainingError


class FinBERTSentimentModel(SupervisedModel):
    """
    Financial BERT-based sentiment analysis model
    
    Uses pre-trained financial BERT models for sentiment classification
    of financial text data.
    """
    
    def _build_model(self):
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            from transformers import pipeline
        except ImportError:
            raise ModelError("Transformers not installed. Install with: pip install transformers torch")
        
        # Default to FinBERT model trained on financial data
        model_name = self.config.parameters.get('model_name', 'ProsusAI/finbert')
        
        try:
            # Create sentiment analysis pipeline
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Create pipeline for easy inference
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1,  # CPU by default, set to 0 for GPU
                return_all_scores=True
            )
            
            return self.pipeline
            
        except Exception as e:
            # Fallback to simpler model
            warnings.warn(f"Failed to load {model_name}, falling back to basic model: {e}")
            self.pipeline = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                return_all_scores=True
            )
            return self.pipeline
    
    def _train_model(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """
        Fine-tune the pre-trained model on custom data
        Note: This is simplified - full fine-tuning would require more setup
        """
        if 'text' not in X.columns:
            raise TrainingError("Text column required for sentiment analysis")
        
        # For now, we'll use the pre-trained model without fine-tuning
        # In production, you'd implement proper fine-tuning here
        
        # Evaluate on provided data
        texts = X['text'].tolist()
        predictions = []
        confidences = []
        
        for text in texts:
            result = self.pipeline(text)
            # Extract sentiment with highest confidence
            best_pred = max(result, key=lambda x: x['score'])
            predictions.append(best_pred['label'])
            confidences.append(best_pred['score'])
        
        # Calculate accuracy if labels are provided
        if y is not None and len(y) > 0:
            # Map predictions to numeric if needed
            pred_numeric = self._map_labels_to_numeric(predictions)
            y_numeric = self._map_labels_to_numeric(y.tolist())
            
            accuracy = np.mean(np.array(pred_numeric) == np.array(y_numeric))
        else:
            accuracy = 0.0
        
        return {
            'accuracy': accuracy,
            'avg_confidence': np.mean(confidences),
            'model_type': 'finbert_sentiment'
        }
    
    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        """Internal prediction method for text data"""
        if 'text' not in X.columns:
            raise ValueError("Text column required for prediction")
        
        texts = X['text'].tolist()
        predictions = []
        
        for text in texts:
            if pd.isna(text) or text.strip() == '':
                predictions.append(0.0)  # Neutral for empty text
                continue
                
            try:
                result = self.pipeline(text)
                # Convert to numeric sentiment score
                sentiment_score = self._extract_sentiment_score(result)
                predictions.append(sentiment_score)
            except Exception as e:
                warnings.warn(f"Failed to process text: {e}")
                predictions.append(0.0)
        
        return np.array(predictions)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict sentiment for text data"""
        return self._predict(X)
    
    def _extract_sentiment_score(self, pipeline_result: List[Dict]) -> float:
        """
        Convert pipeline result to numeric sentiment score
        
        Returns:
            float: Sentiment score between -1 (negative) and 1 (positive)
        """
        # Handle different model output formats
        if isinstance(pipeline_result, list) and len(pipeline_result) > 0:
            if isinstance(pipeline_result[0], list):
                # Multiple scores returned
                scores = pipeline_result[0]
            else:
                scores = pipeline_result
            
            # Map to numeric score
            sentiment_score = 0.0
            for item in scores:
                label = item['label'].lower()
                score = item['score']
                
                if 'positive' in label or 'pos' in label:
                    sentiment_score += score
                elif 'negative' in label or 'neg' in label:
                    sentiment_score -= score
                # Neutral contributes 0
        else:
            sentiment_score = 0.0
        
        # Ensure score is in [-1, 1] range
        return np.clip(sentiment_score, -1.0, 1.0)
    
    def _map_labels_to_numeric(self, labels: List[str]) -> List[float]:
        """Map string labels to numeric values"""
        numeric = []
        for label in labels:
            label_lower = str(label).lower()
            if 'positive' in label_lower or 'pos' in label_lower:
                numeric.append(1.0)
            elif 'negative' in label_lower or 'neg' in label_lower:
                numeric.append(-1.0)
            else:
                numeric.append(0.0)
        return numeric


class VaderSentimentModel(SupervisedModel):
    """
    VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis
    
    Rule-based sentiment analysis tool that is specifically attuned to social media text
    """
    
    def _build_model(self):
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        except ImportError:
            raise ModelError("vaderSentiment not installed. Install with: pip install vaderSentiment")
        
        self.analyzer = SentimentIntensityAnalyzer()
        return self.analyzer
    
    def _train_model(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """
        VADER is rule-based and doesn't require training
        This method evaluates performance on provided data
        """
        if 'text' not in X.columns:
            raise TrainingError("Text column required for sentiment analysis")
        
        # Evaluate on provided data
        predictions = self._predict(X)
        
        if y is not None and len(y) > 0:
            # Calculate correlation with provided labels
            correlation = np.corrcoef(predictions, y.values)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
        
        return {
            'correlation': correlation,
            'model_type': 'vader_sentiment',
            'mean_sentiment': np.mean(predictions)
        }
    
    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        """Internal prediction method using VADER"""
        if 'text' not in X.columns:
            raise ValueError("Text column required for prediction")
        
        predictions = []
        
        for text in X['text']:
            if pd.isna(text) or text.strip() == '':
                predictions.append(0.0)
                continue
            
            try:
                scores = self.analyzer.polarity_scores(text)
                # Use compound score as overall sentiment
                predictions.append(scores['compound'])
            except Exception as e:
                warnings.warn(f"Failed to process text: {e}")
                predictions.append(0.0)
        
        return np.array(predictions)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict sentiment using VADER"""
        return self._predict(X)


class TextBlobSentimentModel(SupervisedModel):
    """
    TextBlob sentiment analysis model
    
    Simple and fast sentiment analysis using TextBlob library
    """
    
    def _build_model(self):
        try:
            from textblob import TextBlob
        except ImportError:
            raise ModelError("TextBlob not installed. Install with: pip install textblob")
        
        self.TextBlob = TextBlob
        return self.TextBlob
    
    def _train_model(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """
        TextBlob is rule-based and doesn't require training
        This method evaluates performance on provided data
        """
        if 'text' not in X.columns:
            raise TrainingError("Text column required for sentiment analysis")
        
        # Evaluate on provided data
        predictions = self._predict(X)
        
        if y is not None and len(y) > 0:
            correlation = np.corrcoef(predictions, y.values)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
        
        return {
            'correlation': correlation,
            'model_type': 'textblob_sentiment',
            'mean_sentiment': np.mean(predictions),
            'sentiment_range': [float(np.min(predictions)), float(np.max(predictions))]
        }
    
    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        """Internal prediction method using TextBlob"""
        if 'text' not in X.columns:
            raise ValueError("Text column required for prediction")
        
        predictions = []
        
        for text in X['text']:
            if pd.isna(text) or text.strip() == '':
                predictions.append(0.0)
                continue
            
            try:
                blob = self.TextBlob(text)
                # Polarity ranges from -1 to 1
                predictions.append(blob.sentiment.polarity)
            except Exception as e:
                warnings.warn(f"Failed to process text: {e}")
                predictions.append(0.0)
        
        return np.array(predictions)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict sentiment using TextBlob"""
        return self._predict(X)


class CustomFinancialSentimentModel(SupervisedModel):
    """
    Custom financial sentiment model using traditional ML approaches
    
    This model can be trained on custom financial text data using various
    feature extraction methods (TF-IDF, word embeddings) and classifiers.
    """
    
    def _build_model(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        
        # Default parameters
        default_params = {
            'max_features': 5000,
            'ngram_range': (1, 2),
            'min_df': 2,
            'max_df': 0.95,
            'stop_words': 'english',
            'C': 1.0,
            'random_state': 42
        }
        
        params = {**default_params, **self.config.parameters}
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=params['max_features'],
                ngram_range=params['ngram_range'],
                min_df=params['min_df'],
                max_df=params['max_df'],
                stop_words=params['stop_words']
            )),
            ('classifier', LogisticRegression(
                C=params['C'],
                random_state=params['random_state'],
                max_iter=1000
            ))
        ])
        
        return self.pipeline
    
    def _train_model(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """Train the custom sentiment model"""
        if 'text' not in X.columns:
            raise TrainingError("Text column required for sentiment analysis")
        
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import classification_report
        
        texts = X['text'].fillna('').astype(str)
        
        # Train the model
        self.pipeline.fit(texts, y)
        
        # Cross-validation
        cv_scores = cross_val_score(self.pipeline, texts, y, cv=5, scoring='accuracy')
        
        # Generate predictions for analysis
        predictions = self.pipeline.predict(texts)
        
        return {
            'cv_accuracy_mean': float(np.mean(cv_scores)),
            'cv_accuracy_std': float(np.std(cv_scores)),
            'train_accuracy': float(np.mean(predictions == y)),
            'model_type': 'custom_financial_sentiment',
            'n_features': self.pipeline.named_steps['tfidf'].get_feature_names_out().shape[0]
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict sentiment using trained model"""
        if 'text' not in X.columns:
            raise ValueError("Text column required for prediction")
        
        texts = X['text'].fillna('').astype(str)
        
        # Get prediction probabilities and convert to sentiment scores
        try:
            probabilities = self.pipeline.predict_proba(texts)
            # Convert to sentiment score assuming classes are [-1, 0, 1] or [0, 1]
            if probabilities.shape[1] == 2:
                # Binary classification: convert to [-1, 1] scale
                sentiment_scores = probabilities[:, 1] * 2 - 1
            else:
                # Multi-class: weight by class values
                classes = self.pipeline.classes_
                sentiment_scores = np.sum(probabilities * classes.reshape(1, -1), axis=1)
            
            return sentiment_scores
        except:
            # Fallback to discrete predictions
            predictions = self.pipeline.predict(texts)
            return predictions.astype(float)


class NewsAggregationModel:
    """
    Model for aggregating news sentiment at instrument-day level
    
    This is not a traditional ML model but a specialized aggregation system
    for financial news sentiment data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.aggregation_methods = config.get('aggregation_methods', ['mean', 'volume_weighted', 'recency_weighted'])
        self.time_decay_lambda = config.get('time_decay_lambda', 0.1)
        self.min_articles = config.get('min_articles', 1)
    
    def aggregate_sentiment(self, news_data: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate news sentiment by instrument and date
        
        Expected columns in news_data:
        - date: publication date
        - instrument: instrument symbol
        - sentiment: sentiment score
        - headline: article headline
        - source: news source
        - timestamp: exact publication time
        """
        
        if news_data.empty:
            return pd.DataFrame()
        
        # Ensure required columns exist
        required_cols = ['date', 'instrument', 'sentiment']
        missing_cols = [col for col in required_cols if col not in news_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Group by instrument and date
        grouped = news_data.groupby(['instrument', 'date'])
        
        aggregated_results = []
        
        for (instrument, date), group in grouped:
            if len(group) < self.min_articles:
                continue
            
            agg_data = {
                'instrument': instrument,
                'date': date,
                'article_count': len(group),
                'unique_sources': group['source'].nunique() if 'source' in group.columns else 1
            }
            
            # Basic aggregations
            sentiments = group['sentiment'].dropna()
            if len(sentiments) == 0:
                continue
            
            agg_data.update({
                'sentiment_mean': sentiments.mean(),
                'sentiment_std': sentiments.std() if len(sentiments) > 1 else 0.0,
                'sentiment_median': sentiments.median(),
                'sentiment_min': sentiments.min(),
                'sentiment_max': sentiments.max()
            })
            
            # Sentiment polarity distribution
            positive_pct = (sentiments > 0.1).mean()
            negative_pct = (sentiments < -0.1).mean()
            neutral_pct = 1 - positive_pct - negative_pct
            
            agg_data.update({
                'positive_pct': positive_pct,
                'negative_pct': negative_pct,
                'neutral_pct': neutral_pct,
                'sentiment_skew': sentiments.skew()
            })
            
            # Time-weighted sentiment (if timestamp available)
            if 'timestamp' in group.columns:
                # Calculate recency weights
                timestamps = pd.to_datetime(group['timestamp'])
                latest_time = timestamps.max()
                time_diffs = (latest_time - timestamps).dt.total_seconds() / 3600  # Hours
                recency_weights = np.exp(-self.time_decay_lambda * time_diffs)
                
                weighted_sentiment = np.average(sentiments, weights=recency_weights)
                agg_data['sentiment_recency_weighted'] = weighted_sentiment
            
            # Volume-weighted sentiment (if article importance/volume available)
            if 'importance' in group.columns:
                importance = group['importance'].fillna(1.0)
                volume_weighted_sentiment = np.average(sentiments, weights=importance)
                agg_data['sentiment_volume_weighted'] = volume_weighted_sentiment
            
            # News shock indicator (unusual volume of news)
            # This would require historical context to calculate z-score
            agg_data['news_shock'] = min(len(group) / 5.0, 2.0)  # Simplified version
            
            aggregated_results.append(agg_data)
        
        return pd.DataFrame(aggregated_results)
    
    def calculate_sentiment_momentum(self, aggregated_data: pd.DataFrame, 
                                   windows: List[int] = [3, 7, 14]) -> pd.DataFrame:
        """Calculate sentiment momentum and changes"""
        
        if aggregated_data.empty:
            return aggregated_data
        
        # Sort by instrument and date
        data = aggregated_data.sort_values(['instrument', 'date']).copy()
        
        # Group by instrument to calculate rolling metrics
        grouped = data.groupby('instrument')
        
        for window in windows:
            # Rolling averages - use transform to maintain index alignment
            data[f'sentiment_ma_{window}d'] = grouped['sentiment_mean'].transform(
                lambda x: x.rolling(window).mean()
            )
            data[f'article_count_ma_{window}d'] = grouped['article_count'].transform(
                lambda x: x.rolling(window).mean()
            )
            
            # Sentiment momentum
            data[f'sentiment_momentum_{window}d'] = (
                data['sentiment_mean'] - data[f'sentiment_ma_{window}d']
            )
        
        # Day-over-day changes
        data['sentiment_change_1d'] = grouped['sentiment_mean'].diff()
        data['article_count_change_1d'] = grouped['article_count'].diff()
        
        # Sentiment volatility
        data['sentiment_volatility_7d'] = grouped['sentiment_mean'].transform(
            lambda x: x.rolling(7).std()
        )
        
        return data