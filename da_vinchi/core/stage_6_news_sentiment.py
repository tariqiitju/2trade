#!/usr/bin/env python3
"""
Stage 6: News and Sentiment Analysis

This stage processes financial news data, performs sentiment analysis, and creates
sentiment-based features for use in prediction models. It integrates news data
from Odin's Eye and sentiment analysis models from Ramanujan.

Key Features:
- News data collection and filtering
- Multiple sentiment analysis models (FinBERT, VADER, TextBlob, Custom)
- Sentiment aggregation by instrument and date
- News volume and shock detection
- Sentiment momentum and trend analysis
- Integration with previous stages for enhanced features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import warnings
import json
from collections import defaultdict

from .stage_base import StageBase, StageData, StageMetadata

# Import Odin's Eye for news data
try:
    from odins_eye import OdinsEye, DateRange
    ODINS_EYE_AVAILABLE = True
except ImportError:
    ODINS_EYE_AVAILABLE = False
    warnings.warn("Odin's Eye not available. News data integration disabled.")

# Import Ramanujan sentiment models
try:
    from ramanujan.models.sentiment import (
        FinBERTSentimentModel, VaderSentimentModel, TextBlobSentimentModel,
        CustomFinancialSentimentModel, NewsAggregationModel
    )
    from ramanujan.config import ModelConfig
    RAMANUJAN_SENTIMENT_AVAILABLE = True
except ImportError:
    RAMANUJAN_SENTIMENT_AVAILABLE = False
    warnings.warn("Ramanujan sentiment models not available. Using basic sentiment analysis.")


class Stage6NewsSentiment(StageBase):
    """
    Stage 6: News and Sentiment Analysis
    
    This stage collects financial news data and performs sentiment analysis to create
    sentiment-based features for each instrument.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "Stage6_NewsSentiment", "1.0.0")
        
        # Extract stage-specific config
        stage_config = config.get('stages', {}).get('stage6_news_sentiment', {})
        
        self.params = {
            # News data collection
            'news_sources': stage_config.get('news_sources', ['all']),
            'lookback_days': stage_config.get('lookback_days', 30),
            'min_articles_per_day': stage_config.get('min_articles_per_day', 1),
            'max_articles_per_day': stage_config.get('max_articles_per_day', 100),
            
            # Sentiment analysis models
            'sentiment_models': stage_config.get('sentiment_models', ['vader', 'textblob']),
            'primary_model': stage_config.get('primary_model', 'vader'),
            'ensemble_sentiment': stage_config.get('ensemble_sentiment', True),
            
            # Feature generation
            'sentiment_windows': stage_config.get('sentiment_windows', [3, 7, 14, 30]),
            'create_momentum_features': stage_config.get('create_momentum_features', True),
            'create_shock_features': stage_config.get('create_shock_features', True),
            'create_volume_features': stage_config.get('create_volume_features', True),
            
            # Aggregation parameters
            'time_decay_lambda': stage_config.get('time_decay_lambda', 0.1),
            'importance_weighting': stage_config.get('importance_weighting', True),
            'source_reliability_weights': stage_config.get('source_reliability_weights', {}),
            
            # Text processing
            'min_article_length': stage_config.get('min_article_length', 50),
            'max_article_length': stage_config.get('max_article_length', 5000),
            'language_filter': stage_config.get('language_filter', 'english'),
            
            # Feature naming
            'feature_prefix': stage_config.get('feature_prefix', 'news'),
            
            # Performance
            'parallel_processing': stage_config.get('parallel_processing', False),
            'batch_size': stage_config.get('batch_size', 100)
        }
        
        # Initialize Odin's Eye
        if ODINS_EYE_AVAILABLE:
            try:
                self.odins_eye = OdinsEye()
            except Exception as e:
                self.logger.warning(f"Failed to initialize Odin's Eye: {e}")
                self.odins_eye = None
        else:
            self.odins_eye = None
        
        # Initialize sentiment models
        self.sentiment_models = {}
        self.news_aggregator = None
        self._initialize_sentiment_models()
        
        self.logger.info(f"Stage 6 initialized with {len(self.sentiment_models)} sentiment models")
        self.logger.info(f"News lookback: {self.params['lookback_days']} days")
        self.logger.info(f"Odin's Eye available: {ODINS_EYE_AVAILABLE}")
    
    def validate_inputs(self, data: pd.DataFrame) -> List[str]:
        """Validate input data for Stage 6"""
        errors = []
        
        # Check required columns
        required_columns = ['instrument']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check datetime index
        if not isinstance(data.index, pd.DatetimeIndex) and 'date' not in data.columns:
            errors.append("No datetime index or date column found")
        
        # Check if news data is available
        if not self.odins_eye:
            errors.append("Odin's Eye not available - cannot fetch news data")
        
        return errors
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names this stage will create"""
        prefix = self.params['feature_prefix']
        features = []
        
        # Basic sentiment features
        features.extend([
            f'{prefix}_sentiment_mean',
            f'{prefix}_sentiment_std',
            f'{prefix}_sentiment_median',
            f'{prefix}_article_count',
            f'{prefix}_unique_sources'
        ])
        
        # Sentiment distribution features
        features.extend([
            f'{prefix}_positive_pct',
            f'{prefix}_negative_pct', 
            f'{prefix}_neutral_pct',
            f'{prefix}_sentiment_skew'
        ])
        
        # Time-based features
        for window in self.params['sentiment_windows']:
            features.extend([
                f'{prefix}_sentiment_ma_{window}d',
                f'{prefix}_sentiment_momentum_{window}d',
                f'{prefix}_article_count_ma_{window}d'
            ])
        
        # Shock and volume features
        if self.params['create_shock_features']:
            features.extend([
                f'{prefix}_news_shock',
                f'{prefix}_sentiment_volatility_7d'
            ])
        
        # Recency-weighted sentiment
        features.append(f'{prefix}_sentiment_recency_weighted')
        
        return features
    
    def process(self, input_data) -> 'StageData':
        """Process data (Stage 6 works with StageData objects)"""
        if isinstance(input_data, pd.DataFrame):
            # Convert DataFrame to StageData if needed
            stage_data = StageData(
                data=input_data,
                metadata=StageMetadata("Stage6", "1.0.0"),
                config={'stages': {}},
                artifacts={}
            )
            return self._process_impl(stage_data)
        else:
            # Already StageData
            return self._process_impl(input_data)
    
    def _process_impl(self, input_data: StageData) -> StageData:
        """
        Main processing logic for Stage 6: News and Sentiment Analysis
        
        Args:
            input_data: StageData from previous stages
            
        Returns:
            StageData with news and sentiment features
        """
        data = input_data.data.copy()
        
        # Ensure datetime index
        data = self._ensure_datetime_index(data)
        
        # Get list of instruments
        instruments = self._get_instruments(data)
        
        self.logger.info(f"Processing news sentiment for {len(instruments)} instruments")
        
        # Determine date range for news collection
        start_date = data.index.min() - timedelta(days=self.params['lookback_days'])
        end_date = data.index.max()
        
        self.logger.info(f"News date range: {start_date.date()} to {end_date.date()}")
        
        # Collect news data
        news_data = self._collect_news_data(instruments, start_date, end_date)
        
        if news_data.empty:
            self.logger.warning("No news data found")
            return input_data
        
        self.logger.info(f"Collected {len(news_data)} news articles")
        
        # Perform sentiment analysis
        news_with_sentiment = self._analyze_sentiment(news_data)
        
        # Aggregate sentiment by instrument and date
        aggregated_sentiment = self._aggregate_sentiment_data(news_with_sentiment)
        
        # Generate sentiment features
        sentiment_features = self._generate_sentiment_features(aggregated_sentiment)
        
        # Merge features back to main data
        enhanced_data = self._merge_sentiment_features(data, sentiment_features)
        
        # Create updated stage data
        artifacts = input_data.artifacts.copy() if input_data.artifacts else {}
        artifacts['stage6_news_sentiment'] = {
            'news_articles_processed': len(news_data),
            'instruments_with_news': len(aggregated_sentiment['instrument'].unique()) if not aggregated_sentiment.empty else 0,
            'sentiment_models_used': list(self.sentiment_models.keys()),
            'date_range': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'feature_count': len(sentiment_features.columns) if not sentiment_features.empty else 0
        }
        
        return StageData(
            data=enhanced_data,
            metadata=StageMetadata(self.stage_name, self.version, {
                'news_articles': len(news_data),
                'sentiment_features': len(sentiment_features.columns) if not sentiment_features.empty else 0
            }),
            config=input_data.config,
            artifacts=artifacts
        )
    
    def _initialize_sentiment_models(self):
        """Initialize sentiment analysis models"""
        if not RAMANUJAN_SENTIMENT_AVAILABLE:
            self.logger.warning("Ramanujan sentiment models not available")
            return
        
        for model_name in self.params['sentiment_models']:
            try:
                # Create simple config dict instead of ModelConfig object
                config_dict = {
                    'model_type': model_name,
                    'parameters': {}
                }
                
                # Create a simple config object
                class SimpleConfig:
                    def __init__(self, config_dict):
                        self.model_type = config_dict['model_type']
                        self.parameters = config_dict['parameters']
                
                config = SimpleConfig(config_dict)
                
                if model_name == 'finbert':
                    model = FinBERTSentimentModel(config)
                elif model_name == 'vader':
                    model = VaderSentimentModel(config)
                elif model_name == 'textblob':
                    model = TextBlobSentimentModel(config)
                elif model_name == 'custom':
                    model = CustomFinancialSentimentModel(config)
                else:
                    self.logger.warning(f"Unknown sentiment model: {model_name}")
                    continue
                
                self.sentiment_models[model_name] = model
                self.logger.info(f"Initialized {model_name} sentiment model")
                
            except Exception as e:
                self.logger.warning(f"Failed to initialize {model_name} model: {e}")
        
        # Initialize news aggregation model
        if self.sentiment_models:
            aggregation_config = {
                'aggregation_methods': ['mean', 'volume_weighted', 'recency_weighted'],
                'time_decay_lambda': self.params['time_decay_lambda'],
                'min_articles': self.params['min_articles_per_day']
            }
            self.news_aggregator = NewsAggregationModel(aggregation_config)
    
    def _ensure_datetime_index(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure data has proper datetime index"""
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'date' in data.columns:
                data = data.set_index('date')
            else:
                # If no date column, assume index is already properly set
                pass
        
        return data.sort_index()
    
    def _get_instruments(self, data: pd.DataFrame) -> List[str]:
        """Get list of instruments from data"""
        if 'instrument' in data.columns:
            return data['instrument'].unique().tolist()
        else:
            return ['single']
    
    def _collect_news_data(self, instruments: List[str], start_date: datetime, 
                          end_date: datetime) -> pd.DataFrame:
        """Collect news data from Odin's Eye"""
        if not self.odins_eye:
            self.logger.warning("Odin's Eye not available")
            return pd.DataFrame()
        
        try:
            # Create date range
            date_range = DateRange(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            # Collect news for all instruments
            all_news = []
            
            for instrument in instruments:
                if instrument == 'single':
                    continue
                
                try:
                    news_articles = self.odins_eye.get_news_data(
                        symbols=instrument,
                        date_range=date_range,
                        sources=self.params['news_sources'] if self.params['news_sources'] != ['all'] else None
                    )
                    
                    # Convert to DataFrame
                    if news_articles:
                        news_df = pd.DataFrame(news_articles)
                        news_df['target_instrument'] = instrument
                        all_news.append(news_df)
                        
                except Exception as e:
                    self.logger.warning(f"Failed to get news for {instrument}: {e}")
                    continue
            
            if all_news:
                combined_news = pd.concat(all_news, ignore_index=True)
                
                # Clean and filter news data
                combined_news = self._filter_news_data(combined_news)
                
                return combined_news
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Failed to collect news data: {e}")
            return pd.DataFrame()
    
    def _filter_news_data(self, news_data: pd.DataFrame) -> pd.DataFrame:
        """Filter and clean news data"""
        if news_data.empty:
            return news_data
        
        original_count = len(news_data)
        
        # Filter by article length
        if 'content' in news_data.columns:
            content_length = news_data['content'].str.len().fillna(0)
            length_filter = (content_length >= self.params['min_article_length']) & \
                           (content_length <= self.params['max_article_length'])
            news_data = news_data[length_filter]
        
        # Remove duplicates based on URL or content
        if 'url' in news_data.columns:
            news_data = news_data.drop_duplicates(subset=['url'])
        elif 'title' in news_data.columns:
            news_data = news_data.drop_duplicates(subset=['title'])
        
        # Filter by language if specified
        if self.params['language_filter'] and 'language' in news_data.columns:
            news_data = news_data[news_data['language'] == self.params['language_filter']]
        
        # Ensure we have required columns
        required_cols = ['published_at', 'target_instrument']
        news_data = news_data.dropna(subset=required_cols)
        
        # Convert published_at to datetime
        if 'published_at' in news_data.columns:
            news_data['published_at'] = pd.to_datetime(news_data['published_at'])
            news_data['date'] = news_data['published_at'].dt.date
        
        filtered_count = len(news_data)
        self.logger.info(f"Filtered news: {original_count} -> {filtered_count} articles")
        
        return news_data
    
    def _analyze_sentiment(self, news_data: pd.DataFrame) -> pd.DataFrame:
        """Analyze sentiment of news articles"""
        if news_data.empty or not self.sentiment_models:
            return news_data
        
        # Prepare text for sentiment analysis
        if 'content' in news_data.columns:
            text_col = 'content'
        elif 'title' in news_data.columns:
            text_col = 'title'
        else:
            self.logger.warning("No text column found for sentiment analysis")
            return news_data
        
        # Create text DataFrame for models
        text_data = pd.DataFrame({
            'text': news_data[text_col].fillna('')
        })
        
        sentiment_scores = {}
        
        # Apply each sentiment model
        for model_name, model in self.sentiment_models.items():
            try:
                self.logger.info(f"Analyzing sentiment with {model_name}")
                scores = model.predict(text_data)
                sentiment_scores[f'sentiment_{model_name}'] = scores
                
            except Exception as e:
                self.logger.warning(f"Sentiment analysis failed for {model_name}: {e}")
                continue
        
        # Add sentiment scores to news data
        for col_name, scores in sentiment_scores.items():
            news_data[col_name] = scores
        
        # Create ensemble sentiment if multiple models
        if len(sentiment_scores) > 1 and self.params['ensemble_sentiment']:
            sentiment_columns = list(sentiment_scores.keys())
            news_data['sentiment_ensemble'] = news_data[sentiment_columns].mean(axis=1)
            news_data['sentiment_primary'] = news_data['sentiment_ensemble']
        elif sentiment_scores:
            # Use primary model or first available
            primary_col = f"sentiment_{self.params['primary_model']}"
            if primary_col in sentiment_scores:
                news_data['sentiment_primary'] = news_data[primary_col]
            else:
                # Use first available model
                first_col = list(sentiment_scores.keys())[0]
                news_data['sentiment_primary'] = news_data[first_col]
        
        return news_data
    
    def _aggregate_sentiment_data(self, news_data: pd.DataFrame) -> pd.DataFrame:
        """Aggregate sentiment data by instrument and date"""
        if news_data.empty or not self.news_aggregator:
            return pd.DataFrame()
        
        try:
            # Prepare data for aggregation
            agg_data = news_data.copy()
            agg_data['instrument'] = agg_data['target_instrument']
            agg_data['sentiment'] = agg_data['sentiment_primary']
            
            # Add source reliability weights if configured
            if self.params['source_reliability_weights'] and 'source' in agg_data.columns:
                weights = []
                for source in agg_data['source']:
                    weight = self.params['source_reliability_weights'].get(source, 1.0)
                    weights.append(weight)
                agg_data['importance'] = weights
            
            # Add timestamp for recency weighting
            if 'published_at' in agg_data.columns:
                agg_data['timestamp'] = agg_data['published_at']
            
            # Aggregate sentiment
            aggregated = self.news_aggregator.aggregate_sentiment(agg_data)
            
            # Calculate momentum features
            if not aggregated.empty:
                aggregated = self.news_aggregator.calculate_sentiment_momentum(
                    aggregated, windows=self.params['sentiment_windows']
                )
            
            return aggregated
            
        except Exception as e:
            self.logger.error(f"Failed to aggregate sentiment data: {e}")
            return pd.DataFrame()
    
    def _generate_sentiment_features(self, aggregated_data: pd.DataFrame) -> pd.DataFrame:
        """Generate sentiment features for merging"""
        if aggregated_data.empty:
            return pd.DataFrame()
        
        # Convert date column to datetime index
        feature_data = aggregated_data.copy()
        if 'date' in feature_data.columns:
            feature_data['date'] = pd.to_datetime(feature_data['date'])
            feature_data = feature_data.set_index('date')
        
        # Rename columns to match expected feature names
        prefix = self.params['feature_prefix']
        column_mapping = {
            'sentiment_mean': f'{prefix}_sentiment_mean',
            'sentiment_std': f'{prefix}_sentiment_std',
            'sentiment_median': f'{prefix}_sentiment_median',
            'article_count': f'{prefix}_article_count',
            'unique_sources': f'{prefix}_unique_sources',
            'positive_pct': f'{prefix}_positive_pct',
            'negative_pct': f'{prefix}_negative_pct',
            'neutral_pct': f'{prefix}_neutral_pct',
            'sentiment_skew': f'{prefix}_sentiment_skew',
            'news_shock': f'{prefix}_news_shock',
            'sentiment_recency_weighted': f'{prefix}_sentiment_recency_weighted',
            'sentiment_volatility_7d': f'{prefix}_sentiment_volatility_7d'
        }
        
        # Add momentum and moving average features
        for window in self.params['sentiment_windows']:
            column_mapping[f'sentiment_ma_{window}d'] = f'{prefix}_sentiment_ma_{window}d'
            column_mapping[f'sentiment_momentum_{window}d'] = f'{prefix}_sentiment_momentum_{window}d'
            column_mapping[f'article_count_ma_{window}d'] = f'{prefix}_article_count_ma_{window}d'
        
        # Rename columns
        feature_data = feature_data.rename(columns=column_mapping)
        
        # Keep only renamed columns and instrument
        feature_columns = ['instrument'] + list(column_mapping.values())
        available_columns = [col for col in feature_columns if col in feature_data.columns]
        
        return feature_data[available_columns]
    
    def _merge_sentiment_features(self, original_data: pd.DataFrame, 
                                 sentiment_features: pd.DataFrame) -> pd.DataFrame:
        """Merge sentiment features back to original data"""
        if sentiment_features.empty:
            return original_data
        
        # Reset indices for merging
        original_reset = original_data.reset_index()
        features_reset = sentiment_features.reset_index()
        
        # Determine date column names
        original_date_col = 'date' if 'date' in original_reset.columns else 'index'
        features_date_col = 'date' if 'date' in features_reset.columns else 'index'
        
        # Merge on date and instrument
        merged = original_reset.merge(
            features_reset,
            left_on=[original_date_col, 'instrument'],
            right_on=[features_date_col, 'instrument'],
            how='left',
            suffixes=('', '_news')
        )
        
        # Remove duplicate columns
        duplicate_cols = [col for col in merged.columns if col.endswith('_news')]
        merged = merged.drop(columns=duplicate_cols)
        
        # Restore index
        if original_date_col in merged.columns:
            merged = merged.set_index(original_date_col)
        
        return merged