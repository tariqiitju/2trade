#!/usr/bin/env python3
"""
Test Stage 7: Model Training

Tests the model training and selection pipeline that integrates
with the Ramanujan ML framework.
"""

import sys
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from pathlib import Path

# Import components
from da_vinchi.core.stage_7_model_training import Stage7ModelTraining
from da_vinchi.core.stage_base import StageData, StageMetadata


def create_mock_feature_data(instruments=['AAPL', 'MSFT', 'GOOGL'], n_days=100):
    """Create mock feature data for model training testing"""
    
    # Generate dates
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    
    data_list = []
    for instrument in instruments:
        for i, date in enumerate(dates):
            # Create realistic financial features
            base_price = 100 + np.random.randn() * 10  # Random walk around 100
            
            row = {
                'date': date,
                'instrument': instrument,
                
                # Basic OHLCV features (would come from Stage 1)
                'close': base_price + np.random.randn() * 2,
                'close_adj': base_price + np.random.randn() * 2,  # Required for target generation
                'volume': np.random.randint(1000000, 10000000),
                'returns_1d': np.random.randn() * 0.02,
                'returns_5d': np.random.randn() * 0.05,
                'volatility_20d': np.random.uniform(0.15, 0.35),
                
                # Technical indicators (Stage 1)
                'rsi_14': np.random.uniform(20, 80),
                'macd_signal': np.random.randn() * 0.5,
                'bollinger_position': np.random.uniform(-1, 1),
                'sma_20': base_price + np.random.randn() * 5,
                'ema_12': base_price + np.random.randn() * 3,
                
                # Volume features (Stage 2)
                'volume_ratio_20d': np.random.uniform(0.5, 2.0),
                'volume_trend_5d': np.random.randn() * 0.1,
                'vwap_distance': np.random.randn() * 0.02,
                
                # Price action features (Stage 3) 
                'support_strength': np.random.uniform(0, 1),
                'resistance_strength': np.random.uniform(0, 1),
                'trend_strength': np.random.uniform(-1, 1),
                
                # Market microstructure (Stage 4)
                'bid_ask_spread': np.random.uniform(0.001, 0.01),
                'order_flow_imbalance': np.random.randn() * 0.1,
                'price_impact': np.random.uniform(0, 0.1),
                
                # Market regime (Stage 5)
                'regime_bull_prob': np.random.uniform(0, 1),
                'regime_bear_prob': np.random.uniform(0, 1),
                'regime_volatility': np.random.uniform(0.1, 0.4),
                
                # News sentiment features (Stage 6)
                'news_sentiment_mean': np.random.uniform(-0.5, 0.5),
                'news_article_count': np.random.randint(0, 20),
                'news_sentiment_momentum_3d': np.random.randn() * 0.1,
                'news_shock': np.random.uniform(0, 2),
            }
            
            data_list.append(row)
    
    df = pd.DataFrame(data_list)
    df = df.set_index('date')
    
    return df


def test_stage7_initialization():
    """Test Stage 7 initialization"""
    print("=== Testing Stage 7 Initialization ===")
    
    try:
        # Test configuration
        config = {
            'stages': {
                'stage7_model_training': {
                    'model_types': ['xgboost', 'random_forest', 'ridge_regression'],
                    'target_types': ['returns', 'volatility'],
                    'target_horizons': [1, 5, 10],
                    'cv_splits': 5,
                    'test_size': 0.2,
                    'feature_selection': True,
                    'max_features': 20
                }
            }
        }
        
        stage7 = Stage7ModelTraining(config)
        
        print(f"Stage 7 initialized successfully")
        print(f"Model types: {stage7.params['model_types']}")
        print(f"Target types: {stage7.params['target_types']}")
        print(f"Target horizons: {stage7.params['target_horizons']}")
        print(f"Ramanujan available: {stage7.ramanujan_available}")
        
        return True
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_target_generation():
    """Test target variable generation"""
    print("=== Testing Target Generation ===")
    
    try:
        # Create mock data
        feature_data = create_mock_feature_data(['AAPL'], n_days=50)
        
        config = {
            'stages': {
                'stage7_model_training': {
                    'model_types': ['ridge_regression'],
                    'target_types': ['returns', 'volatility'],
                    'target_horizons': [1, 5],
                    'min_training_samples': 30
                }
            }
        }
        
        stage7 = Stage7ModelTraining(config)
        
        # Test target generation
        data_with_targets = stage7._generate_targets(feature_data)
        
        print(f"Original data shape: {feature_data.shape}")
        print(f"Data with targets shape: {data_with_targets.shape}")
        print(f"Generated targets: {[col for col in data_with_targets.columns if 'target_' in col]}")
        
        # Check target statistics
        for target in ['target_return_1d', 'target_return_5d']:
            if target in data_with_targets.columns:
                target_data = data_with_targets[target].dropna()
                print(f"{target}: mean={target_data.mean():.4f}, std={target_data.std():.4f}, count={len(target_data)}")
        
        return True
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_training():
    """Test model training pipeline"""
    print("=== Testing Model Training ===")
    
    try:
        # Create mock data with sufficient history
        feature_data = create_mock_feature_data(['AAPL'], n_days=120)
        
        config = {
            'stages': {
                'stage7_model_training': {
                    'model_types': ['ridge_regression'],  # Use simple model for testing
                    'target_types': ['returns'],
                    'target_horizons': [1],
                    'cv_splits': 3,
                    'test_size': 0.2,
                    'min_training_samples': 60,
                    'feature_selection': True,
                    'max_features': 10
                }
            }
        }
        
        stage7 = Stage7ModelTraining(config)
        
        # Create stage data
        stage_data = StageData(
            data=feature_data,
            metadata=StageMetadata("test", "1.0.0"),
            config=config,
            artifacts={}
        )
        
        # Process the data (train models)
        result = stage7.process(stage_data)
        
        print(f"Training completed successfully")
        print(f"Result data shape: {result.data.shape}")
        print(f"Artifacts created: {list(result.artifacts.keys())}")
        
        # Check predictions
        prediction_cols = [col for col in result.data.columns if 'pred_' in col]
        print(f"Prediction columns: {prediction_cols}")
        
        if prediction_cols:
            pred_col = prediction_cols[0]
            pred_data = result.data[pred_col].dropna()
            print(f"Predictions - mean: {pred_data.mean():.4f}, std: {pred_data.std():.4f}, count: {len(pred_data)}")
        
        return True
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_model_comparison():
    """Test multiple model comparison"""
    print("=== Testing Multi-Model Comparison ===")
    
    try:
        # Create mock data
        feature_data = create_mock_feature_data(['AAPL', 'MSFT'], n_days=100)
        
        config = {
            'stages': {
                'stage7_model_training': {
                    'model_types': ['ridge_regression', 'random_forest'],
                    'target_types': ['returns'],
                    'target_horizons': [1, 5],
                    'cv_splits': 3,
                    'test_size': 0.2,
                    'min_training_samples': 50,
                    'feature_selection': True,
                    'max_features': 15
                }
            }
        }
        
        stage7 = Stage7ModelTraining(config)
        
        # Create stage data  
        stage_data = StageData(
            data=feature_data,
            metadata=StageMetadata("test", "1.0.0"),
            config=config,
            artifacts={}
        )
        
        # Process with multiple models
        result = stage7.process(stage_data)
        
        print(f"Multi-model training completed")
        print(f"Result data shape: {result.data.shape}")
        print(f"Artifacts: {list(result.artifacts.keys())}")
        
        # Check for model comparison results
        if 'model_comparison' in result.artifacts:
            comparison = result.artifacts['model_comparison']
            print(f"Model comparison available with {len(comparison)} entries")
            
            # Display best models
            for entry in comparison[:3]:  # Top 3
                print(f"  {entry['model_id']}: {entry['metric']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_importance():
    """Test feature importance analysis"""
    print("=== Testing Feature Importance Analysis ===")
    
    try:
        # Create mock data with clear patterns
        feature_data = create_mock_feature_data(['AAPL'], n_days=80)
        
        # Add a highly predictive synthetic feature
        feature_data['synthetic_predictor'] = feature_data['returns_1d'] * 0.8 + np.random.randn(len(feature_data)) * 0.1
        
        config = {
            'stages': {
                'stage7_model_training': {
                    'model_types': ['random_forest'],  # Good for feature importance
                    'target_types': ['returns'],
                    'target_horizons': [1],
                    'cv_splits': 3,
                    'test_size': 0.2,
                    'min_training_samples': 40
                }
            }
        }
        
        stage7 = Stage7ModelTraining(config)
        
        # Create stage data
        stage_data = StageData(
            data=feature_data,
            metadata=StageMetadata("test", "1.0.0"),
            config=config,
            artifacts={}
        )
        
        # Process with feature importance analysis
        result = stage7.process(stage_data)
        
        print(f"Feature importance analysis completed")
        
        # Check feature importance results
        if 'feature_importance' in result.artifacts:
            importance = result.artifacts['feature_importance']
            print(f"Feature importance available for {len(importance)} features")
            
            # Display top features
            for feature, score in list(importance.items())[:5]:
                print(f"  {feature}: {score:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run Stage 7 model training tests"""
    print("Stage 7: Model Training - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Stage 7 Initialization", test_stage7_initialization),
        ("Target Generation", test_target_generation),
        ("Model Training", test_model_training),
        ("Multi-Model Comparison", test_multi_model_comparison),
        ("Feature Importance", test_feature_importance)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"  FAILED: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("STAGE 7 MODEL TRAINING TEST SUMMARY:")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "+ PASS" if result else "- FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("+ Stage 7 model training pipeline ready!")
        return True
    else:
        print("- Some tests failed - see details above")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)