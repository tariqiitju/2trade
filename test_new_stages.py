#!/usr/bin/env python3
"""
Quick functional test for newly implemented Da Vinchi stages 2, 3, and 4.
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Import the new stages
from da_vinchi.core.stage_0_data_validator import Stage0DataValidator
from da_vinchi.core.stage_1_base_features import Stage1BaseFeatures
from da_vinchi.core.stage_2_cross_sectional import Stage2CrossSectional
from da_vinchi.core.stage_3_regimes_seasonal import Stage3RegimesSeasonal
from da_vinchi.core.stage_4_relationships import Stage4Relationships
from da_vinchi.core.stage_base import StageData, StageMetadata


def create_test_data(n_days=100, n_instruments=3):
    """Create synthetic test data"""
    np.random.seed(42)
    
    instruments = [f'TEST_{i+1}' for i in range(n_instruments)]
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    
    all_data = []
    
    for i, instrument in enumerate(instruments):
        # Generate correlated price paths  
        returns = np.random.normal(0.0005, 0.02, n_days)
        prices = 100 * (1 + returns).cumprod()
        
        # Create OHLCV
        daily_range = np.random.uniform(0.01, 0.03, n_days)
        high = prices * (1 + daily_range/2)
        low = prices * (1 - daily_range/2)
        open_prices = np.roll(prices, 1)
        open_prices[0] = 100
        volume = np.random.lognormal(15, 0.5, n_days).astype(int)
        
        data = pd.DataFrame({
            'date': dates,
            'instrument': instrument,
            'open': open_prices,
            'high': high, 
            'low': low,
            'close': prices,
            'close_adj': prices,
            'volume': volume
        })
        
        all_data.append(data)
    
    return pd.concat(all_data, ignore_index=True)


def create_benchmark_data(dates):
    """Create benchmark returns"""
    np.random.seed(123)
    benchmark_returns = np.random.normal(0.0003, 0.015, len(dates))
    
    return pd.DataFrame({
        'benchmark_return': benchmark_returns
    }, index=dates)


def test_stage_2():
    """Test Stage 2: Cross-sectional Features"""
    print("\n=== Testing Stage 2: Cross-sectional Features ===")
    
    try:
        # Create config
        config = {
            'stages': {
                'stage2_cross_sectional': {
                    'benchmark_symbol': 'MARKET',
                    'beta_windows': [60],
                    'ranking_features': ['return_21d', 'vol_cc_20d'],
                    'min_observations': 20
                }
            }
        }
        
        stage2 = Stage2CrossSectional(config)
        print("+ Stage 2 initialized successfully")
        
        # Test with dummy data that has ALL required features
        n_points = 150  # More data points
        test_data = pd.DataFrame({
            'close_adj': np.random.randn(n_points) * 10 + 100,
            'volume': np.random.randint(1000, 10000, n_points),
            'log_return': np.random.randn(n_points) * 0.02,
            'return_21d': np.random.randn(n_points) * 0.1,
            'return_63d': np.random.randn(n_points) * 0.15,
            'return_126d': np.random.randn(n_points) * 0.2,
            'vol_cc_20d': np.abs(np.random.randn(n_points) * 0.3),
            'vol_cc_60d': np.abs(np.random.randn(n_points) * 0.25),
            'instrument': ['TEST_1'] * (n_points//2) + ['TEST_2'] * (n_points//2)
        }, index=pd.date_range('2023-01-01', periods=n_points))
        
        # Create benchmark data
        benchmark_data = create_benchmark_data(test_data.index.unique())
        
        # Create stage data
        stage_data = StageData(
            data=test_data,
            metadata=StageMetadata("test", "1.0.0"),
            config=config,
            artifacts={'benchmark_data': benchmark_data}
        )
        
        # Execute stage
        result = stage2.execute(stage_data)
        
        print(f"+ Stage 2 execution completed")
        print(f"  Input columns: {len(test_data.columns)}")
        print(f"  Output columns: {len(result.data.columns)}")
        print(f"  New features: {len(result.data.columns) - len(test_data.columns)}")
        
        # Check for key features
        expected_features = ['beta_60d', 'alpha_60d', 'return_21d_rank', 'in_universe']
        found_features = [f for f in expected_features if f in result.data.columns]
        print(f"  Expected features found: {len(found_features)}/{len(expected_features)}")
        
        return True
        
    except Exception as e:
        print(f"- Stage 2 failed: {e}")
        return False


def test_stage_3():
    """Test Stage 3: Regimes & Seasonal"""
    print("\n=== Testing Stage 3: Regimes & Seasonal ===")
    
    try:
        # Create config
        config = {
            'stages': {
                'stage3_regimes': {
                    'regime_method': 'kmeans',
                    'n_regimes': 3,
                    'regime_features': ['vol_cc_20d', 'return_21d'],
                    'cyclical_features': ['day_of_week', 'month_of_year']
                }
            }
        }
        
        stage3 = Stage3RegimesSeasonal(config)
        print("+ Stage 3 initialized successfully")
        print(f"  Model selector available: {stage3.model_selector['available']}")
        print(f"  Selected method: {stage3.model_selector['method']}")
        
        # Test with dummy data
        test_data = pd.DataFrame({
            'close_adj': np.random.randn(80) * 10 + 100,
            'log_return': np.random.randn(80) * 0.02,
            'return_21d': np.random.randn(80) * 0.1,
            'vol_cc_20d': np.abs(np.random.randn(80) * 0.3),
            'instrument': ['TEST_1'] * 80
        }, index=pd.date_range('2023-01-01', periods=80))
        
        stage_data = StageData(
            data=test_data,
            metadata=StageMetadata("test", "1.0.0"),
            config=config,
            artifacts={}
        )
        
        # Execute stage
        result = stage3.execute(stage_data)
        
        print(f"+ Stage 3 execution completed")
        print(f"  Input columns: {len(test_data.columns)}")
        print(f"  Output columns: {len(result.data.columns)}")
        print(f"  New features: {len(result.data.columns) - len(test_data.columns)}")
        
        # Check for key features
        regime_features = [col for col in result.data.columns if 'regime' in col]
        seasonal_features = [col for col in result.data.columns if any(x in col for x in ['dow', 'month', 'is_'])]
        
        print(f"  Regime features: {len(regime_features)}")
        print(f"  Seasonal features: {len(seasonal_features)}")
        
        return True
        
    except Exception as e:
        print(f"- Stage 3 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_stage_4():
    """Test Stage 4: Relationships"""
    print("\n=== Testing Stage 4: Relationships ===")
    
    try:
        # Create config
        config = {
            'stages': {
                'stage4_relationships': {
                    'correlation_method': 'pearson',
                    'correlation_windows': [60],
                    'max_lags': 2,
                    'min_correlation': 0.3
                }
            }
        }
        
        stage4 = Stage4Relationships(config)
        print("+ Stage 4 initialized successfully")
        
        # Test with multi-instrument data
        test_data = pd.DataFrame({
            'close_adj': np.random.randn(120) * 10 + 100,
            'log_return': np.random.randn(120) * 0.02,
            'instrument': ['TEST_1'] * 40 + ['TEST_2'] * 40 + ['TEST_3'] * 40
        }, index=pd.date_range('2023-01-01', periods=40).repeat(3))
        
        stage_data = StageData(
            data=test_data,
            metadata=StageMetadata("test", "1.0.0"),
            config=config,
            artifacts={}
        )
        
        # Execute stage
        result = stage4.execute(stage_data)
        
        print(f"+ Stage 4 execution completed")
        
        # Check artifacts
        relationships = result.artifacts.get('stage4_relationships', {})
        
        if 'correlations' in relationships:
            print(f"  + Correlation analysis completed")
            
        if 'clustering' in relationships:
            n_clusters = relationships['clustering'].get('n_clusters_adaptive', 0)
            print(f"  + Clustering completed: {n_clusters} clusters")
            
        if 'lead_lag' in relationships:
            n_pairs = len(relationships['lead_lag'].get('lead_lag_pairs', []))
            print(f"  + Lead-lag analysis: {n_pairs} relationships found")
            
        if 'cointegration' in relationships:
            n_cointegrated = len(relationships['cointegration'].get('cointegrated_pairs', []))
            print(f"  + Cointegration analysis: {n_cointegrated} pairs found")
        
        return True
        
    except Exception as e:
        print(f"- Stage 4 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline_integration():
    """Test stages working together"""
    print("\n=== Testing Pipeline Integration ===")
    
    try:
        # Create test data
        raw_data = create_test_data(n_days=100, n_instruments=3)
        print(f"Created test data: {raw_data.shape}")
        
        # Process through stages sequentially
        base_config = {'stages': {}}
        
        # Stage 0
        stage0 = Stage0DataValidator(base_config)
        stage_data = StageData(
            data=raw_data,
            metadata=StageMetadata("test", "1.0.0"),
            config=base_config,
            artifacts={}
        )
        stage0_result = stage0.execute(stage_data)
        print(f"Stage 0: {stage0_result.data.shape}")
        
        # Stage 1
        stage1 = Stage1BaseFeatures(base_config)
        stage1_result = stage1.execute(stage0_result)
        print(f"Stage 1: {stage1_result.data.shape}")
        
        # Add benchmark data for Stage 2
        dates = pd.to_datetime(stage1_result.data.index if isinstance(stage1_result.data.index, pd.DatetimeIndex) else stage1_result.data['date'].unique())
        benchmark_data = create_benchmark_data(dates)
        stage1_result.artifacts['benchmark_data'] = benchmark_data
        
        # Stage 2
        stage2_config = base_config.copy()
        stage2_config['stages']['stage2_cross_sectional'] = {
            'beta_windows': [60],
            'ranking_features': ['return_21d', 'vol_cc_20d']
        }
        stage2 = Stage2CrossSectional(stage2_config)
        stage2_result = stage2.execute(stage1_result)
        print(f"Stage 2: {stage2_result.data.shape}")
        
        # Stage 3
        stage3_config = base_config.copy()
        stage3_config['stages']['stage3_regimes'] = {
            'regime_method': 'kmeans',
            'n_regimes': 3,
            'regime_features': ['vol_cc_20d', 'return_21d']
        }
        stage3 = Stage3RegimesSeasonal(stage3_config)
        stage3_result = stage3.execute(stage2_result)
        print(f"Stage 3: {stage3_result.data.shape}")
        
        # Stage 4
        stage4_config = base_config.copy()
        stage4_config['stages']['stage4_relationships'] = {
            'correlation_windows': [60],
            'min_correlation': 0.2
        }
        stage4 = Stage4Relationships(stage4_config)
        stage4_result = stage4.execute(stage3_result)
        print(f"Stage 4: {stage4_result.data.shape}")
        
        final_features = len(stage4_result.data.columns)
        original_features = len(raw_data.columns)
        engineered_features = final_features - original_features
        
        print(f"+ Pipeline integration successful!")
        print(f"  Original features: {original_features}")
        print(f"  Final features: {final_features}")
        print(f"  Engineered features: {engineered_features}")
        
        return True
        
    except Exception as e:
        print(f"- Pipeline integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("Starting Da Vinchi New Stages Functional Tests")
    print("=" * 60)
    
    tests = [
        ("Stage 2 (Cross-sectional)", test_stage_2),
        ("Stage 3 (Regimes & Seasonal)", test_stage_3),
        ("Stage 4 (Relationships)", test_stage_4),
        ("Pipeline Integration", test_pipeline_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        results[test_name] = test_func()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY:")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "+ PASS" if result else "- FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("+ All functional tests passed!")
        return True
    else:
        print("-- Some tests failed - see details above")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)