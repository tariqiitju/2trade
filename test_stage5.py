#!/usr/bin/env python3
"""
Test Stage 5: Target Instrument Feature Generation
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Import components
from da_vinchi.core.stage_0_data_validator import Stage0DataValidator
from da_vinchi.core.stage_1_base_features import Stage1BaseFeatures
from da_vinchi.core.stage_2_cross_sectional import Stage2CrossSectional
from da_vinchi.core.stage_3_regimes_seasonal import Stage3RegimesSeasonal
from da_vinchi.core.stage_4_relationships import Stage4Relationships
from da_vinchi.core.stage_5_target_features import Stage5TargetFeatures
from da_vinchi.core.stage_base import StageData, StageMetadata


def create_multi_instrument_data(n_days=150, n_instruments=6):
    """Create realistic multi-instrument data with known relationships"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=n_days)
    
    # Create common market factor
    market_returns = np.random.normal(0.0005, 0.015, n_days)
    
    # Create sector factors
    tech_factor = np.random.normal(0.0003, 0.01, n_days)
    financial_factor = np.random.normal(0.0002, 0.012, n_days)
    
    instruments_data = []
    
    # Define instrument characteristics
    instrument_specs = [
        {'name': 'AAPL', 'market_beta': 1.2, 'sector': 'tech', 'idio_vol': 0.02},
        {'name': 'MSFT', 'market_beta': 1.0, 'sector': 'tech', 'idio_vol': 0.018},
        {'name': 'GOOGL', 'market_beta': 1.3, 'sector': 'tech', 'idio_vol': 0.025},
        {'name': 'JPM', 'market_beta': 1.1, 'sector': 'financial', 'idio_vol': 0.022},
        {'name': 'BAC', 'market_beta': 1.4, 'sector': 'financial', 'idio_vol': 0.028},
        {'name': 'SPY', 'market_beta': 1.0, 'sector': 'market', 'idio_vol': 0.012}  # ETF
    ]
    
    for spec in instrument_specs:
        # Generate correlated returns
        market_component = spec['market_beta'] * market_returns
        
        if spec['sector'] == 'tech':
            sector_component = 0.5 * tech_factor
        elif spec['sector'] == 'financial':
            sector_component = 0.5 * financial_factor
        else:
            sector_component = np.zeros(n_days)
        
        idio_component = np.random.normal(0, spec['idio_vol'], n_days)
        total_returns = market_component + sector_component + idio_component
        
        # Create OHLCV data
        prices = 100 * (1 + total_returns).cumprod()
        daily_range = np.abs(np.random.normal(0.015, 0.005, n_days))
        
        high = prices * (1 + daily_range/2)
        low = prices * (1 - daily_range/2)
        high = np.maximum(high, prices)
        low = np.minimum(low, prices)
        
        open_prices = np.roll(prices, 1)
        open_prices[0] = 100
        
        volume = np.random.lognormal(15, 0.3, n_days).astype(int)
        
        instrument_data = pd.DataFrame({
            'date': dates,
            'instrument': spec['name'],
            'open': open_prices,
            'high': high,
            'low': low,
            'close': prices,
            'close_adj': prices,
            'volume': volume,
            'log_return': total_returns,  # Add log_return column
            'sector': spec['sector'],
            'true_market_beta': spec['market_beta']
        })
        
        instruments_data.append(instrument_data)
    
    # Combine all instruments
    combined_data = pd.concat(instruments_data, ignore_index=True)
    
    # Create benchmark data
    benchmark_data = pd.DataFrame({
        'date': dates,
        'benchmark_return': market_returns,
        'benchmark_price': 100 * (1 + market_returns).cumprod()
    }).set_index('date')
    
    return combined_data, benchmark_data


def run_full_pipeline(raw_data, benchmark_data, target_instrument='AAPL'):
    """Run the full Da Vinchi pipeline through Stage 5"""
    
    print(f"Running full pipeline for target: {target_instrument}")
    
    # Base config
    base_config = {'stages': {}}
    
    # Stage 0: Data Validation
    print("  Stage 0: Data validation...")
    stage0 = Stage0DataValidator(base_config)
    stage_data = StageData(
        data=raw_data,
        metadata=StageMetadata("test", "1.0.0"),
        config=base_config,
        artifacts={}
    )
    stage0_result = stage0.execute(stage_data)
    print(f"    Output shape: {stage0_result.data.shape}")
    
    # Stage 1: Base Features
    print("  Stage 1: Base features...")
    stage1 = Stage1BaseFeatures(base_config)
    stage1_result = stage1.execute(stage0_result)
    print(f"    Output shape: {stage1_result.data.shape}")
    
    # Stage 2: Cross-sectional
    print("  Stage 2: Cross-sectional...")
    stage1_result.artifacts['benchmark_data'] = benchmark_data
    stage2_config = base_config.copy()
    stage2_config['stages']['stage2_cross_sectional'] = {
        'beta_windows': [60],
        'ranking_features': ['return_21d', 'vol_cc_20d']
    }
    stage2 = Stage2CrossSectional(stage2_config)
    stage2_result = stage2.execute(stage1_result)
    print(f"    Output shape: {stage2_result.data.shape}")
    
    # Stage 3: Regimes & Seasonal
    print("  Stage 3: Regimes & seasonal...")
    stage3_config = base_config.copy()
    stage3_config['stages']['stage3_regimes'] = {
        'regime_method': 'kmeans',
        'n_regimes': 3,
        'regime_features': ['vol_cc_20d', 'return_21d']
    }
    stage3 = Stage3RegimesSeasonal(stage3_config)
    stage3_result = stage3.execute(stage2_result)
    print(f"    Output shape: {stage3_result.data.shape}")
    
    # Stage 4: Relationships
    print("  Stage 4: Relationships...")
    stage4_config = base_config.copy()
    stage4_config['stages']['stage4_relationships'] = {
        'correlation_windows': [60],
        'min_correlation': 0.3,
        'min_instruments': 5
    }
    stage4 = Stage4Relationships(stage4_config)
    stage4_result = stage4.execute(stage3_result)
    print(f"    Output shape: {stage4_result.data.shape}")
    
    # Show relationship analysis results
    relationships = stage4_result.artifacts.get('stage4_relationships', {})
    if 'correlations' in relationships:
        corr_matrix = relationships['correlations'].get('correlation_matrix')
        if corr_matrix is not None:
            print(f"    Correlation matrix: {corr_matrix.shape}")
            # Show correlations for target instrument
            if target_instrument in corr_matrix.index:
                target_corrs = corr_matrix.loc[target_instrument].drop(target_instrument)
                print(f"    {target_instrument} correlations: {target_corrs.to_dict()}")
    
    # Stage 5: Target Features
    print(f"  Stage 5: Target features for {target_instrument}...")
    stage5_config = base_config.copy()
    stage5_config['stages']['stage5_target_features'] = {
        'target_instrument': target_instrument,
        'min_correlation': 0.2,
        'max_peers': 5,
        'momentum_windows': [5, 10, 20],
        'spread_windows': [10, 20],
        'create_momentum_features': True,
        'create_spread_features': True,
        'create_mean_reversion_features': True,
        'create_basket_features': True
    }
    stage5 = Stage5TargetFeatures(stage5_config)
    stage5_result = stage5.execute(stage4_result)
    print(f"    Output shape: {stage5_result.data.shape}")
    
    # Analyze Stage 5 results
    stage5_artifacts = stage5_result.artifacts.get('stage5_target_features', {})
    if stage5_artifacts:
        print(f"    Target instrument: {stage5_artifacts.get('target_instrument')}")
        print(f"    Peer instruments: {stage5_artifacts.get('peer_instruments')}")
        print(f"    Features generated: {stage5_artifacts.get('n_features')}")
        
        feature_categories = stage5_artifacts.get('feature_categories', {})
        for category, features in feature_categories.items():
            print(f"    {category.title()} features ({len(features)}): {features[:3]}...")
    
    return stage5_result


def test_stage5_individual():
    """Test Stage 5 individually with mock relationship data"""
    print("=== Individual Stage 5 Test ===")
    
    # Create test data
    raw_data, benchmark_data = create_multi_instrument_data(n_days=100, n_instruments=6)
    
    # Create mock Stage 4 relationships
    mock_relationships = {
        'correlations': {
            'correlation_matrix': pd.DataFrame({
                'AAPL': [1.0, 0.8, 0.7, 0.3, 0.2, 0.4],
                'MSFT': [0.8, 1.0, 0.9, 0.25, 0.15, 0.35],
                'GOOGL': [0.7, 0.9, 1.0, 0.2, 0.1, 0.3],
                'JPM': [0.3, 0.25, 0.2, 1.0, 0.6, 0.3],
                'BAC': [0.2, 0.15, 0.1, 0.6, 1.0, 0.25],
                'SPY': [0.4, 0.35, 0.3, 0.3, 0.25, 1.0]
            }, index=['AAPL', 'MSFT', 'GOOGL', 'JPM', 'BAC', 'SPY'])
        },
        'clustering': {
            'peer_maps': {
                'AAPL': {
                    'peers': [
                        {'instrument': 'MSFT', 'correlation': 0.8},
                        {'instrument': 'GOOGL', 'correlation': 0.7}
                    ]
                }
            }
        },
        'lead_lag': {
            'lead_lag_pairs': [
                {
                    'leading': 'SPY',
                    'lagging': 'AAPL',
                    'optimal_lag': 1,
                    'max_ccf': 0.5
                }
            ]
        }
    }
    
    # Configure Stage 5
    config = {
        'stages': {
            'stage5_target_features': {
                'target_instrument': 'AAPL',
                'min_correlation': 0.3,
                'max_peers': 4,
                'momentum_windows': [5, 10],
                'spread_windows': [10, 20],
                'create_momentum_features': True,
                'create_spread_features': True,
                'create_mean_reversion_features': True,
                'create_basket_features': True
            }
        }
    }
    
    # Create stage data with mock relationships
    stage_data = StageData(
        data=raw_data,
        metadata=StageMetadata("test", "1.0.0"),
        config=config,
        artifacts={'stage4_relationships': mock_relationships}
    )
    
    # Execute Stage 5
    stage5 = Stage5TargetFeatures(config)
    result = stage5.execute(stage_data)
    
    # Analyze results
    print(f"  Input shape: {raw_data.shape}")
    print(f"  Output shape: {result.data.shape}")
    
    # Count target-specific features
    target_features = [col for col in result.data.columns if col.startswith('tgt_')]
    print(f"  Target features created: {len(target_features)}")
    
    # Show sample features
    if target_features:
        print(f"  Sample features: {target_features[:5]}")
        
        # Show feature values for AAPL
        aapl_data = result.data[result.data['instrument'] == 'AAPL']
        for feature in target_features[:3]:
            non_null_count = aapl_data[feature].count()
            print(f"    {feature}: {non_null_count} non-null values")
    
    stage5_info = result.artifacts.get('stage5_target_features', {})
    print(f"  Peer instruments: {stage5_info.get('peer_instruments', [])}")
    
    return len(target_features) > 0


def main():
    """Run Stage 5 tests"""
    print("Stage 5: Target Features - Test Suite")
    print("=" * 60)
    
    # Test 1: Individual Stage 5 test
    test1_success = test_stage5_individual()
    
    print()
    
    # Test 2: Full pipeline test
    print("=== Full Pipeline Test ===")
    try:
        raw_data, benchmark_data = create_multi_instrument_data(n_days=120, n_instruments=6)
        final_result = run_full_pipeline(raw_data, benchmark_data, target_instrument='AAPL')
        
        # Count final features
        original_cols = len(raw_data.columns)
        final_cols = len(final_result.data.columns)
        features_created = final_cols - original_cols
        
        print(f"\n  Pipeline Summary:")
        print(f"    Original columns: {original_cols}")
        print(f"    Final columns: {final_cols}")
        print(f"    Features created: {features_created}")
        
        # Count Stage 5 specific features
        stage5_features = [col for col in final_result.data.columns if col.startswith('tgt_')]
        print(f"    Stage 5 features: {len(stage5_features)}")
        
        test2_success = len(stage5_features) > 0
        
    except Exception as e:
        print(f"  Full pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        test2_success = False
    
    # Summary
    print("\n" + "=" * 60)
    print("STAGE 5 TEST SUMMARY:")
    print("=" * 60)
    
    tests = [
        ("Individual Stage 5 Test", test1_success),
        ("Full Pipeline Test", test2_success)
    ]
    
    passed = sum(result for _, result in tests)
    total = len(tests)
    
    for test_name, result in tests:
        status = "+ PASS" if result else "- FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("+ Stage 5 implementation successful!")
        return True
    else:
        print("- Some Stage 5 tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)