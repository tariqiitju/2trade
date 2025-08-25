#!/usr/bin/env python3
"""
Da Vinchi Stage Independence Demo

This script demonstrates the new independent stage architecture where:
1. Stages don't connect to Odin's Eye directly  
2. Pipeline manager handles data routing between stages
3. Each stage is configurable and model-switchable
4. Benchmark data is provided via artifacts
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from da_vinchi.core.stage_0_data_validator import Stage0DataValidator
from da_vinchi.core.stage_1_base_features import Stage1BaseFeatures  
from da_vinchi.core.stage_2_cross_sectional import Stage2CrossSectional
from da_vinchi.core.stage_3_regimes_seasonal import Stage3RegimesSeasonal
from da_vinchi.core.stage_4_relationships import Stage4Relationships
from da_vinchi.core.stage_base import StageData, StageMetadata


def create_sample_multi_instrument_data(n_days=252, n_instruments=3):
    """Create sample multi-instrument OHLCV data"""
    np.random.seed(42)
    
    instruments = [f'INST_{i+1}' for i in range(n_instruments)]
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    
    all_data = []
    
    for i, instrument in enumerate(instruments):
        # Generate correlated price paths  
        returns = np.random.normal(0.0005 + i*0.0001, 0.02, n_days)
        prices = 100 * (1 + returns).cumprod()
        
        # Add some correlation between instruments
        if i > 0:
            correlation_factor = 0.3
            prev_returns = all_data[0]['close'].pct_change().fillna(0)
            correlated_component = prev_returns.iloc[-n_days:] * correlation_factor
            uncorrelated_component = returns * (1 - correlation_factor)
            combined_returns = correlated_component + uncorrelated_component
            prices = 100 * (1 + combined_returns).cumprod()
        
        # Create OHLCV
        daily_range = np.random.uniform(0.01, 0.03, n_days)
        high = prices * (1 + daily_range/2)
        low = prices * (1 - daily_range/2)
        open_prices = np.roll(prices, 1)
        open_prices[0] = 100
        
        volume = np.random.lognormal(15, 0.5, n_days).astype(int)
        
        instrument_data = pd.DataFrame({
            'date': dates,
            'instrument': instrument,
            'open': open_prices,
            'high': high, 
            'low': low,
            'close': prices,
            'close_adj': prices,
            'volume': volume
        })
        
        all_data.append(instrument_data)
    
    return pd.concat(all_data, ignore_index=True)


def create_benchmark_data(dates, base_return=0.0003, volatility=0.015):
    """Create synthetic benchmark data"""
    np.random.seed(123)  # Different seed for benchmark
    
    benchmark_returns = np.random.normal(base_return, volatility, len(dates))
    benchmark_prices = 100 * (1 + benchmark_returns).cumprod()
    
    return pd.DataFrame({
        'date': dates,
        'benchmark_return': benchmark_returns,
        'benchmark_price': benchmark_prices
    }).set_index('date')


def demonstrate_stage_independence():
    """Demonstrate independent stage processing"""
    print("=" * 80)
    print("DA VINCHI STAGE INDEPENDENCE DEMONSTRATION")
    print("=" * 80)
    
    # 1. Create sample data (no Odin's Eye dependency)
    print("\n1. Creating sample multi-instrument data...")
    raw_data = create_sample_multi_instrument_data(n_days=300, n_instruments=4)
    print(f"   Created data: {raw_data.shape} rows, {raw_data['instrument'].nunique()} instruments")
    
    # 2. Create benchmark data separately
    print("\n2. Creating benchmark data...")
    dates = pd.to_datetime(raw_data['date'].unique())
    benchmark_data = create_benchmark_data(dates)
    print(f"   Created benchmark: {benchmark_data.shape} rows")
    
    # 3. Configure stages independently
    print("\n3. Configuring independent stages...")
    
    # Basic config template
    base_config = {
        'pipeline': {'output_root': 'da_vinchi/workspace'},
        'stages': {}
    }
    
    # Stage 0 config
    stage0_config = base_config.copy()
    stage0 = Stage0DataValidator(stage0_config)
    print("   ✓ Stage 0 (Data Validator) configured")
    
    # Stage 1 config  
    stage1_config = base_config.copy()
    stage1_config['stages']['stage1_ohlcv_features'] = {
        'volatility_windows': [20, 60],
        'return_windows': [21, 63],
        'ema_spans': [12, 26]
    }
    stage1 = Stage1BaseFeatures(stage1_config)
    print("   ✓ Stage 1 (Base Features) configured")
    
    # Stage 2 config with cross-sectional parameters
    stage2_config = base_config.copy()
    stage2_config['stages']['stage2_cross_sectional'] = {
        'benchmark_symbol': 'MARKET',
        'beta_windows': [60],
        'ranking_features': ['return_21d', 'vol_cc_20d'],
        'min_observations': 30
    }
    stage2 = Stage2CrossSectional(stage2_config)
    print("   ✓ Stage 2 (Cross-sectional) configured")
    
    # Stage 3 config with regime detection
    stage3_config = base_config.copy()
    stage3_config['stages']['stage3_regimes'] = {
        'regime_method': 'kmeans',
        'n_regimes': 3,
        'regime_features': ['vol_cc_20d', 'return_21d'],
        'cyclical_features': ['day_of_week', 'month_of_year']
    }
    stage3 = Stage3RegimesSeasonal(stage3_config)
    print("   ✓ Stage 3 (Regimes & Seasonal) configured")
    
    # Stage 4 config with relationship analysis  
    stage4_config = base_config.copy()
    stage4_config['stages']['stage4_relationships'] = {
        'correlation_windows': [60],
        'correlation_method': 'spearman',
        'max_lags': 3,
        'min_correlation': 0.3
    }
    stage4 = Stage4Relationships(stage4_config)
    print("   ✓ Stage 4 (Relationships) configured")
    
    # 4. Process data through stages sequentially
    print("\n4. Processing data through independent stages...")
    
    # Stage 0: Data validation
    print("   Processing Stage 0: Data Validation...")
    stage_data = StageData(
        data=raw_data,
        metadata=StageMetadata("input", "1.0.0"),
        config=stage0_config,
        artifacts={}
    )
    
    stage0_result = stage0.execute(stage_data)
    print(f"      Input: {stage_data.data.shape} → Output: {stage0_result.data.shape}")
    
    # Stage 1: Base features
    print("   Processing Stage 1: Base Features...")
    stage1_result = stage1.execute(stage0_result)
    new_features = len(stage1_result.data.columns) - len(stage0_result.data.columns)
    print(f"      Added {new_features} base features")
    
    # Stage 2: Cross-sectional (with benchmark data)
    print("   Processing Stage 2: Cross-sectional Features...")
    stage1_result.artifacts['benchmark_data'] = benchmark_data
    stage2_result = stage2.execute(stage1_result)
    new_features = len(stage2_result.data.columns) - len(stage1_result.data.columns)
    print(f"      Added {new_features} cross-sectional features")
    
    # Stage 3: Regimes & seasonal
    print("   Processing Stage 3: Regimes & Seasonal...")
    stage3_result = stage3.execute(stage2_result)
    new_features = len(stage3_result.data.columns) - len(stage2_result.data.columns)
    print(f"      Added {new_features} regime/seasonal features")
    
    # Stage 4: Relationships
    print("   Processing Stage 4: Instrument Relationships...")
    stage4_result = stage4.execute(stage3_result)
    relationships = stage4_result.artifacts.get('stage4_relationships', {})
    print(f"      Analyzed relationships: {len(relationships)} relationship types")
    
    # 5. Show final results
    print("\n5. Final Results:")
    final_data = stage4_result.data
    total_features = len(final_data.columns)
    original_features = len(raw_data.columns)
    engineered_features = total_features - original_features
    
    print(f"   Original columns: {original_features}")
    print(f"   Total columns: {total_features}")
    print(f"   Engineered features: {engineered_features}")
    print(f"   Data shape: {final_data.shape}")
    
    # Show sample of new features by stage
    print("\n6. Sample Features by Stage:")
    
    stage1_features = [col for col in stage1_result.data.columns 
                      if col not in stage0_result.data.columns]
    print(f"   Stage 1 features (sample): {stage1_features[:5]}")
    
    stage2_features = [col for col in stage2_result.data.columns 
                      if col not in stage1_result.data.columns]
    print(f"   Stage 2 features (sample): {stage2_features[:5]}")
    
    stage3_features = [col for col in stage3_result.data.columns 
                      if col not in stage2_result.data.columns]
    print(f"   Stage 3 features (sample): {stage3_features[:5]}")
    
    # Show relationship artifacts
    if relationships:
        print(f"\n7. Relationship Analysis Results:")
        if 'correlations' in relationships:
            corr_info = relationships['correlations']
            n_instruments = len(corr_info.get('instruments', []))
            print(f"   Correlation matrix: {n_instruments}x{n_instruments}")
        
        if 'clustering' in relationships:
            cluster_info = relationships['clustering'] 
            n_clusters = cluster_info.get('n_clusters_adaptive', 0)
            print(f"   Identified {n_clusters} instrument clusters")
        
        if 'lead_lag' in relationships:
            leadlag_info = relationships['lead_lag']
            n_pairs = len(leadlag_info.get('lead_lag_pairs', []))
            print(f"   Found {n_pairs} lead-lag relationships")
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("✓ All stages processed data independently")
    print("✓ No direct Odin's Eye dependencies in stages")
    print("✓ Benchmark data provided via pipeline manager")
    print("✓ Each stage configurable with different models/parameters")
    print("=" * 80)
    
    return final_data, stage4_result.artifacts


def demonstrate_model_switching():
    """Demonstrate switching between different models within stages"""
    print("\n" + "=" * 80)
    print("MODEL SWITCHING DEMONSTRATION")
    print("=" * 80)
    
    # Create simple data
    raw_data = create_sample_multi_instrument_data(n_days=200, n_instruments=3)
    
    # Process through stages 0-1 first
    base_config = {'pipeline': {'output_root': 'da_vinchi/workspace'}, 'stages': {}}
    
    stage0 = Stage0DataValidator(base_config)
    stage1 = Stage1BaseFeatures(base_config)
    
    stage_data = StageData(
        data=raw_data,
        metadata=StageMetadata("input", "1.0.0"),
        config=base_config,
        artifacts={}
    )
    
    stage0_result = stage0.execute(stage_data)
    stage1_result = stage1.execute(stage0_result)
    
    # Test different regime detection models
    print("\n1. Testing different regime detection models:")
    
    models_to_test = ['kmeans', 'gmm', 'dbscan']
    
    for model_name in models_to_test:
        print(f"\n   Testing {model_name.upper()} for regime detection...")
        
        stage3_config = base_config.copy()
        stage3_config['stages']['stage3_regimes'] = {
            'regime_method': model_name,
            'n_regimes': 3,
            'regime_features': ['vol_cc_20d', 'return_21d']
        }
        
        try:
            stage3 = Stage3RegimesSeasonal(stage3_config)
            stage3_result = stage3.execute(stage1_result)
            
            # Check if regime features were created
            regime_features = [col for col in stage3_result.data.columns 
                             if col.startswith('regime_')]
            
            model_info = stage3.model_selector
            model_available = model_info.get('available', False) if model_info else False
            actual_method = model_info.get('method', 'unknown') if model_info else 'unknown'
            
            print(f"      ✓ {model_name}: {len(regime_features)} regime features created")
            print(f"        Model available: {model_available}, Used method: {actual_method}")
            
        except Exception as e:
            print(f"      ✗ {model_name} failed: {str(e)}")
    
    # Test different correlation methods
    print("\n2. Testing different correlation methods:")
    
    correlation_methods = ['pearson', 'spearman', 'kendall']
    
    for method in correlation_methods:
        print(f"\n   Testing {method.upper()} correlation...")
        
        stage4_config = base_config.copy()
        stage4_config['stages']['stage4_relationships'] = {
            'correlation_method': method,
            'correlation_windows': [60],
            'min_correlation': 0.2
        }
        
        try:
            stage4 = Stage4Relationships(stage4_config)
            stage4_result = stage4.execute(stage1_result)
            
            relationships = stage4_result.artifacts.get('stage4_relationships', {})
            correlations = relationships.get('correlations', {})
            
            if 'correlation_matrix' in correlations:
                corr_matrix = correlations['correlation_matrix']
                print(f"      ✓ {method}: {corr_matrix.shape} correlation matrix created")
            else:
                print(f"      ⚠ {method}: No correlation matrix (insufficient instruments?)")
                
        except Exception as e:
            print(f"      ✗ {method} failed: {str(e)}")
    
    print("\n" + "=" * 80)
    print("MODEL SWITCHING DEMONSTRATION COMPLETE")
    print("✓ Stages can dynamically switch between different algorithms")
    print("✓ Graceful fallback when models are not available")
    print("✓ Independent model selection per stage")
    print("=" * 80)


if __name__ == "__main__":
    # Run main demonstration
    final_data, artifacts = demonstrate_stage_independence()
    
    # Run model switching demonstration
    demonstrate_model_switching()
    
    print(f"\nFinal engineered dataset shape: {final_data.shape}")
    print("Demo completed successfully!")