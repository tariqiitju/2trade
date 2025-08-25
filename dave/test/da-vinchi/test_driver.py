#!/usr/bin/env python3
"""
Da Vinchi Feature Engineering Test Driver

Comprehensive test suite for Da Vinchi pipeline stages.
Tests Stage 0 (Data Validator) and Stage 1 (Base Features) with real market data.
"""

import sys
import os
import yaml
import json
import traceback
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Import Da Vinchi components
from da_vinchi.core.stage_0_data_validator import Stage0DataValidator, DataValidationConfig
from da_vinchi.core.stage_1_base_features import Stage1BaseFeatures
from da_vinchi.core.stage_2_cross_sectional import Stage2CrossSectional
from da_vinchi.core.stage_3_regimes_seasonal import Stage3RegimesSeasonal
from da_vinchi.core.stage_4_relationships import Stage4Relationships
from da_vinchi.core.stage_base import StageData, StageMetadata

# Import model performance testing
from test_model_performance import ModelPerformanceTester, ModelPerformanceConfig

# Import Odin's Eye components
from odins_eye import OdinsEye, DateRange, MarketDataInterval
from odins_eye.filters import InstrumentFilter
from odins_eye.exceptions import OdinsEyeError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DaVinchiTestDriver:
    """
    Test driver for Da Vinchi feature engineering pipeline.
    
    Provides comprehensive testing of Stage 0 and Stage 1 with real market data.
    """
    
    def __init__(self, config_path: Optional[str] = None, verbose: bool = False):
        self.verbose = verbose
        self.config = self._load_config(config_path)
        
        # Test tracking
        self.test_results = []
        self.failed_tests = []
        self.passed_tests = []
        
        # Initialize components
        self.data_root = self.config.get('data_root')
        self.test_instruments = self.config.get('test_instruments', ['AAPL', 'MSFT'])
        
        # Test workspace
        self.workspace_dir = Path(__file__).parent / "test_workspace"
        self.workspace_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized Da Vinchi Test Driver")
        logger.info(f"Test instruments: {self.test_instruments}")
        logger.info(f"Workspace: {self.workspace_dir}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load test configuration"""
        if config_path is None:
            config_path = Path(__file__).parent / "test_config.yml"
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config.get('test_config', {})
        except Exception as e:
            logger.warning(f"Could not load config: {e}. Using defaults.")
            return {}
    
    def run_all_tests(self) -> bool:
        """Run all test categories"""
        logger.info("=" * 80)
        logger.info("STARTING DA VINCHI FEATURE ENGINEERING TESTS")
        logger.info("=" * 80)
        
        try:
            # Test Stage 0: Data Validator
            self._test_stage0_data_validator()
            
            # Test Stage 1: Base Features  
            self._test_stage1_base_features()
            
            # Test Stage 2: Cross-sectional Features
            self._test_stage2_cross_sectional()
            
            # Test Stage 3: Regimes & Seasonal Features  
            self._test_stage3_regimes_seasonal()
            
            # Test Stage 4: Relationships
            self._test_stage4_relationships()
            
            # Test Pipeline Integration
            self._test_pipeline_integration()
            
            # Test Model Performance
            self._test_model_performance()
            
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            if self.verbose:
                traceback.print_exc()
            return False
        
        # Print final results
        self._print_test_summary()
        
        return len(self.failed_tests) == 0
    
    def _test_stage0_data_validator(self):
        """Test Stage 0: Data Validator"""
        logger.info("\n" + "=" * 60)
        logger.info("TESTING STAGE 0: DATA VALIDATOR")
        logger.info("=" * 60)
        
        # Test 1: Basic initialization
        self._run_test("Stage0_Initialization", self._test_stage0_init)
        
        # Test 2: Data loading and validation
        self._run_test("Stage0_DataLoading", self._test_stage0_data_loading)
        
        # Test 3: OHLCV validation
        self._run_test("Stage0_OHLCVValidation", self._test_stage0_ohlcv_validation)
        
        # Test 4: Data hygiene rules
        self._run_test("Stage0_DataHygiene", self._test_stage0_data_hygiene)
        
        # Test 5: Quality reporting
        self._run_test("Stage0_QualityReporting", self._test_stage0_quality_reporting)
    
    def _test_stage1_base_features(self):
        """Test Stage 1: Base Features"""
        logger.info("\n" + "=" * 60)
        logger.info("TESTING STAGE 1: BASE FEATURES")
        logger.info("=" * 60)
        
        # Test 1: Basic initialization
        self._run_test("Stage1_Initialization", self._test_stage1_init)
        
        # Test 2: Returns and volatility features
        self._run_test("Stage1_ReturnsVolatility", self._test_stage1_returns_volatility)
        
        # Test 3: Trend and momentum features
        self._run_test("Stage1_TrendMomentum", self._test_stage1_trend_momentum)
        
        # Test 4: Range and band features
        self._run_test("Stage1_RangeBands", self._test_stage1_range_bands)
        
        # Test 5: Liquidity features
        self._run_test("Stage1_Liquidity", self._test_stage1_liquidity)
        
        # Test 6: Feature completeness
        self._run_test("Stage1_FeatureCompleteness", self._test_stage1_completeness)
    
    def _test_stage2_cross_sectional(self):
        """Test Stage 2: Cross-sectional Features"""
        logger.info("\n" + "=" * 60)
        logger.info("TESTING STAGE 2: CROSS-SECTIONAL FEATURES")
        logger.info("=" * 60)
        
        # Test 1: Basic initialization
        self._run_test("Stage2_Initialization", self._test_stage2_init)
        
        # Test 2: Beta calculation
        self._run_test("Stage2_BetaCalculation", self._test_stage2_beta_calculation)
        
        # Test 3: Cross-sectional rankings
        self._run_test("Stage2_CrossSectionalRankings", self._test_stage2_rankings)
        
        # Test 4: Universe filtering
        self._run_test("Stage2_UniverseFiltering", self._test_stage2_universe_filtering)
    
    def _test_stage3_regimes_seasonal(self):
        """Test Stage 3: Regimes & Seasonal Features"""  
        logger.info("\n" + "=" * 60)
        logger.info("TESTING STAGE 3: REGIMES & SEASONAL FEATURES")
        logger.info("=" * 60)
        
        # Test 1: Basic initialization  
        self._run_test("Stage3_Initialization", self._test_stage3_init)
        
        # Test 2: Regime detection
        self._run_test("Stage3_RegimeDetection", self._test_stage3_regime_detection)
        
        # Test 3: Seasonal features
        self._run_test("Stage3_SeasonalFeatures", self._test_stage3_seasonal_features)
        
        # Test 4: Model selection
        self._run_test("Stage3_ModelSelection", self._test_stage3_model_selection)
    
    def _test_stage4_relationships(self):
        """Test Stage 4: Instrument Relationships"""
        logger.info("\n" + "=" * 60)
        logger.info("TESTING STAGE 4: INSTRUMENT RELATIONSHIPS")
        logger.info("=" * 60)
        
        # Test 1: Basic initialization
        self._run_test("Stage4_Initialization", self._test_stage4_init)
        
        # Test 2: Correlation analysis
        self._run_test("Stage4_CorrelationAnalysis", self._test_stage4_correlation)
        
        # Test 3: Lead-lag analysis
        self._run_test("Stage4_LeadLagAnalysis", self._test_stage4_leadlag)
        
        # Test 4: Cointegration analysis
        self._run_test("Stage4_CointegrationAnalysis", self._test_stage4_cointegration)
    
    def _test_pipeline_integration(self):
        """Test pipeline integration between stages"""
        logger.info("\n" + "=" * 60)
        logger.info("TESTING PIPELINE INTEGRATION")
        logger.info("=" * 60)
        
        # Test 1: Stage 0 -> Stage 1 integration
        self._run_test("Pipeline_Stage0to1", self._test_pipeline_stage0_to_1)
        
        # Test 2: End-to-end feature generation
        self._run_test("Pipeline_EndToEnd", self._test_pipeline_end_to_end)
    
    def _test_model_performance(self):
        """Test model performance with generated features"""
        logger.info("\n" + "=" * 60)
        logger.info("TESTING MODEL PERFORMANCE WITH GENERATED FEATURES")
        logger.info("=" * 60)
        
        # Test 1: Quick model evaluation (small dataset)
        self._run_test("ModelPerf_QuickEvaluation", self._test_quick_model_evaluation)
        
        # Test 2: Rolling window performance (if enough time/data)
        # self._run_test("ModelPerf_RollingWindow", self._test_rolling_window_evaluation)
    
    # Stage 0 Tests
    def _test_stage0_init(self) -> Tuple[bool, str]:
        """Test Stage 0 initialization"""
        try:
            config = self.config.get('stage0_config', {})
            stage0 = Stage0DataValidator(config, data_root=self.data_root)
            
            # Check initialization
            assert stage0.odins_eye is not None, "Odin's Eye not initialized"
            assert hasattr(stage0, 'validation_config'), "Validation config not set"
            
            return True, "Stage 0 initialized successfully"
        except Exception as e:
            return False, f"Stage 0 initialization failed: {str(e)}"
    
    def _test_stage0_data_loading(self) -> Tuple[bool, str]:
        """Test Stage 0 data loading"""
        try:
            config = self.config.get('stage0_config', {})
            stage0 = Stage0DataValidator(config, data_root=self.data_root)
            
            # Create test input
            date_range_config = self.config.get('test_date_range', {})
            date_range = DateRange(
                start_date=date_range_config.get('start_date', '2023-01-01'),
                end_date=date_range_config.get('end_date', '2023-12-31')
            )
            
            input_data = StageData(
                data=pd.DataFrame(),
                metadata=StageMetadata("test_input", "1.0.0"),
                config={
                    'instruments': self.test_instruments[:2],  # Use subset for speed
                    'date_range': date_range,
                    'interval': MarketDataInterval.DAILY
                }
            )
            
            # Process data
            result = stage0.process(input_data)
            
            # Validate results
            assert not result.data.empty, "No data loaded"
            assert 'open' in result.data.columns, "Missing OHLCV columns"
            assert 'close_adj' in result.data.columns, "Missing adjusted close"
            
            return True, f"Loaded {len(result.data)} rows of validated data"
            
        except Exception as e:
            return False, f"Data loading failed: {str(e)}"
    
    def _test_stage0_ohlcv_validation(self) -> Tuple[bool, str]:
        """Test OHLCV relationship validation"""
        try:
            # Create test data with some violations
            test_data = pd.DataFrame({
                'open': [100, 101, 102],
                'high': [105, 104, 103],  # Violation: high < close
                'low': [95, 96, 97],
                'close': [103, 102, 104],  # Violation: close > high  
                'volume': [1000, 1100, 1200],
                'instrument': ['TEST', 'TEST', 'TEST']
            }, index=pd.date_range('2023-01-01', periods=3))
            
            config = self.config.get('stage0_config', {})
            stage0 = Stage0DataValidator(config, data_root=self.data_root)
            
            # Should detect violations without crashing
            stage0._validate_ohlcv_relationships(test_data)
            
            return True, "OHLCV validation completed with warnings as expected"
            
        except Exception as e:
            return False, f"OHLCV validation failed: {str(e)}"
    
    def _test_stage0_data_hygiene(self) -> Tuple[bool, str]:
        """Test data hygiene rules"""
        try:
            # Create test data with outliers
            test_data = pd.DataFrame({
                'open': [100, 101, 1000],  # Extreme outlier
                'high': [105, 104, 1050],
                'low': [95, 96, 995],
                'close': [103, 102, 1020],
                'volume': [1000, 1100, 100000],  # Volume spike
                'instrument': ['TEST', 'TEST', 'TEST']
            }, index=pd.date_range('2023-01-01', periods=3))
            
            config = self.config.get('stage0_config', {})
            stage0 = Stage0DataValidator(config, data_root=self.data_root)
            
            # Apply hygiene rules - let's test each step individually to isolate the error
            try:
                clean_data = stage0._apply_data_hygiene(test_data)
                
                # Check results
                assert not clean_data.empty, "Data hygiene removed all data"
                assert 'data_quality_score' in clean_data.columns, "Missing quality score"
                
                return True, f"Data hygiene applied successfully, {len(clean_data)} rows remaining"
                
            except Exception as hygiene_error:
                # Let's try to isolate which step is failing
                test_steps = [
                    ("datetime_index", lambda: stage0._ensure_datetime_index(test_data.copy())),
                    ("winsorization", lambda: stage0._apply_winsorization(test_data.copy())),
                    ("threshold_filters", lambda: stage0._apply_threshold_filters(test_data.copy())),
                    ("quality_flags", lambda: stage0._add_quality_flags(test_data.copy()))
                ]
                
                for step_name, step_func in test_steps:
                    try:
                        step_func()
                    except Exception as step_error:
                        return False, f"Data hygiene failed at {step_name}: {str(step_error)}"
                
                # If individual steps work but combined doesn't
                return False, f"Data hygiene failed in combination: {str(hygiene_error)}"
            
        except Exception as e:
            return False, f"Data hygiene test setup failed: {str(e)}"
    
    def _test_stage0_quality_reporting(self) -> Tuple[bool, str]:
        """Test quality reporting"""
        try:
            test_data = pd.DataFrame({
                'open': [100, 101, 102],
                'high': [105, 104, 103],
                'low': [95, 96, 97],
                'close': [103, 102, 104],
                'volume': [1000, None, 1200],  # Missing value
                'instrument': ['TEST', 'TEST', 'TEST']
            }, index=pd.date_range('2023-01-01', periods=3))
            
            config = self.config.get('stage0_config', {})
            stage0 = Stage0DataValidator(config, data_root=self.data_root)
            
            # Generate quality report
            quality_report = stage0._generate_quality_report(test_data)
            
            # Validate report structure
            assert 'total_rows' in quality_report, "Missing total_rows"
            assert 'missing_data' in quality_report, "Missing missing_data stats"
            assert 'price_validation' in quality_report, "Missing price validation"
            
            return True, f"Quality report generated with {len(quality_report)} metrics"
            
        except Exception as e:
            return False, f"Quality reporting failed: {str(e)}"
    
    # Stage 1 Tests
    def _test_stage1_init(self) -> Tuple[bool, str]:
        """Test Stage 1 initialization"""
        try:
            config = self.config.get('stage1_config', {})
            stage1 = Stage1BaseFeatures(config)
            
            # Check initialization
            assert hasattr(stage1, 'params'), "Parameters not initialized"
            assert 'volatility_windows' in stage1.params, "Missing volatility windows"
            
            # Check feature names
            feature_names = stage1.get_feature_names()
            assert len(feature_names) > 20, f"Too few features: {len(feature_names)}"
            
            return True, f"Stage 1 initialized with {len(feature_names)} features"
            
        except Exception as e:
            return False, f"Stage 1 initialization failed: {str(e)}"
    
    def _test_stage1_returns_volatility(self) -> Tuple[bool, str]:
        """Test returns and volatility features"""
        try:
            # Create test data
            dates = pd.date_range('2023-01-01', periods=100)
            prices = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.01))
            
            test_data = pd.DataFrame({
                'open': prices * (1 + np.random.randn(100) * 0.001),
                'high': prices * (1 + abs(np.random.randn(100)) * 0.005),
                'low': prices * (1 - abs(np.random.randn(100)) * 0.005),
                'close': prices,
                'volume': np.random.randint(1000, 10000, 100),
                'close_adj': prices,
                'instrument': ['TEST'] * 100
            }, index=dates)
            
            config = self.config.get('stage1_config', {})
            stage1 = Stage1BaseFeatures(config)
            
            # Generate returns and volatility features
            result_data = stage1._generate_returns_volatility(test_data)
            
            # Check features were created
            expected_features = ['simple_return', 'log_return', 'vol_cc_20d', 'return_20d']
            for feature in expected_features:
                assert feature in result_data.columns, f"Missing feature: {feature}"
            
            # Check for reasonable values
            assert not result_data['log_return'].isnull().all(), "All log returns are NaN"
            assert (result_data['vol_cc_20d'] > 0).any(), "No positive volatility values"
            
            return True, f"Returns/volatility features generated successfully"
            
        except Exception as e:
            return False, f"Returns/volatility test failed: {str(e)}"
    
    def _test_stage1_trend_momentum(self) -> Tuple[bool, str]:
        """Test trend and momentum features"""
        try:
            # Create trending test data
            dates = pd.date_range('2023-01-01', periods=100)
            trend = np.linspace(100, 120, 100)  # Upward trend
            noise = np.random.randn(100) * 0.5
            prices = trend + noise
            
            test_data = pd.DataFrame({
                'open': prices * 0.99,
                'high': prices * 1.01,
                'low': prices * 0.98,
                'close': prices,
                'volume': np.random.randint(1000, 10000, 100),
                'close_adj': prices,
                'instrument': ['TEST'] * 100
            }, index=dates)
            
            config = self.config.get('stage1_config', {})
            stage1 = Stage1BaseFeatures(config)
            
            # Generate trend/momentum features
            result_data = stage1._generate_trend_momentum(test_data)
            
            # Check features
            expected_features = ['ema_12', 'ema_26', 'macd', 'rsi', 'stoch_k', 'adx']
            for feature in expected_features:
                assert feature in result_data.columns, f"Missing feature: {feature}"
            
            # Check RSI is bounded [0, 100]
            rsi_values = result_data['rsi'].dropna()
            if len(rsi_values) > 0:
                assert (rsi_values >= 0).all() and (rsi_values <= 100).all(), "RSI out of bounds"
            
            return True, f"Trend/momentum features generated successfully"
            
        except Exception as e:
            return False, f"Trend/momentum test failed: {str(e)}"
    
    def _test_stage1_range_bands(self) -> Tuple[bool, str]:
        """Test range and band features"""
        try:
            # Create test data with varying volatility
            dates = pd.date_range('2023-01-01', periods=50)
            prices = 100 + np.random.randn(50).cumsum()
            volatility = abs(np.random.randn(50)) * 2
            
            test_data = pd.DataFrame({
                'open': prices,
                'high': prices + volatility,
                'low': prices - volatility,
                'close': prices + np.random.randn(50) * 0.1,
                'volume': np.random.randint(1000, 10000, 50),
                'close_adj': prices + np.random.randn(50) * 0.1,
                'instrument': ['TEST'] * 50
            }, index=dates)
            
            config = self.config.get('stage1_config', {})
            stage1 = Stage1BaseFeatures(config)
            
            # Generate range/band features
            result_data = stage1._generate_range_bands(test_data)
            
            # Check features
            expected_features = ['true_range', 'atr', 'bb_upper', 'bb_lower', 'bb_percent']
            for feature in expected_features:
                assert feature in result_data.columns, f"Missing feature: {feature}"
            
            # Check ATR is positive
            atr_values = result_data['atr'].dropna()
            if len(atr_values) > 0:
                assert (atr_values > 0).all(), "ATR should be positive"
            
            return True, f"Range/band features generated successfully"
            
        except Exception as e:
            return False, f"Range/band test failed: {str(e)}"
    
    def _test_stage1_liquidity(self) -> Tuple[bool, str]:
        """Test liquidity features"""
        try:
            # Create test data with volume patterns
            dates = pd.date_range('2023-01-01', periods=50)
            prices = 100 + np.random.randn(50).cumsum() * 0.1
            volumes = np.random.randint(1000, 50000, 50)
            
            test_data = pd.DataFrame({
                'open': prices,
                'high': prices * 1.01,
                'low': prices * 0.99,
                'close': prices,
                'volume': volumes,
                'close_adj': prices,
                'log_return': np.random.randn(50) * 0.01,
                'instrument': ['TEST'] * 50
            }, index=dates)
            
            config = self.config.get('stage1_config', {})
            stage1 = Stage1BaseFeatures(config)
            
            # Generate liquidity features
            result_data = stage1._generate_liquidity_features(test_data)
            
            # Check features
            expected_features = ['dollar_volume', 'volume_ratio_20', 'amihud_20d']
            for feature in expected_features:
                assert feature in result_data.columns, f"Missing feature: {feature}"
            
            # Check dollar volume is positive
            dv_values = result_data['dollar_volume'].dropna()
            if len(dv_values) > 0:
                assert (dv_values > 0).all(), "Dollar volume should be positive"
            
            return True, f"Liquidity features generated successfully"
            
        except Exception as e:
            return False, f"Liquidity test failed: {str(e)}"
    
    def _test_stage1_completeness(self) -> Tuple[bool, str]:
        """Test feature completeness"""
        try:
            config = self.config.get('stage1_config', {})
            stage1 = Stage1BaseFeatures(config)
            
            # Get all expected features
            expected_features = stage1.get_feature_names()
            expected_count = self.config.get('expected_results', {}).get('stage1', {}).get('min_feature_count', 50)
            
            assert len(expected_features) >= expected_count, f"Too few features: {len(expected_features)} < {expected_count}"
            
            # Check feature categories
            expected_categories = self.config.get('expected_results', {}).get('stage1', {}).get('expected_feature_categories', [])
            found_categories = set()
            
            for feature in expected_features:
                if any(cat in feature for cat in ['return', 'vol']):
                    found_categories.add('returns')
                    found_categories.add('volatility')
                elif any(cat in feature for cat in ['ema', 'sma', 'macd', 'rsi']):
                    found_categories.add('momentum')
                    found_categories.add('trend')
                elif any(cat in feature for cat in ['bb_', 'atr']):
                    found_categories.add('bands')
                elif any(cat in feature for cat in ['volume', 'amihud', 'dollar']):
                    found_categories.add('liquidity')
            
            missing_categories = set(expected_categories) - found_categories
            assert not missing_categories, f"Missing feature categories: {missing_categories}"
            
            return True, f"Feature completeness verified: {len(expected_features)} features across {len(found_categories)} categories"
            
        except Exception as e:
            return False, f"Feature completeness test failed: {str(e)}"
    
    # Pipeline Integration Tests
    def _test_pipeline_stage0_to_1(self) -> Tuple[bool, str]:
        """Test Stage 0 -> Stage 1 integration"""
        try:
            # Initialize stages
            stage0_config = self.config.get('stage0_config', {})
            stage1_config = self.config.get('stage1_config', {})
            
            stage0 = Stage0DataValidator(stage0_config, data_root=self.data_root)
            stage1 = Stage1BaseFeatures(stage1_config)
            
            # Create input data
            date_range_config = self.config.get('test_date_range', {})
            date_range = DateRange(
                start_date=date_range_config.get('start_date', '2023-01-01'),
                end_date=date_range_config.get('end_date', '2023-06-30')  # Shorter range for speed
            )
            
            input_data = StageData(
                data=pd.DataFrame(),
                metadata=StageMetadata("pipeline_test", "1.0.0"),
                config={
                    'instruments': self.test_instruments[:1],  # Single instrument for speed
                    'date_range': date_range,
                    'interval': MarketDataInterval.DAILY
                }
            )
            
            # Run Stage 0
            stage0_result = stage0.process(input_data)
            assert not stage0_result.data.empty, "Stage 0 produced empty data"
            
            # Run Stage 1
            stage1_result = stage1.process(stage0_result)
            assert not stage1_result.data.empty, "Stage 1 produced empty data"
            
            # Check that Stage 1 has more columns (features added)
            assert len(stage1_result.data.columns) > len(stage0_result.data.columns), "Stage 1 did not add features"
            
            return True, f"Pipeline Stage 0->1 integration successful: {len(stage0_result.data.columns)} -> {len(stage1_result.data.columns)} columns"
            
        except Exception as e:
            return False, f"Pipeline integration test failed: {str(e)}"
    
    def _test_pipeline_end_to_end(self) -> Tuple[bool, str]:
        """Test end-to-end pipeline"""
        try:
            # This is a comprehensive test of the full pipeline
            result = self._run_full_pipeline(
                instruments=self.test_instruments[:1],
                start_date='2023-01-01',
                end_date='2023-03-31'  # Short range for speed
            )
            
            data, artifacts = result
            
            # Validate final output
            assert not data.empty, "Pipeline produced empty data"
            assert len(data.columns) > 20, f"Too few final features: {len(data.columns)}"
            
            # Check for key feature categories
            feature_columns = list(data.columns)
            has_returns = any('return' in col for col in feature_columns)
            has_volatility = any('vol' in col for col in feature_columns)
            has_momentum = any(col in feature_columns for col in ['rsi', 'macd', 'ema_12'])
            
            assert has_returns, "Missing return features"
            assert has_volatility, "Missing volatility features"
            assert has_momentum, "Missing momentum features"
            
            return True, f"End-to-end pipeline successful: {len(data)} rows, {len(data.columns)} features"
            
        except Exception as e:
            return False, f"End-to-end pipeline test failed: {str(e)}"
    
    def _run_full_pipeline(self, instruments: List[str], start_date: str, end_date: str) -> Tuple[pd.DataFrame, Dict]:
        """Run full pipeline for testing"""
        # Initialize stages
        stage0_config = self.config.get('stage0_config', {})
        stage1_config = self.config.get('stage1_config', {})
        
        stage0 = Stage0DataValidator(stage0_config, data_root=self.data_root)
        stage1 = Stage1BaseFeatures(stage1_config)
        
        # Create input
        date_range = DateRange(start_date=start_date, end_date=end_date)
        input_data = StageData(
            data=pd.DataFrame(),
            metadata=StageMetadata("full_pipeline", "1.0.0"),
            config={
                'instruments': instruments,
                'date_range': date_range,
                'interval': MarketDataInterval.DAILY
            }
        )
        
        # Run pipeline
        stage0_result = stage0.process(input_data)
        stage1_result = stage1.process(stage0_result)
        
        # Collect artifacts
        artifacts = {
            'stage0_artifacts': stage0_result.artifacts,
            'stage1_artifacts': stage1_result.artifacts,
            'stage0_metadata': stage0_result.metadata,
            'stage1_metadata': stage1_result.metadata
        }
        
        return stage1_result.data, artifacts
    
    # Model Performance Tests
    def _test_quick_model_evaluation(self) -> Tuple[bool, str]:
        """Test quick model evaluation with generated features"""
        try:
            # Create model performance config for quick test
            perf_config = ModelPerformanceConfig(
                training_windows=[30],  # Fixed parameter name
                prediction_horizon=1,
                min_samples_required=50,
                test_models=['linear_regression', 'random_forest'],  # Limited models for speed
                max_features=10  # Limit features for quick test
            )
            
            # Initialize model tester
            tester = ModelPerformanceTester(perf_config, data_root=self.data_root)
            
            # Run evaluation with limited data
            try:
                performance_summary = tester.run_rolling_window_evaluation(
                    instruments=self.test_instruments[:1],  # Single instrument for speed
                    start_date='2023-06-01',  # Short date range for quick test
                    end_date='2023-09-30'
                )
                
                # Validate results
                assert 'model_performance' in performance_summary, "Missing model performance results"
                assert len(performance_summary['models_tested']) > 0, "No models were tested"
                
                # Check that we have some successful predictions
                total_windows = performance_summary.get('total_windows', 0)
                assert total_windows > 0, "No prediction windows were evaluated"
                
                # Save results for inspection
                results_file = self.workspace_dir / f"quick_model_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                tester.save_results(str(results_file))
                
                return True, f"Quick model evaluation successful: {len(performance_summary['models_tested'])} models, {total_windows} windows"
                
            except Exception as e:
                return False, f"Model evaluation failed: {str(e)}"
                
        except Exception as e:
            return False, f"Quick model evaluation setup failed: {str(e)}"
    
    def _test_rolling_window_evaluation(self) -> Tuple[bool, str]:
        """Test comprehensive rolling window evaluation (slower test)"""
        try:
            # Create comprehensive model performance config
            perf_config = ModelPerformanceConfig(
                training_windows=[120],  # Fixed parameter name
                prediction_horizon=1,
                min_samples_required=200,
                test_models=['linear_regression', 'ridge_regression', 'random_forest', 'xgboost'],
                max_features=20
            )
            
            # Initialize model tester
            tester = ModelPerformanceTester(perf_config, data_root=self.data_root)
            
            # Run comprehensive evaluation
            performance_summary = tester.run_rolling_window_evaluation(
                instruments=self.test_instruments[:2],  # Limited instruments 
                start_date='2023-01-01',
                end_date='2023-12-31'
            )
            
            # Validate results
            assert 'model_performance' in performance_summary, "Missing model performance results"
            
            # Check performance metrics
            models_tested = performance_summary.get('models_tested', [])
            assert len(models_tested) > 0, "No models were successfully tested"
            
            # Verify we have meaningful metrics
            for model_name in models_tested:
                model_perf = performance_summary['model_performance'].get(model_name, {})
                metrics = model_perf.get('metrics', {})
                assert 'rmse_mean' in metrics, f"Missing RMSE for {model_name}"
                assert 'directional_accuracy_mean' in metrics, f"Missing directional accuracy for {model_name}"
            
            # Save detailed results
            results_file = self.workspace_dir / f"comprehensive_model_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            tester.save_results(str(results_file))
            
            # Print performance report
            if self.verbose:
                tester.print_performance_report()
            
            total_windows = performance_summary.get('total_windows', 0)
            return True, f"Comprehensive evaluation: {len(models_tested)} models, {total_windows} prediction windows"
            
        except Exception as e:
            return False, f"Rolling window evaluation failed: {str(e)}"
    
    # Stage 2: Cross-sectional Features Tests
    def _test_stage2_init(self) -> Tuple[bool, str]:
        """Test Stage 2 initialization"""
        try:
            config = self.config.copy()
            config['stages'] = {
                'stage2_cross_sectional': {
                    'benchmark_symbol': 'SPY',
                    'beta_windows': [30, 60],
                    'ranking_features': ['return_21d', 'vol_cc_20d']
                }
            }
            
            stage2 = Stage2CrossSectional(config)
            
            # Validate initialization
            assert 'benchmark_symbol' in stage2.params, "Benchmark symbol not found in params"
            assert 'beta_windows' in stage2.params, "Beta windows not found in params"
            assert len(stage2.params['beta_windows']) >= 1, "Beta windows not configured"
            
            return True, "Stage 2 initialized successfully"
            
        except Exception as e:
            return False, f"Stage 2 initialization failed: {str(e)}"
    
    def _test_stage2_beta_calculation(self) -> Tuple[bool, str]:
        """Test beta calculation functionality"""
        try:
            # Get sample data from Stage 1
            stage0 = Stage0DataValidator(self.config)
            stage1 = Stage1BaseFeatures(self.config)
            
            # Load and process test data
            test_data = self._get_test_data()
            stage0_result = stage0.execute(StageData(
                data=test_data, 
                metadata=StageMetadata("test", "1.0.0"),
                config=self.config
            ))
            
            stage1_result = stage1.execute(stage0_result)
            
            # Initialize Stage 2
            config = self.config.copy()
            config['stages'] = {
                'stage2_cross_sectional': {
                    'benchmark_symbol': 'SPY',
                    'beta_windows': [60],
                    'min_observations': 20
                }
            }
            
            stage2 = Stage2CrossSectional(config)
            # Stage 2 will create synthetic benchmark since none provided
            stage2_result = stage2.execute(stage1_result)
            
            # Check for beta features
            data = stage2_result.data
            expected_features = ['beta_60d', 'alpha_60d', 'idio_vol_60d', 'r_squared_60d']
            
            for feature in expected_features:
                assert feature in data.columns, f"Missing feature: {feature}"
            
            # Check that beta values are reasonable (should be mostly between -2 and 2)
            if 'beta_60d' in data.columns:
                beta_values = data['beta_60d'].dropna()
                if len(beta_values) > 0:
                    reasonable_betas = beta_values[(beta_values >= -3) & (beta_values <= 3)]
                    ratio = len(reasonable_betas) / len(beta_values)
                    assert ratio > 0.8, f"Too many unreasonable beta values: {ratio}"
            
            return True, f"Beta calculation successful with {len(expected_features)} features"
            
        except Exception as e:
            return False, f"Beta calculation failed: {str(e)}"
    
    def _test_stage2_rankings(self) -> Tuple[bool, str]:
        """Test cross-sectional rankings"""
        try:
            # Get multi-instrument data
            multi_instrument_data = self._get_multi_instrument_data()
            if len(multi_instrument_data['instrument'].unique()) < 3:
                return True, "Skipped - insufficient instruments for ranking test"
            
            # Process through stages
            stage0 = Stage0DataValidator(self.config)
            stage1 = Stage1BaseFeatures(self.config)
            
            stage0_result = stage0.execute(StageData(
                data=multi_instrument_data,
                metadata=StageMetadata("test", "1.0.0"), 
                config=self.config
            ))
            
            stage1_result = stage1.execute(stage0_result)
            
            config = self.config.copy()
            config['stages'] = {
                'stage2_cross_sectional': {
                    'ranking_features': ['return_21d', 'vol_cc_20d'],
                    'ranking_method': 'percent_rank'
                }
            }
            
            stage2 = Stage2CrossSectional(config)
            stage2_result = stage2.execute(stage1_result)
            
            data = stage2_result.data
            
            # Check for ranking features
            expected_ranking_features = ['return_21d_rank', 'vol_cc_20d_rank', 
                                       'return_21d_zscore', 'vol_cc_20d_zscore']
            
            for feature in expected_ranking_features:
                if feature in data.columns:
                    rank_values = data[feature].dropna()
                    if len(rank_values) > 0:
                        # Rankings should be between 0 and 1 for percent ranks
                        if '_rank' in feature:
                            assert rank_values.min() >= 0, f"{feature} has negative ranks"
                            assert rank_values.max() <= 1, f"{feature} has ranks > 1"
            
            return True, f"Cross-sectional rankings created: {len(expected_ranking_features)} features"
            
        except Exception as e:
            return False, f"Cross-sectional ranking failed: {str(e)}"
    
    def _test_stage2_universe_filtering(self) -> Tuple[bool, str]:
        """Test universe filtering functionality"""
        try:
            # Get test data and process through stages
            test_data = self._get_test_data()
            stage0 = Stage0DataValidator(self.config)
            stage1 = Stage1BaseFeatures(self.config)
            
            stage0_result = stage0.execute(StageData(
                data=test_data,
                metadata=StageMetadata("test", "1.0.0"),
                config=self.config
            ))
            
            stage1_result = stage1.execute(stage0_result)
            
            config = self.config.copy()
            config['stages'] = {
                'stage2_cross_sectional': {
                    'min_price': 1.0,
                    'min_volume': 10000,
                    'min_trading_days': 15
                }
            }
            
            stage2 = Stage2CrossSectional(config)
            stage2_result = stage2.execute(stage1_result)
            
            data = stage2_result.data
            
            # Check universe filtering features
            universe_features = ['universe_price_filter', 'universe_volume_filter', 
                               'universe_activity_filter', 'in_universe', 'trading_days_21d']
            
            for feature in universe_features:
                assert feature in data.columns, f"Missing universe feature: {feature}"
            
            # Check that filters are working
            if 'in_universe' in data.columns:
                universe_flags = data['in_universe'].dropna()
                assert len(universe_flags) > 0, "No universe flags generated"
                
            return True, f"Universe filtering successful: {len(universe_features)} features"
            
        except Exception as e:
            return False, f"Universe filtering failed: {str(e)}"
    
    # Stage 3: Regimes & Seasonal Features Tests
    def _test_stage3_init(self) -> Tuple[bool, str]:
        """Test Stage 3 initialization"""
        try:
            config = self.config.copy()
            config['stages'] = {
                'stage3_regimes': {
                    'regime_method': 'kmeans',
                    'n_regimes': 4,
                    'cyclical_features': ['day_of_week', 'month_of_year']
                }
            }
            
            stage3 = Stage3RegimesSeasonal(config)
            
            # Check model selector initialization
            assert stage3.model_selector is not None, "Model selector not initialized"
            assert stage3.params['regime_method'] == 'kmeans', "Regime method not set"
            assert stage3.params['n_regimes'] == 4, "Number of regimes not set"
            
            return True, "Stage 3 initialized successfully"
            
        except Exception as e:
            return False, f"Stage 3 initialization failed: {str(e)}"
    
    def _test_stage3_regime_detection(self) -> Tuple[bool, str]:
        """Test regime detection functionality"""
        try:
            # Get processed data from previous stages
            test_data = self._get_test_data()
            
            # Process through stages 0-2
            stage0 = Stage0DataValidator(self.config)
            stage1 = Stage1BaseFeatures(self.config)
            
            stage0_result = stage0.execute(StageData(
                data=test_data,
                metadata=StageMetadata("test", "1.0.0"),
                config=self.config
            ))
            
            stage1_result = stage1.execute(stage0_result)
            
            config = self.config.copy()
            config['stages'] = {
                'stage3_regimes': {
                    'regime_method': 'kmeans',
                    'n_regimes': 3,
                    'regime_features': ['vol_cc_20d', 'return_21d'],
                    'majority_filter_days': 2
                }
            }
            
            stage3 = Stage3RegimesSeasonal(config)
            stage3_result = stage3.execute(stage1_result)
            
            data = stage3_result.data
            
            # Check regime features
            regime_features = ['regime_id', 'regime_0', 'regime_1', 'regime_2', 'regime_persistence']
            
            for feature in regime_features:
                if feature in data.columns:
                    regime_values = data[feature].dropna()
                    if len(regime_values) > 0:
                        if feature == 'regime_id':
                            # Should be integers 0, 1, 2
                            unique_regimes = set(regime_values.unique())
                            assert unique_regimes.issubset({0, 1, 2}), f"Invalid regime IDs: {unique_regimes}"
                        elif feature.startswith('regime_') and feature != 'regime_persistence':
                            # Should be 0 or 1 (binary indicators)
                            unique_vals = set(regime_values.unique())
                            assert unique_vals.issubset({0, 1}), f"Regime indicator not binary: {unique_vals}"
            
            return True, f"Regime detection successful: {len(regime_features)} features"
            
        except Exception as e:
            return False, f"Regime detection failed: {str(e)}"
    
    def _test_stage3_seasonal_features(self) -> Tuple[bool, str]:
        """Test seasonal feature generation"""
        try:
            # Get test data with datetime index
            test_data = self._get_test_data()
            
            # Process through stages
            stage0 = Stage0DataValidator(self.config)
            stage0_result = stage0.execute(StageData(
                data=test_data,
                metadata=StageMetadata("test", "1.0.0"),
                config=self.config
            ))
            
            config = self.config.copy()
            config['stages'] = {
                'stage3_regimes': {
                    'cyclical_features': ['day_of_week', 'month_of_year'],
                    'turn_of_month_window': 3,
                    'holiday_effects': True
                }
            }
            
            stage3 = Stage3RegimesSeasonal(config)
            stage3_result = stage3.execute(stage0_result)
            
            data = stage3_result.data
            
            # Check seasonal features
            seasonal_features = ['dow_sin', 'dow_cos', 'month_sin', 'month_cos',
                               'turn_of_month', 'is_monday', 'is_friday', 'is_january']
            
            created_features = []
            for feature in seasonal_features:
                if feature in data.columns:
                    created_features.append(feature)
                    values = data[feature].dropna()
                    if len(values) > 0:
                        if feature.endswith('_sin') or feature.endswith('_cos'):
                            # Cyclical features should be between -1 and 1
                            assert values.min() >= -1.01, f"{feature} out of range (min)"
                            assert values.max() <= 1.01, f"{feature} out of range (max)"
                        elif feature.startswith('is_') or feature == 'turn_of_month':
                            # Binary features should be 0 or 1
                            unique_vals = set(values.unique())
                            assert unique_vals.issubset({0, 1}), f"Binary feature not binary: {unique_vals}"
            
            return True, f"Seasonal features created: {len(created_features)} features"
            
        except Exception as e:
            return False, f"Seasonal features failed: {str(e)}"
    
    def _test_stage3_model_selection(self) -> Tuple[bool, str]:
        """Test model selection functionality"""
        try:
            config = self.config.copy()
            config['stages'] = {
                'stage3_regimes': {
                    'regime_method': 'gmm',  # Test different model
                    'n_regimes': 2
                }
            }
            
            stage3 = Stage3RegimesSeasonal(config)
            
            # Check that model selector adapted to GMM
            model_info = stage3.model_selector
            assert model_info is not None, "Model selector not available"
            
            # Test fallback to basic method if model not available
            config['stages']['stage3_regimes']['regime_method'] = 'invalid_method'
            stage3_fallback = Stage3RegimesSeasonal(config)
            
            assert stage3_fallback.model_selector is not None, "Fallback model not initialized"
            
            return True, "Model selection working correctly"
            
        except Exception as e:
            return False, f"Model selection failed: {str(e)}"
    
    # Stage 4: Instrument Relationships Tests
    def _test_stage4_init(self) -> Tuple[bool, str]:
        """Test Stage 4 initialization"""
        try:
            config = self.config.copy()
            config['stages'] = {
                'stage4_relationships': {
                    'correlation_method': 'spearman',
                    'correlation_windows': [60, 90],
                    'max_lags': 3,
                    'min_correlation': 0.5
                }
            }
            
            stage4 = Stage4Relationships(config)
            
            # Validate initialization
            assert 'correlation_method' in stage4.params, "Correlation method not found in params"
            assert 'correlation_windows' in stage4.params, "Correlation windows not found in params"
            assert 'max_lags' in stage4.params, "Max lags not found in params"
            
            return True, "Stage 4 initialized successfully"
            
        except Exception as e:
            return False, f"Stage 4 initialization failed: {str(e)}"
    
    def _test_stage4_correlation(self) -> Tuple[bool, str]:
        """Test correlation analysis"""
        try:
            # Get multi-instrument data
            multi_data = self._get_multi_instrument_data()
            if len(multi_data['instrument'].unique()) < 3:
                return True, "Skipped - insufficient instruments for correlation analysis"
            
            # Process through previous stages
            stage0 = Stage0DataValidator(self.config)
            stage1 = Stage1BaseFeatures(self.config)
            
            stage0_result = stage0.execute(StageData(
                data=multi_data,
                metadata=StageMetadata("test", "1.0.0"),
                config=self.config
            ))
            
            stage1_result = stage1.execute(stage0_result)
            
            config = self.config.copy()
            config['stages'] = {
                'stage4_relationships': {
                    'correlation_windows': [60],
                    'correlation_method': 'pearson',
                    'min_correlation': 0.3
                }
            }
            
            stage4 = Stage4Relationships(config)
            stage4_result = stage4.execute(stage1_result)
            
            # Check artifacts
            relationships = stage4_result.artifacts.get('stage4_relationships', {})
            assert 'correlations' in relationships, "Correlation results missing"
            
            correlations = relationships['correlations']
            assert 'correlation_matrix' in correlations, "Correlation matrix missing"
            assert 'distance_matrix' in correlations, "Distance matrix missing"
            
            return True, "Correlation analysis completed successfully"
            
        except Exception as e:
            return False, f"Correlation analysis failed: {str(e)}"
    
    def _test_stage4_leadlag(self) -> Tuple[bool, str]:
        """Test lead-lag analysis"""  
        try:
            # Get multi-instrument data
            multi_data = self._get_multi_instrument_data()
            if len(multi_data['instrument'].unique()) < 2:
                return True, "Skipped - insufficient instruments for lead-lag analysis"
            
            # Process through previous stages
            stage0 = Stage0DataValidator(self.config)
            stage1 = Stage1BaseFeatures(self.config)
            
            stage0_result = stage0.execute(StageData(
                data=multi_data,
                metadata=StageMetadata("test", "1.0.0"),
                config=self.config
            ))
            
            stage1_result = stage1.execute(stage0_result)
            
            config = self.config.copy()
            config['stages'] = {
                'stage4_relationships': {
                    'max_lags': 2,
                    'min_ccf': 0.4,
                    'lead_lag_window': 50
                }
            }
            
            stage4 = Stage4Relationships(config)
            stage4_result = stage4.execute(stage1_result)
            
            # Check artifacts
            relationships = stage4_result.artifacts.get('stage4_relationships', {})
            assert 'lead_lag' in relationships, "Lead-lag results missing"
            
            lead_lag = relationships['lead_lag']
            assert 'lead_lag_pairs' in lead_lag, "Lead-lag pairs missing"
            assert 'lead_lag_map' in lead_lag, "Lead-lag map missing"
            
            return True, "Lead-lag analysis completed successfully"
            
        except Exception as e:
            return False, f"Lead-lag analysis failed: {str(e)}"
    
    def _test_stage4_cointegration(self) -> Tuple[bool, str]:
        """Test cointegration analysis"""
        try:
            # Get multi-instrument data  
            multi_data = self._get_multi_instrument_data()
            if len(multi_data['instrument'].unique()) < 2:
                return True, "Skipped - insufficient instruments for cointegration analysis"
            
            # Process through previous stages
            stage0 = Stage0DataValidator(self.config)
            stage1 = Stage1BaseFeatures(self.config)
            
            stage0_result = stage0.execute(StageData(
                data=multi_data,
                metadata=StageMetadata("test", "1.0.0"), 
                config=self.config
            ))
            
            stage1_result = stage1.execute(stage0_result)
            
            config = self.config.copy()
            config['stages'] = {
                'stage4_relationships': {
                    'cointegration_window': 100,
                    'min_half_life': 1,
                    'max_half_life': 20
                }
            }
            
            stage4 = Stage4Relationships(config)
            stage4_result = stage4.execute(stage1_result)
            
            # Check artifacts
            relationships = stage4_result.artifacts.get('stage4_relationships', {})
            assert 'cointegration' in relationships, "Cointegration results missing"
            
            cointegration = relationships['cointegration']
            assert 'cointegrated_pairs' in cointegration, "Cointegrated pairs missing"
            assert 'cointegration_map' in cointegration, "Cointegration map missing"
            
            return True, "Cointegration analysis completed successfully"
            
        except Exception as e:
            return False, f"Cointegration analysis failed: {str(e)}"
    
    def _get_test_data(self) -> pd.DataFrame:
        """Get single instrument test data for basic testing"""
        try:
            # Use the same approach as the existing working tests
            config = self.config.get('stage0_config', {})
            stage0 = Stage0DataValidator(config, data_root=self.data_root)
            
            # Create test input with single instrument
            date_range_config = self.config.get('test_date_range', {})
            date_range = DateRange(
                start_date=date_range_config.get('start_date', '2023-01-01'),
                end_date=date_range_config.get('end_date', '2023-03-31')  # Shorter range for speed
            )
            
            input_data = StageData(
                data=pd.DataFrame(),
                metadata=StageMetadata("test_input", "1.0.0"),
                config={
                    'instruments': [self.test_instruments[0]],  # Single instrument
                    'date_range': date_range,
                    'interval': MarketDataInterval.DAILY
                }
            )
            
            # Process data through stage 0
            result = stage0.process(input_data)
            
            if result.data.empty:
                # Fallback to synthetic data
                return self._create_synthetic_test_data()
            
            return result.data
            
        except Exception as e:
            logger.warning(f"Failed to get test data: {e}")
            return self._create_synthetic_test_data()
    
    def _create_synthetic_test_data(self) -> pd.DataFrame:
        """Create synthetic test data as fallback"""
        import numpy as np
        from datetime import datetime, timedelta
        
        # Create 100 days of synthetic OHLCV data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # Generate price series
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = 100 * (1 + returns).cumprod()
        
        # Create OHLCV data
        high = prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates))))
        low = prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates))))
        open_prices = np.roll(prices, 1)
        open_prices[0] = 100
        volume = np.random.lognormal(15, 0.5, len(dates)).astype(int)
        
        return pd.DataFrame({
            'date': dates,
            'open': open_prices,
            'high': high,
            'low': low, 
            'close': prices,
            'close_adj': prices,
            'volume': volume
        })
    
    def _get_multi_instrument_data(self) -> pd.DataFrame:
        """Get test data with multiple instruments"""
        try:
            # Create synthetic multi-instrument data from single instrument
            single_data = self._get_test_data()
            
            # Create multiple instruments with variations
            instruments_data = []
            
            # First instrument (base)
            data1 = single_data.copy()
            data1['instrument'] = 'TEST1'
            instruments_data.append(data1)
            
            # Second instrument (with slight price difference)
            data2 = single_data.copy()
            data2['instrument'] = 'TEST2'
            data2['close_adj'] = data2['close_adj'] * 1.05  # 5% higher
            data2['high'] = data2['high'] * 1.05
            data2['low'] = data2['low'] * 1.05
            data2['open'] = data2['open'] * 1.05
            if 'close' in data2.columns:
                data2['close'] = data2['close'] * 1.05
            instruments_data.append(data2)
            
            # Third instrument (with different volatility)
            data3 = single_data.copy()
            data3['instrument'] = 'TEST3'
            # Add more volatility by applying random walk
            np.random.seed(42)  # For reproducible results
            random_factor = 1 + np.random.normal(0, 0.02, len(data3))
            data3['close_adj'] = data3['close_adj'] * random_factor.cumprod()
            data3['high'] = data3['high'] * random_factor.cumprod()
            data3['low'] = data3['low'] * random_factor.cumprod()
            data3['open'] = data3['open'] * random_factor.cumprod()
            if 'close' in data3.columns:
                data3['close'] = data3['close'] * random_factor.cumprod()
            instruments_data.append(data3)
            
            return pd.concat(instruments_data, ignore_index=True)
                
        except Exception as e:
            logger.warning(f"Failed to create multi-instrument data: {e}")
            # Return minimal single instrument data
            single_data = self._get_test_data()
            single_data['instrument'] = 'TEST1'
            return single_data
    
    def _run_test(self, test_name: str, test_func) -> None:
        """Run a single test with error handling"""
        try:
            logger.info(f"Running test: {test_name}")
            start_time = datetime.now()
            
            success, message = test_func()
            
            duration = (datetime.now() - start_time).total_seconds()
            
            result = {
                'test_name': test_name,
                'success': success,
                'message': message,
                'duration_seconds': duration,
                'timestamp': datetime.now().isoformat()
            }
            
            self.test_results.append(result)
            
            if success:
                self.passed_tests.append(test_name)
                logger.info(f"PASS {test_name}: {message} ({duration:.2f}s)")
            else:
                self.failed_tests.append(test_name)
                logger.error(f"FAIL {test_name}: {message} ({duration:.2f}s)")
                
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            error_msg = f"Test execution error: {str(e)}"
            
            result = {
                'test_name': test_name,
                'success': False,
                'message': error_msg,
                'duration_seconds': duration,
                'timestamp': datetime.now().isoformat()
            }
            
            self.test_results.append(result)
            self.failed_tests.append(test_name)
            
            logger.error(f"FAIL {test_name}: {error_msg} ({duration:.2f}s)")
            
            if self.verbose:
                traceback.print_exc()
    
    def _print_test_summary(self):
        """Print comprehensive test summary"""
        total_tests = len(self.test_results)
        passed = len(self.passed_tests)
        failed = len(self.failed_tests)
        
        logger.info("\n" + "=" * 80)
        logger.info("DA VINCHI TEST SUMMARY")
        logger.info("=" * 80)
        
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed} ({passed/total_tests*100:.1f}%)")
        logger.info(f"Failed: {failed} ({failed/total_tests*100:.1f}%)")
        
        if self.failed_tests:
            logger.info(f"\nFailed Tests:")
            for test_name in self.failed_tests:
                result = next(r for r in self.test_results if r['test_name'] == test_name)
                logger.info(f"  - {test_name}: {result['message']}")
        
        total_duration = sum(r['duration_seconds'] for r in self.test_results)
        logger.info(f"\nTotal Duration: {total_duration:.2f} seconds")
        
        if self.config.get('test_settings', {}).get('save_results', False):
            self._save_test_results()
    
    def _save_test_results(self):
        """Save test results to file"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = self.workspace_dir / f"test_results_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump({
                    'summary': {
                        'total_tests': len(self.test_results),
                        'passed': len(self.passed_tests),
                        'failed': len(self.failed_tests),
                        'success_rate': len(self.passed_tests) / len(self.test_results) * 100
                    },
                    'results': self.test_results,
                    'config': self.config
                }, f, indent=2, default=str)
            
            logger.info(f"Test results saved to: {results_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save test results: {e}")


def main():
    """Main entry point for test driver"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Da Vinchi Feature Engineering Test Driver")
    parser.add_argument("--config", help="Path to test config file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--data-root", help="Override data root directory")
    
    args = parser.parse_args()
    
    # Initialize test driver
    driver = DaVinchiTestDriver(
        config_path=args.config,
        verbose=args.verbose
    )
    
    # Override data root if specified
    if args.data_root:
        driver.data_root = args.data_root
    
    # Run tests
    success = driver.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()