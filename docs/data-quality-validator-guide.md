# 2Trade Data Quality Validator Guide

## Overview

The enhanced Data Quality Validator in Consuela provides comprehensive validation and analysis for all data types in the 2Trade system, including market data, economic indicators, news data, and Google Trends data.

## Features

### 🔍 Comprehensive Data Analysis
- **Market Data**: OHLCV validation, price relationship checks, volume analysis, technical indicators presence
- **Economic Data**: FRED indicator validation, freshness analysis, null value detection, extreme value identification
- **News Data**: Dummy data detection, sentiment analysis, source quality assessment, article quality checks
- **Trends Data**: Search interest validation, metadata completeness, related queries coverage

### ⚙️ Configuration-Driven Validation
- YAML-based configuration system (`consuela/config/data-quality-config.yml`)
- Customizable thresholds and validation rules
- Quality scoring weights and penalty systems
- Performance optimization settings

### 📊 Advanced Quality Metrics
- **Quality Scores**: 0-100 scoring system with configurable weights
- **Issue Classification**: Critical, High, Medium, Low severity levels
- **Data Freshness**: Configurable freshness thresholds by data type
- **Completeness Analysis**: Missing data gap detection and analysis

### 📈 Enhanced Reporting
- JSON and human-readable text reports
- Historical tracking and trend analysis
- Actionable recommendations
- Quality score breakdowns by data type

## Usage

### Basic Commands

```bash
# Navigate to Consuela scripts directory
cd consuela/scripts

# Quick evaluation (limited analysis for faster results)
python data_quality_evaluator.py --quick

# Full comprehensive evaluation
python data_quality_evaluator.py

# Verbose output with detailed logging
python data_quality_evaluator.py --verbose

# Custom data root directory
python data_quality_evaluator.py --data-root /path/to/your/data

# Custom output directory for reports
python data_quality_evaluator.py --output-dir /path/to/reports
```

### Configuration Options

The validator uses `consuela/config/data-quality-config.yml` for configuration:

#### Market Data Validation Rules
```yaml
market_data:
  price_validation:
    min_price: 0.01
    max_daily_change: 0.50
    max_intraday_gap: 0.20
  
  volume_validation:
    min_volume: 0
    max_volume_spike: 100
    min_volume_threshold: 0.01
  
  completeness:
    min_completeness_ratio: 0.80
    max_gap_days: 10
    freshness_threshold_days: 7
```

#### Economic Data Validation Rules
```yaml
economic_data:
  freshness_thresholds:
    "Daily": 7
    "Weekly": 14
    "Monthly": 45
    "Quarterly": 120
    "Annual": 400
  
  null_thresholds:
    max_null_percentage: 15
    high_null_threshold: 25
```

#### News Data Validation Rules
```yaml
news_data:
  article_quality:
    min_headline_length: 10
    max_headline_length: 200
    min_summary_length: 20
  
  dummy_data_patterns:
    dummy_sources:
      - "sample_news_generator"
      - "test_source"
    dummy_headline_patterns:
      - "Sample news headline"
      - "Test headline"
```

## Quality Scoring System

### Score Ranges
- **95-100**: Excellent quality
- **80-94**: Good quality  
- **60-79**: Acceptable quality
- **40-59**: Poor quality
- **0-39**: Critical issues

### Penalty Weights
- **Critical Issues**: 25 point penalty
- **High Severity**: 15 point penalty
- **Medium Severity**: 8 point penalty
- **Low Severity**: 3 point penalty

## Report Interpretation

### Market Data Quality Issues

**Critical Issues:**
- Price relationship violations (High < Low, etc.)
- Zero or negative prices
- Invalid OHLC data

**High Severity:**
- Stale data (>30 days old)
- Low data completeness (<50%)
- Frequent large price gaps

**Medium Severity:**
- Moderate data staleness (7-30 days)
- Low completeness (50-80%)
- Volume anomalies

### Economic Data Quality Issues

**High Priority Indicators:**
- UNRATE (Unemployment Rate)
- FEDFUNDS (Federal Funds Rate)
- CPIAUCSL (Consumer Price Index)
- GDP (Gross Domestic Product)
- PAYEMS (Nonfarm Payrolls)
- VIXCLS (VIX Volatility Index)

**Common Issues:**
- High null value percentages (>15%)
- Stale data beyond frequency thresholds
- Missing critical economic indicators
- Extreme outlier values

### News Data Quality Issues

**Dummy Data Detection:**
- Articles from "sample_news_generator"
- Headlines starting with "Sample news headline"
- Test or dummy source patterns

**Quality Checks:**
- Empty or very short headlines
- Missing required fields (date, symbol, source)
- Invalid sentiment scores
- High dummy data contamination ratio

### Trends Data Quality Issues

**Validation Checks:**
- Invalid search scores (outside 0-100 range)
- Missing required metadata fields
- Empty search interest data
- Insufficient related queries

## Best Practices

### Regular Monitoring
1. **Daily Quick Checks**: Run `--quick` evaluation daily
2. **Weekly Comprehensive**: Full evaluation weekly
3. **Monthly Trending**: Compare quality metrics over time
4. **Alert Thresholds**: Set up alerts for critical issues

### Data Quality Maintenance
1. **Address Critical Issues First**: Focus on critical severity issues
2. **Fill Data Gaps**: Prioritize missing data combinations
3. **Update Stale Data**: Regular data refresh schedules
4. **Remove Dummy Data**: Clean up test/sample data

### Configuration Tuning
1. **Adjust Thresholds**: Customize based on your data patterns
2. **Weight Optimization**: Tune quality score weights
3. **Performance Settings**: Optimize sampling for large datasets
4. **Alert Configuration**: Set appropriate alert thresholds

## Advanced Features

### Historical Tracking
- Quality metrics tracked over time
- Trend analysis and comparison
- 365-day historical retention
- Comparison with 7, 30, 90-day periods

### Performance Optimization
- Configurable sampling for large datasets
- Parallel processing support
- Result caching (6-hour duration)
- Quick mode for faster analysis

### Integration Points
- **Odin's Eye Integration**: Seamless data access
- **Consuela Configuration**: Uses instrument lists
- **Export Formats**: JSON, TXT, CSV support

## Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Install missing dependencies
pip install -r consuela/config/requirements.txt
```

**Data Access Errors:**
```bash
# Verify Odin's Eye configuration
# Check data root directory permissions
# Ensure instrument lists are accessible
```

**Configuration Errors:**
```bash
# Validate YAML configuration file
# Check file permissions
# Verify configuration syntax
```

### Performance Issues

**Large Dataset Handling:**
- Use `--quick` mode for faster results
- Adjust sampling configuration
- Enable parallel processing
- Use result caching

**Memory Usage:**
- Reduce max_instruments_sample in config
- Process data in batches
- Monitor system resources

## Output Files

### Generated Reports
- `data_quality_report_YYYYMMDD_HHMMSS.json`: Comprehensive JSON report
- `data_quality_summary_YYYYMMDD_HHMMSS.txt`: Human-readable summary
- `data_quality_evaluation.log`: Detailed execution log

### Report Locations
- Default: `consuela/house-keeping-report/`
- Custom: Use `--output-dir` parameter

## API Integration

### Python Usage
```python
from consuela.scripts.data_quality_evaluator import DataQualityEvaluator

# Initialize with custom configuration
evaluator = DataQualityEvaluator(
    data_root="/custom/data/path",
    output_dir="/custom/reports/path",
    config_path="/custom/config.yml"
)

# Run comprehensive evaluation
report_path = evaluator.run_full_evaluation()

# Run specific analysis types
market_analysis = evaluator.evaluate_data_coverage()
economic_analysis = evaluator.evaluate_economic_data_coverage()
news_analysis = evaluator.evaluate_news_data_coverage()
trends_analysis = evaluator.evaluate_trends_data_coverage()
```

## Maintenance

### Configuration Updates
1. **Regular Review**: Review thresholds quarterly
2. **Threshold Tuning**: Adjust based on data quality trends
3. **New Data Types**: Add validation rules for new data sources
4. **Performance Optimization**: Update sampling and caching settings

### System Integration
1. **Automated Scheduling**: Set up cron jobs for regular evaluation
2. **Alert Systems**: Integrate with monitoring systems
3. **Dashboard Integration**: Connect to quality dashboards
4. **Reporting Automation**: Automated report distribution

This enhanced data quality validator provides comprehensive analysis and monitoring capabilities for all data types in your 2Trade system, ensuring high-quality data for model training and trading decisions.