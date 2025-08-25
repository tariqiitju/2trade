#!/usr/bin/env python3
"""
Populate Comprehensive Instrument List

Downloads instrument lists from various public sources and creates a comprehensive
base-list-all.yml with thousands of instruments including:
- All NASDAQ stocks
- All NYSE stocks  
- Major ETFs
- International ADRs
- REITs
- Delisted historical stocks
"""

import requests
import pandas as pd
import yaml
from typing import Dict, List, Any
from pathlib import Path
import logging
import time
import json

logger = logging.getLogger(__name__)


class InstrumentListPopulator:
    """Downloads and consolidates instrument lists from multiple sources"""
    
    def __init__(self):
        self.instruments = []
        self.sources_used = []
        
    def download_nasdaq_listed(self) -> List[Dict[str, Any]]:
        """Download all NASDAQ listed stocks"""
        print("Downloading NASDAQ listed stocks...")
        
        try:
            # NASDAQ provides official stock screener data
            url = "https://api.nasdaq.com/api/screener/stocks"
            params = {
                'tableonly': 'true',
                'limit': '25000',
                'offset': '0'
            }
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, params=params, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                stocks = data.get('data', {}).get('table', {}).get('rows', [])
                
                instruments = []
                for stock in stocks:
                    if isinstance(stock, dict):
                        symbol = stock.get('symbol', '').strip()
                        name = stock.get('name', '').strip()
                        sector = stock.get('sector', '').strip() or 'Unknown'
                        
                        if symbol and symbol.replace('.', '').replace('-', '').isalnum():
                            instruments.append({
                                'symbol': symbol,
                                'name': name,
                                'sector': self._normalize_sector(sector),
                                'market_cap_category': 'large_cap',
                                'exchange': 'NASDAQ',
                                'status': 'active',
                                'source': 'nasdaq_api'
                            })
                
                print(f"Downloaded {len(instruments)} NASDAQ stocks")
                self.sources_used.append("NASDAQ API")
                return instruments
                
        except Exception as e:
            print(f"Failed to download NASDAQ data: {e}")
        
        return []
    
    def download_nyse_listed(self) -> List[Dict[str, Any]]:
        """Download NYSE listed stocks"""
        print("Downloading NYSE listed stocks...")
        
        try:
            # Use SEC EDGAR company tickers list
            url = "https://www.sec.gov/files/company_tickers.json"
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; InvestmentResearch/1.0)',
                'Accept': 'application/json'
            }
            
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                
                instruments = []
                for key, company in data.items():
                    if isinstance(company, dict):
                        symbol = company.get('ticker', '').strip()
                        name = company.get('title', '').strip()
                        
                        if symbol and len(symbol) <= 5 and symbol.replace('.', '').replace('-', '').isalnum():
                            instruments.append({
                                'symbol': symbol,
                                'name': name,
                                'sector': 'Unknown',  # SEC data doesn't include sector
                                'market_cap_category': 'large_cap',
                                'exchange': 'NYSE',  # Assumption - could be NASDAQ too
                                'status': 'active',
                                'source': 'sec_edgar'
                            })
                
                print(f"Downloaded {len(instruments)} stocks from SEC EDGAR")
                self.sources_used.append("SEC EDGAR")
                return instruments[:2000]  # Limit to avoid too many
                
        except Exception as e:
            print(f"Failed to download SEC EDGAR data: {e}")
        
        return []
    
    def download_etf_list(self) -> List[Dict[str, Any]]:
        """Download comprehensive ETF list"""
        print("Downloading ETF list...")
        
        # Comprehensive list of major ETFs
        major_etfs = [
            # Market Index ETFs
            {'symbol': 'SPY', 'name': 'SPDR S&P 500 ETF Trust', 'category': 'Large Cap Blend'},
            {'symbol': 'VOO', 'name': 'Vanguard S&P 500 ETF', 'category': 'Large Cap Blend'},
            {'symbol': 'IVV', 'name': 'iShares Core S&P 500 ETF', 'category': 'Large Cap Blend'},
            {'symbol': 'QQQ', 'name': 'Invesco QQQ Trust', 'category': 'Large Cap Growth'},
            {'symbol': 'VTI', 'name': 'Vanguard Total Stock Market ETF', 'category': 'Large Cap Blend'},
            {'symbol': 'IWM', 'name': 'iShares Russell 2000 ETF', 'category': 'Small Cap Blend'},
            {'symbol': 'VEA', 'name': 'Vanguard FTSE Developed Markets ETF', 'category': 'Foreign Large Blend'},
            {'symbol': 'VWO', 'name': 'Vanguard FTSE Emerging Markets ETF', 'category': 'Diversified Emerging Mkts'},
            {'symbol': 'EFA', 'name': 'iShares MSCI EAFE ETF', 'category': 'Foreign Large Blend'},
            {'symbol': 'EEM', 'name': 'iShares MSCI Emerging Markets ETF', 'category': 'Diversified Emerging Mkts'},
            
            # Sector ETFs
            {'symbol': 'XLK', 'name': 'Technology Select Sector SPDR Fund', 'category': 'Technology'},
            {'symbol': 'XLF', 'name': 'Financial Select Sector SPDR Fund', 'category': 'Financial'},
            {'symbol': 'XLV', 'name': 'Health Care Select Sector SPDR Fund', 'category': 'Healthcare'},
            {'symbol': 'XLE', 'name': 'Energy Select Sector SPDR Fund', 'category': 'Energy'},
            {'symbol': 'XLI', 'name': 'Industrial Select Sector SPDR Fund', 'category': 'Industrial'},
            {'symbol': 'XLP', 'name': 'Consumer Staples Select Sector SPDR Fund', 'category': 'Consumer Staples'},
            {'symbol': 'XLY', 'name': 'Consumer Discretionary Select Sector SPDR Fund', 'category': 'Consumer Discretionary'},
            {'symbol': 'XLU', 'name': 'Utilities Select Sector SPDR Fund', 'category': 'Utilities'},
            {'symbol': 'XLB', 'name': 'Materials Select Sector SPDR Fund', 'category': 'Materials'},
            {'symbol': 'XLRE', 'name': 'Real Estate Select Sector SPDR Fund', 'category': 'Real Estate'},
            {'symbol': 'XBI', 'name': 'SPDR S&P Biotech ETF', 'category': 'Healthcare'},
            
            # Bond ETFs
            {'symbol': 'AGG', 'name': 'iShares Core U.S. Aggregate Bond ETF', 'category': 'Intermediate Core Bond'},
            {'symbol': 'BND', 'name': 'Vanguard Total Bond Market ETF', 'category': 'Intermediate Core Bond'},
            {'symbol': 'TLT', 'name': 'iShares 20+ Year Treasury Bond ETF', 'category': 'Long Government'},
            {'symbol': 'SHY', 'name': 'iShares 1-3 Year Treasury Bond ETF', 'category': 'Short Government'},
            {'symbol': 'IEF', 'name': 'iShares 7-10 Year Treasury Bond ETF', 'category': 'Intermediate Government'},
            {'symbol': 'TIP', 'name': 'iShares TIPS Bond ETF', 'category': 'Inflation-Protected Bond'},
            {'symbol': 'HYG', 'name': 'iShares iBoxx High Yield Corporate Bond ETF', 'category': 'High Yield Bond'},
            {'symbol': 'LQD', 'name': 'iShares iBoxx Investment Grade Corporate Bond ETF', 'category': 'Corporate Bond'},
            
            # Commodity ETFs
            {'symbol': 'GLD', 'name': 'SPDR Gold Trust', 'category': 'Commodities Precious Metals'},
            {'symbol': 'SLV', 'name': 'iShares Silver Trust', 'category': 'Commodities Precious Metals'},
            {'symbol': 'USO', 'name': 'United States Oil Fund', 'category': 'Commodities Energy'},
            {'symbol': 'UNG', 'name': 'United States Natural Gas Fund', 'category': 'Commodities Energy'},
            {'symbol': 'DBA', 'name': 'Invesco DB Agriculture Fund', 'category': 'Commodities Agriculture'},
            {'symbol': 'PDBC', 'name': 'Invesco Optimum Yield Diversified Commodity Strategy No K-1 ETF', 'category': 'Commodities Broad Basket'},
            
            # International ETFs
            {'symbol': 'IEFA', 'name': 'iShares Core MSCI EAFE ETF', 'category': 'Foreign Large Blend'},
            {'symbol': 'IEMG', 'name': 'iShares Core MSCI Emerging Markets ETF', 'category': 'Diversified Emerging Mkts'},
            {'symbol': 'VGK', 'name': 'Vanguard FTSE Europe ETF', 'category': 'Europe Stock'},
            {'symbol': 'VPL', 'name': 'Vanguard FTSE Pacific ETF', 'category': 'Pacific/Asia ex-Japan Stk'},
            {'symbol': 'EWJ', 'name': 'iShares MSCI Japan ETF', 'category': 'Japan Stock'},
            {'symbol': 'EWZ', 'name': 'iShares MSCI Brazil ETF', 'category': 'Latin America Stock'},
            {'symbol': 'FXI', 'name': 'iShares China Large-Cap ETF', 'category': 'China Region'},
            {'symbol': 'INDA', 'name': 'iShares MSCI India ETF', 'category': 'India Equity'},
            
            # Style ETFs
            {'symbol': 'IWF', 'name': 'iShares Russell 1000 Growth ETF', 'category': 'Large Growth'},
            {'symbol': 'IWD', 'name': 'iShares Russell 1000 Value ETF', 'category': 'Large Value'},
            {'symbol': 'IJH', 'name': 'iShares Core S&P Mid-Cap ETF', 'category': 'Mid-Cap Blend'},
            {'symbol': 'IJR', 'name': 'iShares Core S&P Small-Cap ETF', 'category': 'Small Blend'},
            {'symbol': 'VTV', 'name': 'Vanguard Value ETF', 'category': 'Large Value'},
            {'symbol': 'VUG', 'name': 'Vanguard Growth ETF', 'category': 'Large Growth'},
            
            # REITs
            {'symbol': 'VNQ', 'name': 'Vanguard Real Estate ETF', 'category': 'Real Estate'},
            {'symbol': 'IYR', 'name': 'iShares U.S. Real Estate ETF', 'category': 'Real Estate'},
            {'symbol': 'RWR', 'name': 'SPDR Dow Jones REIT ETF', 'category': 'Real Estate'},
            {'symbol': 'SCHH', 'name': 'Schwab U.S. REIT ETF', 'category': 'Real Estate'},
            
            # Thematic/Innovation ETFs
            {'symbol': 'ARKK', 'name': 'ARK Innovation ETF', 'category': 'Miscellaneous Sector'},
            {'symbol': 'ARKQ', 'name': 'ARK Autonomous Technology & Robotics ETF', 'category': 'Technology'},
            {'symbol': 'ARKW', 'name': 'ARK Next Generation Internet ETF', 'category': 'Technology'},
            {'symbol': 'ICLN', 'name': 'iShares Global Clean Energy ETF', 'category': 'Alternative Energy'},
            {'symbol': 'JETS', 'name': 'U.S. Global Jets ETF', 'category': 'Transportation'},
            {'symbol': 'SOXX', 'name': 'iShares PHLX Semiconductor ETF', 'category': 'Technology'},
            {'symbol': 'IBB', 'name': 'iShares Nasdaq Biotechnology ETF', 'category': 'Healthcare'},
            {'symbol': 'IGSB', 'name': 'iShares 1-3 Year Credit Bond ETF', 'category': 'Short-Term Bond'},
            
            # Volatility
            {'symbol': 'VXX', 'name': 'iPath Series B S&P 500 VIX Short-Term Futures ETN', 'category': 'Volatility'},
            {'symbol': 'UVXY', 'name': 'ProShares Ultra VIX Short-Term Futures ETF', 'category': 'Volatility'},
        ]
        
        instruments = []
        for etf in major_etfs:
            instruments.append({
                'symbol': etf['symbol'],
                'name': etf['name'],
                'sector': 'ETF',
                'market_cap_category': 'etf',
                'exchange': 'NYSE',  # Most ETFs trade on NYSE Arca
                'status': 'active',
                'category': etf['category'],
                'source': 'curated_etf_list'
            })
        
        print(f"Added {len(instruments)} major ETFs")
        self.sources_used.append("Curated ETF List")
        return instruments
    
    def get_international_adrs(self) -> List[Dict[str, Any]]:
        """Get major international ADRs"""
        print("Adding international ADRs...")
        
        adrs = [
            # Chinese ADRs
            {'symbol': 'BABA', 'name': 'Alibaba Group Holding Limited', 'country': 'China', 'sector': 'Technology'},
            {'symbol': 'JD', 'name': 'JD.com Inc.', 'country': 'China', 'sector': 'Consumer Discretionary'},
            {'symbol': 'NTES', 'name': 'NetEase Inc.', 'country': 'China', 'sector': 'Technology'},
            {'symbol': 'BIDU', 'name': 'Baidu Inc.', 'country': 'China', 'sector': 'Technology'},
            {'symbol': 'NIO', 'name': 'NIO Inc.', 'country': 'China', 'sector': 'Consumer Discretionary'},
            {'symbol': 'LI', 'name': 'Li Auto Inc.', 'country': 'China', 'sector': 'Consumer Discretionary'},
            {'symbol': 'XPEV', 'name': 'XPeng Inc.', 'country': 'China', 'sector': 'Consumer Discretionary'},
            {'symbol': 'PDD', 'name': 'PDD Holdings Inc.', 'country': 'China', 'sector': 'Consumer Discretionary'},
            
            # Taiwan
            {'symbol': 'TSM', 'name': 'Taiwan Semiconductor Manufacturing Company Limited', 'country': 'Taiwan', 'sector': 'Technology'},
            
            # Netherlands/Europe
            {'symbol': 'ASML', 'name': 'ASML Holding N.V.', 'country': 'Netherlands', 'sector': 'Technology'},
            {'symbol': 'UL', 'name': 'Unilever PLC', 'country': 'United Kingdom', 'sector': 'Consumer Staples'},
            {'symbol': 'NVO', 'name': 'Novo Nordisk A/S', 'country': 'Denmark', 'sector': 'Healthcare'},
            {'symbol': 'NVS', 'name': 'Novartis AG', 'country': 'Switzerland', 'sector': 'Healthcare'},
            {'symbol': 'TM', 'name': 'Toyota Motor Corporation', 'country': 'Japan', 'sector': 'Consumer Discretionary'},
            {'symbol': 'SNY', 'name': 'Sanofi', 'country': 'France', 'sector': 'Healthcare'},
            {'symbol': 'SAP', 'name': 'SAP SE', 'country': 'Germany', 'sector': 'Technology'},
            
            # Canadian
            {'symbol': 'SHOP', 'name': 'Shopify Inc.', 'country': 'Canada', 'sector': 'Technology'},
            {'symbol': 'RY', 'name': 'Royal Bank of Canada', 'country': 'Canada', 'sector': 'Financial Services'},
            {'symbol': 'TD', 'name': 'Toronto-Dominion Bank', 'country': 'Canada', 'sector': 'Financial Services'},
            
            # Brazilian
            {'symbol': 'VALE', 'name': 'Vale S.A.', 'country': 'Brazil', 'sector': 'Materials'},
            {'symbol': 'PBR', 'name': 'Petróleo Brasileiro S.A. - Petrobras', 'country': 'Brazil', 'sector': 'Energy'},
            
            # Other Notable
            {'symbol': 'SE', 'name': 'Sea Limited', 'country': 'Singapore', 'sector': 'Technology'},
            {'symbol': 'GRAB', 'name': 'Grab Holdings Limited', 'country': 'Singapore', 'sector': 'Technology'},
        ]
        
        instruments = []
        for adr in adrs:
            instruments.append({
                'symbol': adr['symbol'],
                'name': adr['name'],
                'sector': adr['sector'],
                'market_cap_category': 'large_cap',
                'exchange': 'NYSE',
                'status': 'active',
                'country': adr['country'],
                'source': 'curated_adr_list'
            })
        
        print(f"Added {len(instruments)} international ADRs")
        self.sources_used.append("Curated ADR List")
        return instruments
    
    def get_delisted_historical(self) -> List[Dict[str, Any]]:
        """Get historically significant delisted companies"""
        print("Adding delisted historical companies...")
        
        delisted = [
            # Financial Crisis Era
            {'symbol': 'LEH', 'name': 'Lehman Brothers Holdings Inc.', 'sector': 'Financial Services', 'delisting_date': '2008-09-15', 'delisting_reason': 'bankruptcy'},
            {'symbol': 'BSC', 'name': 'Bear Stearns Companies Inc.', 'sector': 'Financial Services', 'delisting_date': '2008-05-30', 'delisting_reason': 'acquisition'},
            {'symbol': 'WB', 'name': 'Wachovia Corporation', 'sector': 'Financial Services', 'delisting_date': '2008-12-31', 'delisting_reason': 'acquisition'},
            {'symbol': 'WM', 'name': 'Washington Mutual Inc.', 'sector': 'Financial Services', 'delisting_date': '2008-09-26', 'delisting_reason': 'bankruptcy'},
            
            # Dot-com Era
            {'symbol': 'WCOM', 'name': 'WorldCom Inc.', 'sector': 'Communication Services', 'delisting_date': '2002-07-21', 'delisting_reason': 'bankruptcy'},
            {'symbol': 'ENRNQ', 'name': 'Enron Corporation', 'sector': 'Energy', 'delisting_date': '2001-12-02', 'delisting_reason': 'bankruptcy'},
            
            # Tech Acquisitions
            {'symbol': 'YHOO', 'name': 'Yahoo! Inc.', 'sector': 'Technology', 'delisting_date': '2017-06-13', 'delisting_reason': 'acquisition'},
            {'symbol': 'AOL', 'name': 'AOL Inc.', 'sector': 'Communication Services', 'delisting_date': '2015-06-23', 'delisting_reason': 'acquisition'},
            
            # Recent Bankruptcies
            {'symbol': 'BBBY', 'name': 'Bed Bath & Beyond Inc.', 'sector': 'Consumer Discretionary', 'delisting_date': '2023-05-03', 'delisting_reason': 'bankruptcy'},
            {'symbol': 'EXPR', 'name': 'Express Inc.', 'sector': 'Consumer Discretionary', 'delisting_date': '2023-04-22', 'delisting_reason': 'bankruptcy'},
            {'symbol': 'REVG', 'name': 'REV Group Inc.', 'sector': 'Industrial', 'delisting_date': '2023-03-15', 'delisting_reason': 'acquisition'},
            
            # Energy Bankruptcies
            {'symbol': 'CHK', 'name': 'Chesapeake Energy Corporation', 'sector': 'Energy', 'delisting_date': '2020-06-28', 'delisting_reason': 'bankruptcy'},
            {'symbol': 'WLL', 'name': 'Whiting Petroleum Corporation', 'sector': 'Energy', 'delisting_date': '2020-04-01', 'delisting_reason': 'bankruptcy'},
            
            # Retail Bankruptcies
            {'symbol': 'JCP', 'name': 'J.C. Penney Company Inc.', 'sector': 'Consumer Discretionary', 'delisting_date': '2020-05-15', 'delisting_reason': 'bankruptcy'},
            {'symbol': 'HTZ', 'name': 'Hertz Global Holdings Inc.', 'sector': 'Consumer Discretionary', 'delisting_date': '2020-06-16', 'delisting_reason': 'bankruptcy'},
        ]
        
        instruments = []
        for company in delisted:
            instruments.append({
                'symbol': company['symbol'],
                'name': company['name'],
                'sector': company['sector'],
                'market_cap_category': 'large_cap',
                'exchange': 'NYSE',
                'status': 'delisted',
                'delisting_date': company['delisting_date'],
                'delisting_reason': company['delisting_reason'],
                'source': 'curated_delisted_list'
            })
        
        print(f"Added {len(instruments)} delisted historical companies")
        self.sources_used.append("Curated Delisted List")
        return instruments
    
    def _normalize_sector(self, sector: str) -> str:
        """Normalize sector names"""
        sector_map = {
            'Technology': 'Technology',
            'Consumer Discretionary': 'Consumer Discretionary',
            'Consumer Staples': 'Consumer Staples',
            'Health Care': 'Healthcare',
            'Healthcare': 'Healthcare',
            'Financials': 'Financial Services',
            'Financial Services': 'Financial Services',
            'Energy': 'Energy',
            'Materials': 'Materials',
            'Industrials': 'Industrial',
            'Industrial': 'Industrial',
            'Utilities': 'Utilities',
            'Real Estate': 'Real Estate',
            'Communication Services': 'Communication Services',
            'Communications': 'Communication Services',
        }
        
        return sector_map.get(sector, sector)
    
    def consolidate_instruments(self) -> List[Dict[str, Any]]:
        """Consolidate and deduplicate all instruments"""
        print("\nConsolidating all instrument lists...")
        
        all_instruments = []
        
        # Download from various sources
        all_instruments.extend(self.download_nasdaq_listed())
        time.sleep(1)  # Rate limiting
        
        all_instruments.extend(self.download_nyse_listed())
        time.sleep(1)
        
        all_instruments.extend(self.download_etf_list())
        all_instruments.extend(self.get_international_adrs())
        all_instruments.extend(self.get_delisted_historical())
        
        # Deduplicate by symbol
        seen_symbols = set()
        unique_instruments = []
        
        for instrument in all_instruments:
            symbol = instrument['symbol']
            if symbol not in seen_symbols:
                seen_symbols.add(symbol)
                unique_instruments.append(instrument)
        
        print(f"\nTotal instruments after deduplication: {len(unique_instruments)}")
        print(f"Sources used: {', '.join(self.sources_used)}")
        
        return unique_instruments
    
    def create_base_list_yaml(self, instruments: List[Dict[str, Any]], output_path: str):
        """Create the comprehensive base-list-all.yml file"""
        print(f"\nCreating {output_path} with {len(instruments)} instruments...")
        
        # Group instruments by category
        mega_cap = []
        large_cap = []
        mid_cap = []
        etfs = []
        adrs = []
        delisted = []
        
        for inst in instruments:
            if inst['status'] == 'delisted':
                delisted.append(inst)
            elif inst.get('market_cap_category') == 'etf':
                etfs.append(inst)
            elif inst.get('country'):  # Has country = ADR
                adrs.append(inst)
            elif inst.get('market_cap_category') == 'mega_cap':
                mega_cap.append(inst)
            else:
                large_cap.append(inst)
        
        # Create YAML content
        yaml_content = {
            'version': '2.0.0',
            'last_updated': '2024-08-24',
            'total_count': len(instruments),
            'instruments': instruments,
            'metadata': {
                'total_instruments': len(instruments),
                'active_count': len([i for i in instruments if i['status'] == 'active']),
                'delisted_count': len(delisted),
                'etf_count': len(etfs),
                'international_adr_count': len(adrs),
                'sources_used': self.sources_used,
                'categories': {
                    'mega_cap': len(mega_cap),
                    'large_cap': len(large_cap),
                    'mid_cap': len(mid_cap),
                    'etfs': len(etfs),
                    'international_adrs': len(adrs),
                    'delisted_historical': len(delisted)
                }
            }
        }
        
        # Write to file
        with open(output_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, indent=2, allow_unicode=True)
        
        print(f"[OK] Created {output_path}")
        print(f"   Total instruments: {len(instruments)}")
        print(f"   Active: {len(instruments) - len(delisted)}")
        print(f"   Delisted: {len(delisted)}")
        print(f"   ETFs: {len(etfs)}")
        print(f"   International ADRs: {len(adrs)}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Populate comprehensive instrument list")
    parser.add_argument("--output", default="../consuela/config/instrument-list/base-list-all.yml", 
                       help="Output file path")
    parser.add_argument("--max-instruments", type=int, default=10000, 
                       help="Maximum instruments to include")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("POPULATING COMPREHENSIVE INSTRUMENT LIST")
    print("=" * 60)
    
    # Initialize populator
    populator = InstrumentListPopulator()
    
    # Consolidate all instruments
    instruments = populator.consolidate_instruments()
    
    # Limit if requested
    if len(instruments) > args.max_instruments:
        print(f"Limiting to {args.max_instruments} instruments")
        instruments = instruments[:args.max_instruments]
    
    # Create output file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    populator.create_base_list_yaml(instruments, str(output_path))
    
    print("\n" + "=" * 60)
    print("[SUCCESS] COMPREHENSIVE INSTRUMENT LIST CREATED")
    print(f"Location: {output_path}")
    print(f"Total Instruments: {len(instruments)}")
    print("\nNext steps:")
    print("1. Test with: python data_download_driver.py --analysis-only")
    print("2. Download sample: python data_download_driver.py --limit 100")
    print("3. Full download: python data_download_driver.py")


if __name__ == "__main__":
    main()