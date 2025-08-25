#!/usr/bin/env python3
"""
Add Missing Instruments to Base List

Adds instruments that are in favorites/popular lists but missing from base-list-all.yml.
This ensures all curated instruments are included in the comprehensive database.
"""

import yaml
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from consuela.config.instrument_list_loader import (
    load_base_list_all, load_favorites_instruments, load_popular_instruments
)


def load_missing_instruments():
    """Load missing instruments from the generated file"""
    missing_file = Path("missing_instruments.yml")
    
    if not missing_file.exists():
        print("missing_instruments.yml not found. Run compare_instrument_lists.py --create-missing-file first.")
        return []
    
    with open(missing_file, 'r') as f:
        data = yaml.safe_load(f)
    
    return data.get('missing_instruments', [])


def add_missing_to_base_list():
    """Add missing instruments to base-list-all.yml"""
    print("ADDING MISSING INSTRUMENTS TO BASE LIST")
    print("=" * 60)
    
    # Load current base list
    base_path = Path("../consuela/config/instrument-list/base-list-all.yml")
    
    with open(base_path, 'r') as f:
        base_data = yaml.safe_load(f)
    
    current_instruments = base_data.get('instruments', [])
    current_symbols = {inst.get('symbol') for inst in current_instruments}
    
    print(f"Current base list: {len(current_instruments)} instruments")
    
    # Load missing instruments
    missing_instruments = load_missing_instruments()
    
    if not missing_instruments:
        print("No missing instruments found.")
        return
    
    print(f"Missing instruments: {len(missing_instruments)}")
    
    # Filter to only truly missing ones (in case base list was updated)
    truly_missing = []
    for inst in missing_instruments:
        if inst.get('symbol') not in current_symbols:
            truly_missing.append(inst)
    
    if not truly_missing:
        print("[OK] All missing instruments are already in base list!")
        return
    
    print(f"Truly missing: {len(truly_missing)}")
    print("\nAdding instruments:")
    print("-" * 40)
    
    # Add missing instruments
    added_count = 0
    for inst in truly_missing:
        symbol = inst.get('symbol')
        name = inst.get('name', 'N/A')
        
        # Clean up the instrument data
        clean_inst = {
            'symbol': symbol,
            'name': name,
            'sector': inst.get('sector', 'ETF' if 'ETF' in name or 'Fund' in name else 'Unknown'),
            'market_cap_category': 'etf' if ('ETF' in name or 'Fund' in name) else inst.get('market_cap_category', 'large_cap'),
            'exchange': inst.get('exchange', 'NYSE'),
            'status': 'active',
            'source': 'popular_list_addition'
        }
        
        # Add category for ETFs
        if 'ETF' in name or 'Fund' in name:
            if 'Bond' in name or 'Treasury' in name:
                clean_inst['category'] = 'Bond ETF'
            elif 'Gold' in name or 'Silver' in name or 'Commodity' in name:
                clean_inst['category'] = 'Commodity ETF'
            elif 'Real Estate' in name or 'REIT' in name:
                clean_inst['category'] = 'Real Estate ETF'
            elif 'S&P 500' in name:
                clean_inst['category'] = 'Large Cap Blend'
            elif 'Russell' in name:
                clean_inst['category'] = 'Small Cap Blend' if '2000' in name else 'Large Cap Blend'
            else:
                clean_inst['category'] = 'Sector ETF'
        
        current_instruments.append(clean_inst)
        added_count += 1
        
        print(f"  + {symbol:8} | {name[:50]:<50}")
    
    # Update metadata
    base_data['instruments'] = current_instruments
    base_data['total_count'] = len(current_instruments)
    
    if 'metadata' in base_data:
        base_data['metadata']['total_instruments'] = len(current_instruments)
        base_data['metadata']['active_count'] = len([i for i in current_instruments if i.get('status') == 'active'])
        
        # Update categories
        etf_count = len([i for i in current_instruments if i.get('market_cap_category') == 'etf' or 'ETF' in i.get('name', '')])
        base_data['metadata']['etf_count'] = etf_count
        
        if 'categories' in base_data['metadata']:
            base_data['metadata']['categories']['etfs'] = etf_count
    
    # Backup original file
    backup_path = base_path.with_suffix('.yml.backup')
    import shutil
    shutil.copy2(base_path, backup_path)
    print(f"\nBackup created: {backup_path}")
    
    # Write updated file
    with open(base_path, 'w') as f:
        yaml.dump(base_data, f, default_flow_style=False, indent=2, allow_unicode=True)
    
    print(f"\n[SUCCESS] Added {added_count} instruments to {base_path}")
    print(f"Total instruments now: {len(current_instruments)}")
    
    return len(current_instruments)


def verify_addition():
    """Verify that instruments were added correctly"""
    print(f"\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    # Re-run comparison
    base_instruments = load_base_list_all()
    popular_instruments = load_popular_instruments()
    
    base_symbols = {inst.get('symbol') for inst in base_instruments}
    popular_symbols = {inst.get('symbol') for inst in popular_instruments}
    
    still_missing = popular_symbols - base_symbols
    
    print(f"Base list now: {len(base_symbols)} symbols")
    print(f"Popular list: {len(popular_symbols)} symbols")
    print(f"Still missing: {len(still_missing)} symbols")
    
    if still_missing:
        print(f"Still missing from base: {sorted(list(still_missing))}")
    else:
        print("[SUCCESS] All popular instruments now in base list!")
    
    # Show some key ETFs that should now be included
    key_etfs = ['SPY', 'VOO', 'QQQ', 'IWM', 'GLD', 'SLV', 'AGG', 'TLT', 'XLK', 'XLF']
    print(f"\nKey ETFs status:")
    for etf in key_etfs:
        status = "[OK]" if etf in base_symbols else "[MISSING]"
        print(f"  {etf}: {status}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Add missing instruments to base list")
    parser.add_argument("--verify-only", action="store_true", help="Only verify, don't add")
    
    args = parser.parse_args()
    
    if args.verify_only:
        verify_addition()
    else:
        # Add missing instruments
        total_count = add_missing_to_base_list()
        
        if total_count:
            # Verify addition worked
            verify_addition()
            
            print(f"\n" + "=" * 60)
            print("NEXT STEPS")
            print("=" * 60)
            print("1. Test with: python data_download_driver.py --analysis-only")
            print("2. Download key ETFs: python data_download_driver.py --limit 50")
            print("3. Full download: python data_download_driver.py")


if __name__ == "__main__":
    main()