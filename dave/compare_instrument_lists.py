#!/usr/bin/env python3
"""
Compare Instrument Lists

Compares instruments across the three instrument lists:
- base-list-all.yml (5000+ instruments)
- favorites_instruments.yml 
- popular_instruments.yml

Shows instruments that are in favorites/popular but missing from base list.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from consuela.config.instrument_list_loader import (
    load_base_list_all, load_favorites_instruments, load_popular_instruments
)


def extract_symbols(instruments):
    """Extract symbols from instrument list"""
    return set(inst.get('symbol', '') for inst in instruments if inst.get('symbol'))


def compare_lists():
    """Compare all three instrument lists"""
    print("INSTRUMENT LISTS COMPARISON")
    print("=" * 60)
    
    # Load all lists
    print("Loading instrument lists...")
    base_instruments = load_base_list_all()
    favorites_instruments = load_favorites_instruments()
    popular_instruments = load_popular_instruments()
    
    # Extract symbols
    base_symbols = extract_symbols(base_instruments)
    favorites_symbols = extract_symbols(favorites_instruments)
    popular_symbols = extract_symbols(popular_instruments)
    
    print(f"Base list: {len(base_symbols)} symbols")
    print(f"Favorites list: {len(favorites_symbols)} symbols")
    print(f"Popular list: {len(popular_symbols)} symbols")
    
    # Find missing instruments
    favorites_missing = favorites_symbols - base_symbols
    popular_missing = popular_symbols - base_symbols
    
    print(f"\n" + "=" * 60)
    print("MISSING FROM BASE LIST")
    print("=" * 60)
    
    # Show favorites missing from base
    if favorites_missing:
        print(f"\nFAVORITES missing from base list ({len(favorites_missing)} symbols):")
        print("-" * 50)
        
        # Get full details for missing favorites
        favorites_details = {inst.get('symbol'): inst for inst in favorites_instruments}
        
        for symbol in sorted(favorites_missing):
            if symbol in favorites_details:
                inst = favorites_details[symbol]
                name = inst.get('name', 'N/A')
                sector = inst.get('sector', 'N/A')
                exchange = inst.get('exchange', 'N/A')
                print(f"  {symbol:8} | {name[:40]:<40} | {sector[:15]:<15} | {exchange}")
            else:
                print(f"  {symbol:8} | (Details not found)")
    else:
        print(f"\n[OK] All favorites instruments are in base list")
    
    # Show popular missing from base  
    if popular_missing:
        print(f"\nPOPULAR missing from base list ({len(popular_missing)} symbols):")
        print("-" * 50)
        
        # Get full details for missing popular
        popular_details = {inst.get('symbol'): inst for inst in popular_instruments}
        
        for symbol in sorted(popular_missing):
            if symbol in popular_details:
                inst = popular_details[symbol]
                name = inst.get('name', 'N/A')
                sector = inst.get('sector', 'N/A')
                exchange = inst.get('exchange', 'N/A')
                print(f"  {symbol:8} | {name[:40]:<40} | {sector[:15]:<15} | {exchange}")
            else:
                print(f"  {symbol:8} | (Details not found)")
    else:
        print(f"\n[OK] All popular instruments are in base list")
    
    # Combined missing
    all_missing = favorites_missing | popular_missing
    
    if all_missing:
        print(f"\n" + "=" * 60)
        print(f"SUMMARY: {len(all_missing)} unique instruments missing from base list")
        print("=" * 60)
        
        print(f"\nMissing symbols list:")
        missing_list = sorted(list(all_missing))
        
        # Print in columns
        for i in range(0, len(missing_list), 8):
            row = missing_list[i:i+8]
            print("  " + "  ".join(f"{symbol:<8}" for symbol in row))
        
        # Show overlap
        overlap_missing = favorites_missing & popular_missing
        if overlap_missing:
            print(f"\nSymbols missing from BOTH favorites and popular ({len(overlap_missing)}):")
            overlap_list = sorted(list(overlap_missing))
            for i in range(0, len(overlap_list), 8):
                row = overlap_list[i:i+8]
                print("  " + "  ".join(f"{symbol:<8}" for symbol in row))
    else:
        print(f"\n[SUCCESS] All favorites and popular instruments are included in base list!")
    
    # Show coverage statistics
    print(f"\n" + "=" * 60)
    print("COVERAGE STATISTICS")
    print("=" * 60)
    
    favorites_coverage = len(favorites_symbols - favorites_missing) / len(favorites_symbols) * 100 if favorites_symbols else 0
    popular_coverage = len(popular_symbols - popular_missing) / len(popular_symbols) * 100 if popular_symbols else 0
    
    print(f"Favorites coverage:  {favorites_coverage:.1f}% ({len(favorites_symbols - favorites_missing)}/{len(favorites_symbols)})")
    print(f"Popular coverage:    {popular_coverage:.1f}% ({len(popular_symbols - popular_missing)}/{len(popular_symbols)})")
    
    # Show what's in base but not in favorites/popular (preview)
    base_only = base_symbols - favorites_symbols - popular_symbols
    print(f"Base-only symbols:   {len(base_only)} (not in favorites or popular)")
    
    if base_only:
        print(f"\nSample of base-only symbols (first 20):")
        sample = sorted(list(base_only))[:20]
        for i in range(0, len(sample), 10):
            row = sample[i:i+10]
            print("  " + "  ".join(f"{symbol:<6}" for symbol in row))
    
    return {
        'favorites_missing': favorites_missing,
        'popular_missing': popular_missing,
        'all_missing': all_missing,
        'base_symbols': base_symbols,
        'favorites_symbols': favorites_symbols,
        'popular_symbols': popular_symbols
    }


def create_missing_instruments_file(comparison_result):
    """Create a YAML file with missing instruments for easy addition"""
    
    all_missing = comparison_result['all_missing']
    
    if not all_missing:
        print("\nNo missing instruments to create file for.")
        return
    
    output_file = Path("missing_instruments.yml")
    
    print(f"\nCreating {output_file} with missing instruments...")
    
    # Load details from favorites and popular
    favorites_instruments = load_favorites_instruments()
    popular_instruments = load_popular_instruments()
    
    favorites_details = {inst.get('symbol'): inst for inst in favorites_instruments}
    popular_details = {inst.get('symbol'): inst for inst in popular_instruments}
    
    # Create YAML content
    yaml_content = []
    yaml_content.append("# Missing Instruments from Favorites and Popular Lists")
    yaml_content.append("# These instruments should be added to base-list-all.yml")
    yaml_content.append("")
    yaml_content.append("missing_instruments:")
    
    for symbol in sorted(all_missing):
        # Try to get details from favorites first, then popular
        inst = favorites_details.get(symbol) or popular_details.get(symbol)
        
        if inst:
            yaml_content.append(f"  - symbol: \"{symbol}\"")
            yaml_content.append(f"    name: \"{inst.get('name', 'N/A')}\"")
            yaml_content.append(f"    sector: \"{inst.get('sector', 'Unknown')}\"")
            yaml_content.append(f"    market_cap_category: \"{inst.get('market_cap_category', 'large_cap')}\"")
            yaml_content.append(f"    exchange: \"{inst.get('exchange', 'NYSE')}\"")
            yaml_content.append(f"    status: \"active\"")
            
            # Add source info
            source_list = []
            if symbol in favorites_details:
                source_list.append("favorites")
            if symbol in popular_details:
                source_list.append("popular")
            yaml_content.append(f"    source_lists: {source_list}")
            yaml_content.append("")
        else:
            yaml_content.append(f"  - symbol: \"{symbol}\"")
            yaml_content.append(f"    name: \"Unknown\"")
            yaml_content.append(f"    sector: \"Unknown\"")
            yaml_content.append(f"    market_cap_category: \"large_cap\"")
            yaml_content.append(f"    exchange: \"NYSE\"")
            yaml_content.append(f"    status: \"active\"")
            yaml_content.append(f"    note: \"Details not found in source lists\"")
            yaml_content.append("")
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(yaml_content))
    
    print(f"[OK] Created {output_file} with {len(all_missing)} missing instruments")
    print(f"     You can review and add these to base-list-all.yml if desired")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare instrument lists")
    parser.add_argument("--create-missing-file", action="store_true", 
                       help="Create YAML file with missing instruments")
    
    args = parser.parse_args()
    
    # Run comparison
    result = compare_lists()
    
    # Create missing file if requested
    if args.create_missing_file:
        create_missing_instruments_file(result)
    
    print(f"\nDone! Use --create-missing-file to generate a file with missing instruments.")


if __name__ == "__main__":
    main()