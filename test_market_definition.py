#!/usr/bin/env python3
"""
Quick test of the market definition functionality
"""

import json
from ai_agents.agents.market_intelligence_agent import _define_specific_market_from_hubspot_data

def test_with_real_bajanwheels_data():
    """Test with BajanWheels data (similar to what's in your system)"""
    
    bajanwheels_data = {
        'associated_companies': [{
            'properties': {
                'name': 'BajanWheels Ltd',
                'describe_the_business_product_in_one_sentence': 'BajanWheels is shaping Barbados\' future transit network by providing a public transit companion app that offers real-time updates, seamless payments, and a safer commute.',
                'what_is_your_usp__what_makes_you_different_from_your_competitors_': 'We provide real-time tracking and digital payments specifically designed for Caribbean public transit systems.',
                'what_sector_is_your_business_product_': 'Transportation',
                'what_best_describes_your_customer_base_': 'Commuters and travelers in Barbados seeking efficient, real-time public transportation solutions.',
                'where_is_your_business_based_': 'Bridgetown, Saint Michael, Barbados'
            }
        }]
    }
    
    print("=" * 60)
    print("TESTING WITH BAJANWHEELS DATA")
    print("=" * 60)
    
    market_def = _define_specific_market_from_hubspot_data(bajanwheels_data)
    
    print(f"📊 Original Sector: Transportation")
    print(f"🎯 Enhanced Market: {market_def['market_focus']}")
    print(f"🌍 Geographic Scope: {market_def['geographic_scope']}")
    print(f"🔍 Search Terms: {', '.join(market_def['search_terms'])}")
    print(f"💡 Reasoning: {market_def['reasoning']}")
    print()
    
    # Verify it caught the transit-specific keywords
    expected_keywords = ['transit', 'public', 'transportation']
    found_keywords = [term for term in expected_keywords if any(keyword in term.lower() for keyword in ['transit', 'public', 'transportation'])]
    
    if 'Public Transit Technology Market' in market_def['market_focus']:
        print("✅ SUCCESS: Correctly identified as Public Transit Technology Market")
    else:
        print("❌ ISSUE: Did not identify as transit technology market")
        
    if market_def['geographic_scope'] == 'Caribbean':
        print("✅ SUCCESS: Correctly identified Caribbean geographic scope")
    else:
        print("❌ ISSUE: Did not identify Caribbean scope")
    
    return market_def

def test_comparison():
    """Compare old vs new approach"""
    
    # Test data that would benefit from specific targeting
    test_cases = [
        {
            'name': 'Kefir Startup',
            'data': {
                'associated_companies': [{
                    'properties': {
                        'describe_the_business_product_in_one_sentence': 'We produce artisanal kefir with live probiotics for gut health.',
                        'what_sector_is_your_business_product_': 'Health and Wellness',
                        'where_is_your_business_based_': 'London, UK'
                    }
                }]
            },
            'expected_improvement': 'Should target kefir/probiotic market instead of broad health'
        },
        {
            'name': 'BNPL Company',
            'data': {
                'associated_companies': [{
                    'properties': {
                        'describe_the_business_product_in_one_sentence': 'Buy now pay later solution for e-commerce.',
                        'what_sector_is_your_business_product_': 'FinTech',
                        'where_is_your_business_based_': 'New York, USA'
                    }
                }]
            },
            'expected_improvement': 'Should target BNPL market instead of broad fintech'
        }
    ]
    
    print("=" * 60)
    print("OLD VS NEW APPROACH COMPARISON")
    print("=" * 60)
    
    for case in test_cases:
        print(f"\n🏢 {case['name']}")
        print("-" * 30)
        
        old_approach = case['data']['associated_companies'][0]['properties']['what_sector_is_your_business_product_']
        new_approach = _define_specific_market_from_hubspot_data(case['data'])
        
        print(f"❌ Old: '{old_approach}' market")
        print(f"✅ New: '{new_approach['market_focus']}'")
        print(f"💡 Expected: {case['expected_improvement']}")

if __name__ == "__main__":
    test_with_real_bajanwheels_data()
    test_comparison() 