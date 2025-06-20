#!/usr/bin/env python3
"""
Test Enhanced Market Research with HubSpot Data

This script demonstrates the improved market research agent that uses HubSpot company data 
to determine specific, targeted markets instead of generic industry sectors.

Example: Instead of researching "health and wellness" broadly, it would research 
"Kefir and Probiotic Beverages Market" specifically.
"""

import json
import logging
from ai_agents.agents.market_intelligence_agent import _define_specific_market_from_hubspot_data

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_market_definition():
    """Test the market definition functionality with sample HubSpot data"""
    
    # Example HubSpot data for a kefir company
    kefir_company_data = {
        'associated_companies': [{
            'properties': {
                'name': 'ProBio Kefir Co.',
                'describe_the_business_product_in_one_sentence': 'We produce organic kefir and probiotic beverages for health-conscious consumers.',
                'what_is_your_usp__what_makes_you_different_from_your_competitors_': 'Our kefir contains 12 unique probiotic strains and is made from grass-fed organic milk.',
                'what_sector_is_your_business_product_': 'Health and Wellness',
                'what_best_describes_your_customer_base_': 'Health-conscious consumers aged 25-45 who prioritize gut health and organic products.',
                'where_is_your_business_based_': 'London, UK'
            }
        }]
    }
    
    # Example HubSpot data for a transit technology company (like BajanWheels)
    transit_company_data = {
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
    
    # Example HubSpot data for a FinTech BNPL company
    bnpl_company_data = {
        'associated_companies': [{
            'properties': {
                'name': 'PayLater Pro',
                'describe_the_business_product_in_one_sentence': 'We provide buy now pay later solutions for online retailers to increase conversion rates.',
                'what_is_your_usp__what_makes_you_different_from_your_competitors_': 'Our BNPL platform integrates with 50+ e-commerce platforms and offers flexible installment options.',
                'what_sector_is_your_business_product_': 'FinTech',
                'what_best_describes_your_customer_base_': 'E-commerce retailers and online shoppers seeking flexible payment options.',
                'where_is_your_business_based_': 'San Francisco, CA, USA'
            }
        }]
    }
    
    test_cases = [
        ("Kefir Company", kefir_company_data),
        ("Transit Technology Company", transit_company_data),
        ("BNPL FinTech Company", bnpl_company_data)
    ]
    
    print("=" * 80)
    print("ENHANCED MARKET RESEARCH DEMONSTRATION")
    print("=" * 80)
    print()
    
    for company_type, hubspot_data in test_cases:
        print(f"🏢 {company_type}")
        print("-" * 50)
        
        # Show original generic approach
        generic_sector = hubspot_data['associated_companies'][0]['properties']['what_sector_is_your_business_product_']
        print(f"❌ OLD APPROACH (Generic): Would research '{generic_sector}' market")
        
        # Show new specific approach
        market_def = _define_specific_market_from_hubspot_data(hubspot_data)
        print(f"✅ NEW APPROACH (Specific): Researches '{market_def['market_focus']}' market")
        print(f"   🎯 Geographic Scope: {market_def['geographic_scope']}")
        print(f"   🔍 Search Terms: {', '.join(market_def['search_terms'])}")
        print(f"   💡 Reasoning: {market_def['reasoning']}")
        print()
    
    print("=" * 80)
    print("BENEFITS OF ENHANCED APPROACH:")
    print("=" * 80)
    print("• 🎯 More targeted market data (e.g., kefir market vs health & wellness)")
    print("• 🌍 Geographic specificity (e.g., Caribbean vs Global)")
    print("• 📊 Better competitor identification (niche players vs broad industry)")
    print("• 💰 More accurate market sizing (specific segment vs entire industry)")
    print("• 📈 Relevant growth trends (product-specific vs industry-wide)")
    print("• ⚡ Faster, more relevant research with specific search terms")
    print()

def test_edge_cases():
    """Test edge cases and fallback behavior"""
    
    print("=" * 80)
    print("TESTING EDGE CASES AND FALLBACKS")
    print("=" * 80)
    print()
    
    # Test with minimal data
    minimal_data = {
        'associated_companies': [{
            'properties': {
                'name': 'Tech Startup',
                'what_sector_is_your_business_product_': 'Technology'
            }
        }]
    }
    
    print("📝 Test Case: Minimal HubSpot Data")
    market_def = _define_specific_market_from_hubspot_data(minimal_data)
    print(f"   Result: {market_def['market_focus']}")
    print(f"   Reasoning: {market_def['reasoning']}")
    print()
    
    # Test with no data
    print("📝 Test Case: No HubSpot Data")
    market_def = _define_specific_market_from_hubspot_data(None)
    print(f"   Result: {market_def['market_focus']}")
    print(f"   Reasoning: {market_def['reasoning']}")
    print()
    
    # Test with rich description but no keyword matches
    unique_business_data = {
        'associated_companies': [{
            'properties': {
                'name': 'Quantum Flux Innovators',
                'describe_the_business_product_in_one_sentence': 'We develop quantum entanglement solutions for industrial manufacturing optimization.',
                'what_sector_is_your_business_product_': 'DeepTech',
                'where_is_your_business_based_': 'Cambridge, UK'
            }
        }]
    }
    
    print("📝 Test Case: Unique Business (No Pattern Match)")
    market_def = _define_specific_market_from_hubspot_data(unique_business_data)
    print(f"   Result: {market_def['market_focus']}")
    print(f"   Geographic: {market_def['geographic_scope']}")
    print(f"   Reasoning: {market_def['reasoning']}")
    print()

if __name__ == "__main__":
    logger.info("Starting Enhanced Market Research Demonstration...")
    
    test_market_definition()
    test_edge_cases()
    
    logger.info("Demonstration completed!")
    
    print("=" * 80)
    print("HOW TO USE ENHANCED MARKET RESEARCH:")
    print("=" * 80)
    print("1. The system automatically uses HubSpot data when available")
    print("2. For CLI usage with HubSpot data:")
    print("   python -m ai_agents.agents.market_intelligence_agent \\")
    print("     --sector 'Health and Wellness' \\")
    print("     --hubspot-data path/to/hubspot_deal_data.json")
    print("3. The orchestrator automatically passes HubSpot data for enhanced research")
    print("4. Results will be much more specific and targeted to the actual business")
    print() 