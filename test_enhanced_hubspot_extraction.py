#!/usr/bin/env python3
"""
Test Enhanced HubSpot Data Extraction
Tests the updated _extract_company_info_from_hubspot function
"""

import json
import sys
from ai_agents.ui.agent_runner import AgentRunner

def test_enhanced_extraction():
    """Test the enhanced HubSpot data extraction"""
    runner = AgentRunner()
    
    # Test deal ID from our previous test
    test_deal_id = "241939260658"  # Shopthru
    
    print("=" * 80)
    print(f"Testing Enhanced HubSpot Data Extraction for Deal: {test_deal_id}")
    print("=" * 80)
    
    try:
        # Test the enhanced extraction
        extracted_data = runner._extract_company_info_from_hubspot(test_deal_id)
        
        print(f"\n📊 EXTRACTED DATA STRUCTURE:")
        print(f"Company Name: {extracted_data.get('company_name')}")
        print(f"Industry: {extracted_data.get('industry')}")
        print(f"Founders: {extracted_data.get('founders')}")
        print(f"LinkedIn URLs: {extracted_data.get('linkedin_urls')}")
        
        if extracted_data.get('company_info'):
            company_info = extracted_data['company_info']
            
            print(f"\n🏢 COMPANY INFO:")
            for key, value in company_info.items():
                if key != 'key_metrics' and value:
                    print(f"  {key}: {value}")
            
            print(f"\n💰 KEY METRICS:")
            if company_info.get('key_metrics'):
                for metric, value in company_info['key_metrics'].items():
                    print(f"  {metric}: {value}")
            else:
                print("  No key metrics found")
        
        # Save the extracted data for inspection
        with open("enhanced_extraction_test.json", "w") as f:
            json.dump(extracted_data, f, indent=2)
        
        print(f"\n✅ Test completed successfully!")
        print(f"📁 Full data saved to: enhanced_extraction_test.json")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_extraction() 