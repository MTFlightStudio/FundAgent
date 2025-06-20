#!/usr/bin/env python3
"""
Test Enhanced HubSpot Data Extraction for Multiple Deals
"""

import json
import sys
from ai_agents.ui.agent_runner import AgentRunner

def test_deal_extraction(runner, deal_id, deal_name):
    """Test extraction for a single deal"""
    print(f"\n{'='*80}")
    print(f"Testing Deal: {deal_name} (ID: {deal_id})")
    print(f"{'='*80}")
    
    try:
        extracted_data = runner._extract_company_info_from_hubspot(deal_id)
        
        if extracted_data.get('company_info'):
            company_info = extracted_data['company_info']
            
            print(f"\n🏢 COMPANY: {company_info.get('company_name', 'N/A')}")
            print(f"🌐 Website: {company_info.get('website', 'N/A')}")
            print(f"🏭 Industry: {company_info.get('industry', 'N/A')}")
            print(f"📍 Location: {company_info.get('location_hq', 'N/A')}")
            print(f"🎯 Customer: {company_info.get('target_customer', 'N/A')}")
            print(f"📈 Stage: {company_info.get('funding_stage', 'N/A')}")
            
            print(f"\n💰 FINANCIAL METRICS:")
            if company_info.get('key_metrics'):
                metrics = company_info['key_metrics']
                print(f"  LTM Revenue: {metrics.get('LTM Revenue', 'N/A')}")
                print(f"  Monthly Revenue: {metrics.get('Monthly Revenue', 'N/A')}")
                print(f"  Raising Amount: {metrics.get('Current Raise Amount', 'N/A')}")
                print(f"  Valuation: {metrics.get('Valuation', 'N/A')}")
                print(f"  Prior Funding: {metrics.get('Prior Funding', 'N/A')}")
                print(f"  Team Equity: {metrics.get('Team Equity', 'N/A')}%")
                print(f"  Founders: {metrics.get('Founders', 'N/A')}")
            
            print(f"\n📋 DESCRIPTION:")
            if company_info.get('description'):
                desc = company_info['description']
                print(f"  {desc[:200]}{'...' if len(desc) > 200 else ''}")
            
            print(f"\n🚀 USP/BUSINESS MODEL:")
            if company_info.get('business_model'):
                usp = company_info['business_model']
                print(f"  {usp[:200]}{'...' if len(usp) > 200 else ''}")
            
            # Check data quality
            print(f"\n✅ DATA QUALITY CHECK:")
            metrics = company_info.get('key_metrics', {})
            quality_issues = []
            
            if not metrics.get('LTM Revenue'):
                quality_issues.append("Missing LTM Revenue")
            if not metrics.get('Valuation'):
                quality_issues.append("Missing Valuation")
            if not metrics.get('Current Raise Amount'):
                quality_issues.append("Missing Raising Amount")
            if not company_info.get('industry'):
                quality_issues.append("Missing Industry")
            
            if quality_issues:
                print(f"  ⚠️  Issues: {', '.join(quality_issues)}")
            else:
                print(f"  ✅ All key fields populated")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    """Test multiple deals"""
    runner = AgentRunner()
    
    # Test deals we've identified
    test_deals = [
        ("241939260658", "Shopthru"),
        ("240849482947", "Fermentful"), 
        ("240785272023", "Magnate"),
        ("240852995287", "Mode Sports"),
    ]
    
    print("Testing Enhanced HubSpot Data Extraction for Multiple Deals")
    print(f"Total deals to test: {len(test_deals)}")
    
    successful = 0
    failed = 0
    
    for deal_id, deal_name in test_deals:
        if test_deal_extraction(runner, deal_id, deal_name):
            successful += 1
        else:
            failed += 1
    
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Successful extractions: {successful}")
    print(f"❌ Failed extractions: {failed}")
    print(f"📊 Success rate: {successful/len(test_deals)*100:.1f}%")

if __name__ == "__main__":
    main() 