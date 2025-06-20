#!/usr/bin/env python3
"""
HubSpot Data Comparison Test Script
Fetches data from multiple deals to compare with HubSpot UI manually
"""

import json
import sys
from ai_agents.services.hubspot_client import get_deal_with_associated_data

def print_separator(title):
    """Print a formatted separator with title"""
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}")

def extract_financial_metrics(company_props):
    """Extract financial metrics from company properties"""
    financial_data = {
        "ltm_revenue": company_props.get("what_is_your_ltm__last_12_months__revenue_"),
        "monthly_revenue": company_props.get("what_is_your_current_monthly_revenue_"),
        "raising_amount": company_props.get("how_much_are_you_raising_at_this_stage_"),
        "valuation": company_props.get("what_valuation_are_you_raising_at_"),
        "prior_funding": company_props.get("how_much_have_you_raised_prior_to_this_round_"),
        "team_equity": company_props.get("how_much_of_the_equity_do_you_your_team_have_"),
        "business_stage": company_props.get("what_best_describes_your_stage_of_business_"),
    }
    return financial_data

def extract_company_basics(company_props):
    """Extract basic company information"""
    return {
        "company_name": company_props.get("name"),
        "domain": company_props.get("domain"),
        "website": company_props.get("website"),
        "description": company_props.get("description"),
        "one_sentence_desc": company_props.get("describe_the_business_product_in_one_sentence"),
        "usp": company_props.get("what_is_your_usp__what_makes_you_different_from_your_competitors_"),
        "location": company_props.get("where_is_your_business_based_"),
        "sector": company_props.get("what_sector_is_your_business_product_"),
        "customer_base": company_props.get("what_best_describes_your_customer_base_"),
        "employees": company_props.get("how_many_employees_do_you_have__full_time_equivalents_"),
    }

def extract_founder_info(contact_list):
    """Extract founder information from contacts"""
    founders = []
    for contact_data in contact_list:
        contact_props = contact_data.get("properties", {})
        founders.append({
            "name": f"{contact_props.get('firstname', '')} {contact_props.get('lastname', '')}".strip(),
            "email": contact_props.get("email"),
            "linkedin": contact_props.get("hs_linkedin_url"),
            "job_title": contact_props.get("jobtitle"),
            "all_founders_list": contact_props.get("list_name_of_all_founders"),
            "other_linkedin_profiles": contact_props.get("attach_link_to_all_founders_linkedin_profiles"),
            "uk_work_rights": contact_props.get("do_you_and_the_founders_have_the_right_to_work_in_the_uk_"),
        })
    return founders

def test_deal_data_extraction(deal_id):
    """Test data extraction for a specific deal"""
    print_separator(f"TESTING DEAL ID: {deal_id}")
    
    try:
        # Fetch comprehensive deal data
        deal_data = get_deal_with_associated_data(deal_id)
        
        if not deal_data or not deal_data.get("deal"):
            print(f"❌ ERROR: Could not fetch deal data for ID {deal_id}")
            return
        
        # Extract deal information
        deal_info = deal_data["deal"]
        print(f"\n📋 DEAL INFORMATION:")
        print(f"  Deal Name: {deal_info.get('dealname', 'N/A')}")
        print(f"  Deal ID: {deal_info.get('hs_object_id', 'N/A')}")
        print(f"  Stage: {deal_info.get('deal_stage_label', 'N/A')} (ID: {deal_info.get('dealstage', 'N/A')})")
        print(f"  Amount: {deal_info.get('amount', 'N/A')}")
        print(f"  Equity Request: {deal_info.get('request', 'N/A')}")
        
        # Extract company information
        companies = deal_data.get("associated_companies", [])
        if companies:
            print(f"\n🏢 COMPANY INFORMATION ({len(companies)} company/companies):")
            for i, company_data in enumerate(companies, 1):
                company_props = company_data.get("properties", {})
                
                print(f"\n  Company {i}:")
                basics = extract_company_basics(company_props)
                for key, value in basics.items():
                    if value:  # Only show non-empty values
                        print(f"    {key.replace('_', ' ').title()}: {value}")
                
                print(f"\n  💰 FINANCIAL METRICS:")
                financials = extract_financial_metrics(company_props)
                for key, value in financials.items():
                    print(f"    {key.replace('_', ' ').title()}: {value if value else 'N/A'}")
        else:
            print(f"\n🏢 COMPANY INFORMATION: No associated companies found")
        
        # Extract founder information
        contacts = deal_data.get("associated_contacts", [])
        if contacts:
            print(f"\n👤 FOUNDER/CONTACT INFORMATION ({len(contacts)} contact/contacts):")
            founders = extract_founder_info(contacts)
            for i, founder in enumerate(founders, 1):
                print(f"\n  Contact {i}:")
                for key, value in founder.items():
                    if value:  # Only show non-empty values
                        print(f"    {key.replace('_', ' ').title()}: {value}")
        else:
            print(f"\n👤 FOUNDER/CONTACT INFORMATION: No associated contacts found")
        
        # Save raw data to file for detailed inspection
        filename = f"deal_{deal_id}_raw_data.json"
        with open(filename, 'w') as f:
            json.dump(deal_data, f, indent=2)
        print(f"\n💾 Raw data saved to: {filename}")
        
        print(f"\n✅ Successfully processed deal {deal_id}")
        
    except Exception as e:
        print(f"❌ ERROR processing deal {deal_id}: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to test multiple deals"""
    print_separator("HUBSPOT DATA COMPARISON TEST")
    print("This script will fetch data from multiple HubSpot deals for manual comparison")
    print("Please provide deal IDs to test (one per line, empty line to start testing):")
    
    deal_ids = []
    
    # Get deal IDs from user input
    while True:
        deal_id = input("Enter deal ID (or press Enter to start): ").strip()
        if not deal_id:
            break
        deal_ids.append(deal_id)
    
    if not deal_ids:
        # Default test deals if none provided
        print("No deal IDs provided. Using default test deals...")
        deal_ids = [
            "227710582988",  # Your original test deal
            # Add more default deal IDs here if you have them
        ]
    
    print(f"\nTesting {len(deal_ids)} deal(s): {', '.join(deal_ids)}")
    
    # Test each deal
    for deal_id in deal_ids:
        test_deal_data_extraction(deal_id)
    
    print_separator("TESTING COMPLETE")
    print(f"Tested {len(deal_ids)} deals. Check the generated JSON files for detailed data.")
    print("Compare the output above with what you see in HubSpot UI to identify discrepancies.")

if __name__ == "__main__":
    main() 