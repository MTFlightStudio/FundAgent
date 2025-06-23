#!/usr/bin/env python3
"""
Pipeline Summary Analysis Script
Analyzes the results from the batch deal processing and provides key statistics
"""

import json
import sys
from datetime import datetime

def load_and_analyze_pipeline_data(filename):
    """Load and analyze pipeline data from JSON results file"""
    
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        analysis = data['pipeline_analysis']['analysis']
        stages = data['pipeline_analysis']['stages']
        
        print('🎯 HUBSPOT DEALS PIPELINE SUMMARY')
        print('=' * 60)
        print(f'📊 Total Deals Analyzed: {analysis["total_deals_analyzed"]:,}')
        print(f'💰 Total Pipeline Value: ${analysis["total_pipeline_value"]:,.2f}')
        print(f'🏢 Active Stages: {analysis["stages_with_deals"]} out of {len(stages)}')
        print(f'⏰ Analysis Date: {analysis["analysis_timestamp"][:10]}')
        print()
        
        print('📈 DEALS BY STAGE (Ranked by Volume):')
        print('-' * 80)
        print(f'{"Stage Name":30} | {"Deals":5} | {"Percentage":10} | {"Total Value":15} | {"Avg Deal Size":12}')
        print('-' * 80)
        
        # Sort stages by deal count
        stages_sorted = sorted(stages.items(), key=lambda x: x[1]['deal_count'], reverse=True)
        
        for stage_id, info in stages_sorted:
            count = info['deal_count']
            if count > 0:
                # Calculate stage value
                stage_value = sum(float(deal.get('amount', 0) or 0) for deal in info['deals'])
                percentage = (count / analysis['total_deals_analyzed']) * 100
                avg_deal_size = stage_value / count if count > 0 else 0
                
                stage_name = info['label'][:28]  # Truncate long names
                print(f'{stage_name:30} | {count:5d} | {percentage:8.1f}% | ${stage_value:13,.0f} | ${avg_deal_size:10,.0f}')
        
        print()
        print('🔍 KEY INSIGHTS:')
        print('-' * 30)
        
        # Calculate key metrics
        rejected_emailed = stages['1457423574']['deal_count']
        rejected_lost = stages['closedlost']['deal_count']
        total_rejected = rejected_emailed + rejected_lost
        
        screening_calls = stages['presentationscheduled']['deal_count']
        sb_kill_continue = stages['1245906141']['deal_count']
        investment_memo = stages['decisionmakerboughtin']['deal_count']
        accepted = stages['closedwon']['deal_count']
        
        active_pipeline = analysis['total_deals_analyzed'] - total_rejected
        
        print(f'• Total Rejected: {total_rejected:,} ({(total_rejected/analysis["total_deals_analyzed"])*100:.1f}%)')
        print(f'  - Rejected Emailed: {rejected_emailed:,}')
        print(f'  - Rejected Lost: {rejected_lost:,}')
        print()
        print(f'• Active Pipeline: {active_pipeline:,} ({(active_pipeline/analysis["total_deals_analyzed"])*100:.1f}%)')
        print(f'  - Screening Calls: {screening_calls:,}')
        print(f'  - SB Kill/Continue: {sb_kill_continue:,}')
        print(f'  - Investment Memo: {investment_memo:,}')
        print()
        print(f'• Accepted Deals: {accepted:,}')
        print(f'• Average Deal Size: ${analysis["total_pipeline_value"]/analysis["total_deals_analyzed"]:,.0f}')
        
        # Calculate conversion rates
        if total_rejected + active_pipeline > 0:
            conversion_to_screening = (screening_calls / (total_rejected + active_pipeline)) * 100
            print(f'• Conversion to Screening: {conversion_to_screening:.2f}%')
        
        if screening_calls > 0:
            conversion_to_memo = (investment_memo / screening_calls) * 100
            print(f'• Screening to Memo: {conversion_to_memo:.2f}%')
        
        if investment_memo > 0:
            conversion_to_accepted = (accepted / investment_memo) * 100
            print(f'• Memo to Accepted: {conversion_to_accepted:.2f}%')
        
        print()
        print('💼 DEAL SIZE ANALYSIS:')
        print('-' * 25)
        
        # Analyze deal sizes across all deals
        all_deal_amounts = []
        for stage_id, stage_info in stages.items():
            for deal in stage_info.get('deals', []):
                amount = deal.get('amount')
                if amount and amount != 'null':
                    try:
                        # Handle different amount formats
                        amount_str = str(amount).replace(',', '').replace('.000', '')
                        if amount_str and amount_str != '0':
                            all_deal_amounts.append(float(amount_str))
                    except (ValueError, TypeError):
                        continue
        
        if all_deal_amounts:
            all_deal_amounts.sort()
            
            print(f'• Deals with Amount Data: {len(all_deal_amounts):,} out of {analysis["total_deals_analyzed"]:,}')
            print(f'• Smallest Deal: ${min(all_deal_amounts):,.0f}')
            print(f'• Largest Deal: ${max(all_deal_amounts):,.0f}')
            print(f'• Median Deal: ${all_deal_amounts[len(all_deal_amounts)//2]:,.0f}')
            
            # Deal size ranges
            under_100k = len([x for x in all_deal_amounts if x < 100000])
            between_100k_500k = len([x for x in all_deal_amounts if 100000 <= x < 500000])
            between_500k_1m = len([x for x in all_deal_amounts if 500000 <= x < 1000000])
            over_1m = len([x for x in all_deal_amounts if x >= 1000000])
            
            print()
            print('Deal Size Distribution:')
            print(f'  < £100k:       {under_100k:3d} deals ({(under_100k/len(all_deal_amounts))*100:.1f}%)')
            print(f'  £100k-£500k:  {between_100k_500k:3d} deals ({(between_100k_500k/len(all_deal_amounts))*100:.1f}%)')
            print(f'  £500k-£1M:    {between_500k_1m:3d} deals ({(between_500k_1m/len(all_deal_amounts))*100:.1f}%)')
            print(f'  > £1M:        {over_1m:3d} deals ({(over_1m/len(all_deal_amounts))*100:.1f}%)')
        
        return True
        
    except FileNotFoundError:
        print(f"❌ Error: File '{filename}' not found")
        return False
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON in file '{filename}'")
        return False
    except KeyError as e:
        print(f"❌ Error: Missing expected data in file: {e}")
        return False
    except Exception as e:
        print(f"❌ Error analyzing data: {str(e)}")
        return False

def main():
    """Main function"""
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        # Default to the latest file
        filename = 'multiple_deals_paginated_results_20250623_112446.json'
    
    print(f"📁 Analyzing pipeline data from: {filename}")
    print()
    
    success = load_and_analyze_pipeline_data(filename)
    
    if not success:
        print("\n💡 Available commands:")
        print("  python pipeline_summary.py                           # Use default file")
        print("  python pipeline_summary.py results_filename.json     # Use specific file")
        sys.exit(1)

if __name__ == "__main__":
    main() 