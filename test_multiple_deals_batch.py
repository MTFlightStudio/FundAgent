#!/usr/bin/env python3
"""
Multiple Deals Batch Processing Test Script
Fetches multiple deal IDs from HubSpot with their stages and associated data
for exploring batch processing capabilities
"""

import json
import sys
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import argparse

# Import HubSpot client functions
from ai_agents.services.hubspot_client import (
    list_objects,
    get_deal_with_associated_data,
    get_pipeline_stages,
    HUBSPOT_ACCESS_TOKEN
)

def print_separator(title: str, char: str = "=", width: int = 80):
    """Print a formatted separator with title"""
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}")

def print_status(message: str, status: str = "info"):
    """Print a status message with colored formatting"""
    colors = {
        "info": "ℹ️",
        "success": "✅", 
        "warning": "⚠️",
        "error": "❌"
    }
    print(f"{colors.get(status, 'ℹ️')} {message}")

def get_all_deals_from_pipeline(
    pipeline_id: str = "default",
    limit: int = 100,
    filter_by_stage: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Fetch all deals from a specific pipeline
    
    Args:
        pipeline_id: HubSpot pipeline ID (default: "default")
        limit: Maximum number of deals to fetch
        filter_by_stage: Optional list of stage IDs to filter by
    
    Returns:
        List of deal objects with basic properties
    """
    print_status(f"Fetching deals from pipeline: {pipeline_id} (limit: {limit})")
    
    # Define properties to fetch for deals
    deal_properties = [
        "dealname", "amount", "dealstage", "pipeline", "closedate",
        "hubspot_owner_id", "hs_object_id", "createdate", "hs_lastmodifieddate",
        "request", "dealtype", "hs_deal_stage_probability"
    ]
    
    try:
        # Fetch deals using the list_objects function
        response = list_objects(
            object_type="deals",
            properties=deal_properties,
            limit=limit
        )
        
        if not response or "results" not in response:
            print_status("No deals found or invalid response", "warning")
            return []
        
        deals = response["results"]
        print_status(f"Retrieved {len(deals)} deals from API", "success")
        
        # Filter by pipeline if not default
        if pipeline_id != "default":
            deals = [deal for deal in deals if deal.get("properties", {}).get("pipeline") == pipeline_id]
            print_status(f"Filtered to {len(deals)} deals in pipeline {pipeline_id}", "info")
        
        # Filter by stage if specified
        if filter_by_stage:
            original_count = len(deals)
            deals = [deal for deal in deals if deal.get("properties", {}).get("dealstage") in filter_by_stage]
            print_status(f"Filtered to {len(deals)} deals in specified stages (was {original_count})", "info")
        
        return deals
        
    except Exception as e:
        print_status(f"Error fetching deals: {str(e)}", "error")
        return []

def get_deal_stages_summary(pipeline_id: str = "default") -> Dict[str, Any]:
    """
    Get all stages for a pipeline with deal counts
    
    Args:
        pipeline_id: HubSpot pipeline ID
    
    Returns:
        Dictionary with stage information and counts
    """
    print_status(f"Fetching pipeline stages for: {pipeline_id}")
    
    try:
        stages = get_pipeline_stages(pipeline_id, "deals")
        
        if not stages:
            print_status("No stages found", "warning")
            return {}
        
        # Create stage summary
        stage_summary = {
            "pipeline_id": pipeline_id,
            "total_stages": len(stages),
            "stages": {}
        }
        
        for stage in stages:
            stage_id = stage.get("id")
            stage_label = stage.get("label", "Unknown")
            stage_summary["stages"][stage_id] = {
                "label": stage_label,
                "id": stage_id,
                "deal_count": 0  # Will be updated when we count deals
            }
        
        print_status(f"Found {len(stages)} stages in pipeline", "success")
        return stage_summary
        
    except Exception as e:
        print_status(f"Error fetching pipeline stages: {str(e)}", "error")
        return {}

def analyze_deals_by_stage(deals: List[Dict[str, Any]], stage_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze deals and group them by stage
    
    Args:
        deals: List of deal objects
        stage_summary: Stage information from get_deal_stages_summary
    
    Returns:
        Updated stage summary with deal counts and deal lists
    """
    print_status(f"Analyzing {len(deals)} deals by stage")
    
    # Initialize stage deal lists
    for stage_id in stage_summary.get("stages", {}):
        stage_summary["stages"][stage_id]["deals"] = []
    
    # Group deals by stage
    for deal in deals:
        deal_props = deal.get("properties", {})
        deal_stage = deal_props.get("dealstage")
        deal_name = deal_props.get("dealname", "Unnamed Deal")
        deal_id = deal_props.get("hs_object_id")
        deal_amount = deal_props.get("amount")
        
        if deal_stage and deal_stage in stage_summary.get("stages", {}):
            stage_summary["stages"][deal_stage]["deals"].append({
                "id": deal_id,
                "name": deal_name,
                "amount": deal_amount,
                "stage_id": deal_stage,
                "properties": deal_props
            })
            stage_summary["stages"][deal_stage]["deal_count"] += 1
    
    # Add summary statistics
    total_deals = len(deals)
    total_amount = sum(
        float(deal.get("properties", {}).get("amount", 0) or 0)
        for deal in deals
    )
    
    stage_summary["analysis"] = {
        "total_deals_analyzed": total_deals,
        "total_pipeline_value": total_amount,
        "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
        "stages_with_deals": len([s for s in stage_summary.get("stages", {}).values() if s["deal_count"] > 0])
    }
    
    print_status(f"Analysis complete: {total_deals} deals, ${total_amount:,.2f} total value", "success")
    return stage_summary

def get_comprehensive_deal_data(deal_id: str) -> Optional[Dict[str, Any]]:
    """
    Get comprehensive data for a single deal including all associations
    
    Args:
        deal_id: HubSpot deal ID
    
    Returns:
        Comprehensive deal data or None if failed
    """
    try:
        print_status(f"Fetching comprehensive data for deal: {deal_id}")
        comprehensive_data = get_deal_with_associated_data(deal_id)
        
        if comprehensive_data and comprehensive_data.get("deal"):
            deal_name = comprehensive_data["deal"].get("dealname", "Unknown")
            company_count = len(comprehensive_data.get("associated_companies", []))
            contact_count = len(comprehensive_data.get("associated_contacts", []))
            
            print_status(f"  Deal: {deal_name} | Companies: {company_count} | Contacts: {contact_count}", "info")
            return comprehensive_data
        else:
            print_status(f"  Failed to get data for deal {deal_id}", "warning")
            return None
            
    except Exception as e:
        print_status(f"  Error fetching deal {deal_id}: {str(e)}", "error")
        return None

def batch_process_deals(
    deal_ids: List[str],
    include_comprehensive_data: bool = False
) -> Dict[str, Any]:
    """
    Process multiple deals in batch
    
    Args:
        deal_ids: List of deal IDs to process
        include_comprehensive_data: Whether to fetch full associated data
    
    Returns:
        Batch processing results
    """
    print_separator(f"BATCH PROCESSING {len(deal_ids)} DEALS")
    
    results = {
        "batch_id": f"batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        "total_deals": len(deal_ids),
        "processed_deals": 0,
        "successful_deals": 0,
        "failed_deals": 0,
        "deals": {},
        "errors": [],
        "processing_start": datetime.now(timezone.utc).isoformat()
    }
    
    for i, deal_id in enumerate(deal_ids, 1):
        print_status(f"Processing deal {i}/{len(deal_ids)}: {deal_id}")
        
        try:
            if include_comprehensive_data:
                deal_data = get_comprehensive_deal_data(deal_id)
            else:
                # Just get basic deal data
                deal_data = {"deal_id": deal_id, "status": "basic_fetch_only"}
            
            if deal_data:
                results["deals"][deal_id] = deal_data
                results["successful_deals"] += 1
            else:
                results["failed_deals"] += 1
                results["errors"].append(f"Failed to fetch data for deal {deal_id}")
            
            results["processed_deals"] += 1
            
        except Exception as e:
            results["failed_deals"] += 1
            error_msg = f"Error processing deal {deal_id}: {str(e)}"
            results["errors"].append(error_msg)
            print_status(f"  {error_msg}", "error")
    
    results["processing_end"] = datetime.now(timezone.utc).isoformat()
    
    print_separator("BATCH PROCESSING COMPLETE")
    print_status(f"Processed: {results['processed_deals']}/{results['total_deals']}", "info")
    print_status(f"Successful: {results['successful_deals']}", "success")
    print_status(f"Failed: {results['failed_deals']}", "error" if results['failed_deals'] > 0 else "info")
    
    return results

def main():
    """Main function for the script"""
    parser = argparse.ArgumentParser(description="HubSpot Multiple Deals Batch Processing Test")
    parser.add_argument("--pipeline", "-p", default="default", help="Pipeline ID to fetch deals from")
    parser.add_argument("--limit", "-l", type=int, default=20, help="Maximum number of deals to fetch")
    parser.add_argument("--stages", "-s", nargs="+", help="Filter by specific stage IDs")
    parser.add_argument("--comprehensive", "-c", action="store_true", help="Fetch comprehensive data for each deal")
    parser.add_argument("--deal-ids", "-d", nargs="+", help="Specific deal IDs to process (skips discovery)")
    parser.add_argument("--output", "-o", help="Output filename for results")
    
    args = parser.parse_args()
    
    print_separator("HUBSPOT MULTIPLE DEALS BATCH PROCESSING TEST")
    
    # Check HubSpot connection
    if not HUBSPOT_ACCESS_TOKEN:
        print_status("HUBSPOT_ACCESS_TOKEN not found. Cannot proceed.", "error")
        return False
    
    print_status("HubSpot access token found", "success")
    
    try:
        if args.deal_ids:
            # Process specific deal IDs
            print_status(f"Processing specific deal IDs: {args.deal_ids}")
            deal_ids = args.deal_ids
            
            # Batch process the specified deals
            results = batch_process_deals(deal_ids, args.comprehensive)
            
        else:
            # Discover deals from pipeline
            print_separator("STEP 1: DISCOVER DEALS FROM PIPELINE")
            
            # Get pipeline stages
            stage_summary = get_deal_stages_summary(args.pipeline)
            if not stage_summary:
                print_status("Could not fetch pipeline stages", "error")
                return False
            
            # Get deals from pipeline
            deals = get_all_deals_from_pipeline(
                pipeline_id=args.pipeline,
                limit=args.limit,
                filter_by_stage=args.stages
            )
            
            if not deals:
                print_status("No deals found", "warning")
                return False
            
            print_separator("STEP 2: ANALYZE DEALS BY STAGE")
            
            # Analyze deals by stage
            analysis = analyze_deals_by_stage(deals, stage_summary)
            
            # Print stage summary
            print_status("PIPELINE STAGE SUMMARY:", "info")
            for stage_id, stage_info in analysis.get("stages", {}).items():
                stage_label = stage_info["label"]
                deal_count = stage_info["deal_count"]
                print(f"  📊 {stage_label} ({stage_id}): {deal_count} deals")
                
                # Show first few deals in each stage
                for deal in stage_info["deals"][:3]:  # Show first 3 deals
                    deal_name = deal["name"]
                    deal_amount = deal.get("amount", "No amount")
                    print(f"    - {deal_name} (${deal_amount})")
                
                if len(stage_info["deals"]) > 3:
                    print(f"    ... and {len(stage_info['deals']) - 3} more deals")
            
            # Extract deal IDs for batch processing
            deal_ids = [deal.get("properties", {}).get("hs_object_id") for deal in deals]
            deal_ids = [deal_id for deal_id in deal_ids if deal_id]  # Remove None values
            
            print_separator("STEP 3: BATCH PROCESS DEALS")
            
            if args.comprehensive:
                print_status("Comprehensive mode: Fetching full associated data for each deal")
            else:
                print_status("Basic mode: Quick processing without full associated data")
            
            # Batch process the discovered deals
            results = batch_process_deals(deal_ids, args.comprehensive)
            
            # Add analysis to results
            results["pipeline_analysis"] = analysis
        
        # Save results to file
        output_filename = args.output or f"multiple_deals_batch_results_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print_separator("RESULTS SAVED")
        print_status(f"Results saved to: {output_filename}", "success")
        print_status(f"Total deals processed: {results.get('successful_deals', 0)}", "info")
        
        if results.get("errors"):
            print_status(f"Errors encountered: {len(results['errors'])}", "warning")
            for error in results["errors"][:5]:  # Show first 5 errors
                print(f"  ⚠️ {error}")
            if len(results["errors"]) > 5:
                print(f"  ... and {len(results['errors']) - 5} more errors")
        
        return True
        
    except Exception as e:
        print_status(f"Script failed with error: {str(e)}", "error")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
