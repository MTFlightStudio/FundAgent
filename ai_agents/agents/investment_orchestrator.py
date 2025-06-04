import os
import json
import logging
import datetime
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import sys

# Attempt to import HubSpot client and PDF extractor
try:
    from ai_agents.services import hubspot_client
except ImportError:
    hubspot_client = None
    logging.warning("HubSpot client not found. HubSpot interactions will fail.")

try:
    from ai_agents.tools import pdf_extractor
except ImportError:
    pdf_extractor = None
    logging.warning("PDF extractor not found. Pitch deck processing will fail.")

# Configure logging
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# --- Placeholder CLI/Tool Functions ---
# Replace these with actual calls to your scripts or functions

def run_founder_research_cli(founder_name: str, linkedin_url: str) -> Optional[Dict[str, Any]]:
    logger.info(f"Executing founder research for {founder_name} ({linkedin_url})...")
    try:
        # Construct the command
        # Assuming the script is invokable as a module: python -m ai_agents.agents.founder_research
        command = [
            sys.executable,  # Use the current Python interpreter
            "-m", "ai_agents.agents.founder_research_agent",
            "--linkedin_url", linkedin_url,
            "--name", founder_name
        ]
        
        logger.debug(f"Executing command: {' '.join(command)}")
        
        process = subprocess.run(command, capture_output=True, text=True, check=False, timeout=300) # 5 min timeout

        if process.returncode != 0:
            logger.error(f"Founder research script for {founder_name} failed with return code {process.returncode}.")
            logger.error(f"Stderr: {process.stderr}")
            return None
        
        try:
            result = json.loads(process.stdout)
            logger.info(f"Successfully completed founder research for {founder_name}.")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON output from founder research for {founder_name}: {e}")
            logger.error(f"Stdout: {process.stdout}")
            return None

    except subprocess.TimeoutExpired:
        logger.error(f"Founder research for {founder_name} timed out.")
        return None
    except FileNotFoundError:
        logger.error(f"Founder research script (ai_agents.agents.founder_research_agent) not found. Ensure it's in the PYTHONPATH.")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during founder research for {founder_name}: {e}", exc_info=True)
        return None

def run_company_research_cli(company_name: str) -> Optional[Dict[str, Any]]:
    logger.info(f"Executing company research for: {company_name}...")
    try:
        # Construct the command
        # Assuming the script is invokable as a module: python -m ai_agents.agents.company_research_agent
        # And it accepts a --company_name argument
        command = [
            sys.executable,  # Use the current Python interpreter
            "-m", "ai_agents.agents.company_research_agent",
            "--company_name", company_name
        ]
        
        logger.debug(f"Executing command: {' '.join(command)}")

        process = subprocess.run(command, capture_output=True, text=True, check=False, timeout=600) # 10 min timeout

        if process.returncode != 0:
            logger.error(f"Company research script for '{company_name}' failed with return code {process.returncode}.")
            logger.error(f"Stderr: {process.stderr}")
            return None
        
        try:
            result = json.loads(process.stdout)
            logger.info(f"Successfully completed company research for '{company_name}'.")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON output from company research for '{company_name}': {e}")
            logger.error(f"Stdout: {process.stdout}")
            return None
            
    except subprocess.TimeoutExpired:
        logger.error(f"Company research for '{company_name}' timed out.")
        return None
    except FileNotFoundError:
        logger.error(f"Company research script (e.g., ai_agents.agents.company_research_agent) not found. Ensure it's in the PYTHONPATH.")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during company research for '{company_name}': {e}", exc_info=True)
        return None

def run_market_intelligence_cli(sector: str) -> Optional[Dict[str, Any]]:
    """Runs the market_intelligence_agent.py script as a subprocess."""
    logger.info(f"Executing market intelligence for sector: {sector}...")
    command = [
        sys.executable, "-m", "ai_agents.agents.market_intelligence_agent", # Ensure this line is correct
        "--sector", sector
    ]
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=False, timeout=600) # 10 min timeout

        if process.returncode != 0:
            logger.error(f"Market intelligence script for sector '{sector}' failed with return code {process.returncode}.")
            logger.error(f"Stderr: {process.stderr}")
            return None
        
        try:
            result = json.loads(process.stdout)
            logger.info(f"Successfully completed market intelligence for sector '{sector}'.")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON output from market intelligence for sector '{sector}': {e}")
            logger.error(f"Stdout: {process.stdout}")
            return None
            
    except subprocess.TimeoutExpired:
        logger.error(f"Market intelligence for sector '{sector}' timed out.")
        return None
    except FileNotFoundError:
        logger.error(f"Market intelligence script (e.g., ai_agents.agents.market_intelligence_agent) not found. Ensure it's in the PYTHONPATH.")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during market intelligence for sector '{sector}': {e}", exc_info=True)
        return None

def run_decision_support_cli(
    hubspot_data: Dict[str, Any],
    pitch_deck_data: Optional[Dict[str, Any]],
    founder_profiles: List[Dict[str, Any]],
    market_analysis: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    logger.info("Simulating decision support analysis...")
    # This remains a placeholder. When you have a decision support agent script:
    # 1. Determine how it accepts complex data (e.g., via temporary JSON files, complex CLI args).
    # 2. Implement the subprocess call similar to the research agents.
    # For now, returning mock data based on some simple logic.
    
    recommendation = "Review" # Default
    confidence = "Medium"
    issues = []

    if not pitch_deck_data or pitch_deck_data.get("status") != "success" or not pitch_deck_data.get("structured_data"):
        issues.append("Pitch deck missing or could not be structured.")
        confidence = "Low"
    
    if not founder_profiles:
        issues.append("Founder profiles could not be generated.")
        confidence = "Low"
        
    if not market_analysis:
        issues.append("Market analysis could not be generated.")
    
    if hubspot_data.get("deal", {}).get("dealstage") == "closedlost": # Example property
        recommendation = "Pass (Already Closed Lost)"
        confidence = "High"

    return {
        "recommendation": recommendation,
        "confidence_level": confidence,
        "key_findings": f"Simulated analysis. HubSpot data processed. Pitch Deck status: {pitch_deck_data.get('status') if pitch_deck_data else 'N/A'}. Founders: {len(founder_profiles)}. Market data: {'Available' if market_analysis else 'N/A'}",
        "identified_risks_opportunities": issues if issues else ["Simulated: Standard review suggested."],
        "supporting_data_summary": {
            "deal_name": hubspot_data.get("deal", {}).get("dealname", "N/A"),
            "pitch_deck_summary": pitch_deck_data.get("structured_data", {}).get("executive_summary", "N/A") if pitch_deck_data and pitch_deck_data.get("structured_data") else "N/A"
        },
        "timestamp": datetime.datetime.utcnow().isoformat()
    }

def generate_investment_report(deal_id: str, all_data: Dict[str, Any]) -> str:
    logger.info(f"Generating comprehensive investment report for deal {deal_id}...")
    # This would format all collected data into a structured report (e.g., PDF, HTML, Markdown)
    report_path = f"investment_report_deal_{deal_id}_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(report_path, 'w') as f:
            json.dump(all_data, f, indent=4)
        logger.info(f"Report saved to {report_path}")
        return f"Report generated and saved to {report_path}. Content: {json.dumps(all_data)[:200]}..." # Return path or summary
    except Exception as e:
        logger.error(f"Failed to save report to {report_path}: {e}")
        return f"Failed to generate report for deal {deal_id}."


# --- Helper Functions ---

def extract_pitch_deck_url_from_hubspot_data(hubspot_data: Dict[str, Any]) -> Optional[str]:
    """
    Extracts the pitch deck URL from the HubSpot data.
    Looks in company properties first, then deal properties.
    The property name is assumed to be 'please_attach_your_pitch_deck'.
    """
    # Check associated companies
    if hubspot_data and hubspot_data.get("associated_companies"):
        for company in hubspot_data["associated_companies"]:
            properties = company.get("properties", {})
            if properties.get("please_attach_your_pitch_deck"):
                logger.info(f"Found pitch deck URL in company {company.get('id')}: {properties.get('please_attach_your_pitch_deck')}")
                return properties.get("please_attach_your_pitch_deck")

    # Check deal properties as a fallback (though less common for file URLs)
    if hubspot_data and hubspot_data.get("deal"):
        deal_props = hubspot_data["deal"]
        if deal_props.get("please_attach_your_pitch_deck"): # Assuming it might also be a deal property
            logger.info(f"Found pitch deck URL in deal properties: {deal_props.get('please_attach_your_pitch_deck')}")
            return deal_props.get("please_attach_your_pitch_deck")
            
    logger.info("Pitch deck URL 'please_attach_your_pitch_deck' not found in associated companies or deal properties.")
    return None

def update_hubspot_deal_with_assessment(deal_id: str, assessment: Dict[str, Any]) -> bool:
    """
    Placeholder for updating the HubSpot deal with the investment assessment.
    """
    logger.info(f"Simulating update of HubSpot deal {deal_id} with assessment.")
    if not hubspot_client:
        logger.error("HubSpot client not available. Cannot update deal.")
        return False
        
    # In a real scenario, you'd map `assessment` fields to specific HubSpot deal properties
    # For example, create custom properties in HubSpot like:
    # - "Investment Recommendation" (Text)
    # - "Investment Confidence Score" (Number)
    # - "Analysis Summary" (Multi-line text)
    # - "Last Analysis Date" (Date)

    properties_to_update = {
        "investment_recommendation": assessment.get("recommendation", "N/A"),
        "investment_confidence_score": assessment.get("confidence_score", 0.0) * 100, # Assuming score 0-1, HubSpot might want 0-100
        "investment_analysis_summary": assessment.get("summary", "No summary available."),
        "last_investment_analysis_date": datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ') # HubSpot expects ISO format UTC
    }
    logger.info(f"Properties to update for deal {deal_id}: {properties_to_update}")

    try:
        # This function would need to be implemented in hubspot_client.py
        # success = hubspot_client.update_deal_properties(deal_id, properties_to_update)
        # if success:
        #     logger.info(f"Successfully updated deal {deal_id} in HubSpot.")
        #     return True
        # else:
        #     logger.error(f"Failed to update deal {deal_id} in HubSpot.")
        #     return False
        logger.warning("Actual HubSpot deal update is simulated. `hubspot_client.update_deal_properties` not implemented.")
        return True # Simulate success
    except Exception as e:
        logger.error(f"Error during simulated HubSpot deal update for {deal_id}: {e}")
        return False


# --- Main Orchestration Function ---

def analyze_investment_opportunity(deal_id: str) -> Optional[str]:
    """
    Analyzes an investment opportunity based on a HubSpot Deal ID.
    Fetches data, runs research, performs decision support, and generates a report.
    """
    logger.info(f"Starting investment analysis for Deal ID: {deal_id}")
    
    if not hubspot_client:
        logger.error("HubSpot client is not available. Cannot proceed with analysis.")
        return None
    if not pdf_extractor:
        logger.warning("PDF extractor is not available. Pitch deck processing will be skipped.")
    
    all_collected_data: Dict[str, Any] = {
        "deal_id": deal_id,
        "orchestration_start_time": datetime.datetime.utcnow().isoformat(),
        "hubspot_deal_data": None,
        "pitch_deck_analysis": None,
        "founder_profiles": [],
        "market_analysis": None,
        "decision_support_output": None,
        "errors": []
    }

    # 1. Fetch Comprehensive Deal Data from HubSpot
    logger.info(f"Fetching comprehensive data for deal: {deal_id} from HubSpot...")
    try:
        hubspot_data = hubspot_client.get_deal_with_associated_data(deal_id)
        if not hubspot_data or not hubspot_data.get("deal"):
            logger.error(f"Failed to retrieve valid data from HubSpot for deal ID: {deal_id}")
            all_collected_data["errors"].append(f"Failed to retrieve HubSpot data for deal {deal_id}.")
            # Save partial data and exit or handle error appropriately
            report_filename = f"investment_report_deal_{deal_id}_error_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_filename, 'w') as f:
                json.dump(all_collected_data, f, indent=2)
            logger.info(f"Partial error report saved to {report_filename}")
            return report_filename # Or None if preferred
        all_collected_data['hubspot_deal_data'] = hubspot_data
        logger.info(f"Successfully fetched HubSpot data for deal {deal_id}.")
    except Exception as e:
        logger.error(f"Exception while fetching HubSpot data for deal {deal_id}: {e}", exc_info=True)
        all_collected_data["errors"].append(f"Exception fetching HubSpot data: {str(e)}")
        # Save partial data and exit
        report_filename = f"investment_report_deal_{deal_id}_error_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(all_collected_data, f, indent=2)
        logger.info(f"Partial error report saved to {report_filename}")
        return report_filename # Or None

    # 2. Process Pitch Deck (if URL available)
    if pdf_extractor:
        pitch_deck_url = extract_pitch_deck_url_from_hubspot_data(all_collected_data['hubspot_deal_data'])
        if pitch_deck_url:
            logger.info(f"Pitch deck URL found: {pitch_deck_url}. Processing...")
            try:
                # Assuming attempt_llm_structuring=True is desired for the orchestrator's use
                pitch_deck_result = pdf_extractor.process_pdf_source(pitch_deck_url, attempt_llm_structuring=True)
                all_collected_data['pitch_deck_analysis'] = pitch_deck_result
                if pitch_deck_result and pitch_deck_result.get("status") == "success":
                    logger.info("Pitch deck processed successfully.")
                else:
                    logger.warning(f"Pitch deck processing failed or returned non-success status: {pitch_deck_result.get('error_message', 'No error message')}")
                    all_collected_data["errors"].append(f"Pitch deck processing issue: {pitch_deck_result.get('error_message', 'Unknown')}")
            except Exception as e:
                logger.error(f"Exception during pitch deck processing for URL {pitch_deck_url}: {e}", exc_info=True)
                all_collected_data["errors"].append(f"Exception processing pitch deck: {str(e)}")
                all_collected_data['pitch_deck_analysis'] = {"status": "error", "error_message": str(e), "url": pitch_deck_url}
        else:
            logger.info("No pitch deck URL found in HubSpot data.")
            all_collected_data['pitch_deck_analysis'] = {"status": "skipped", "message": "No URL found"}
    else:
        logger.info("PDF extractor not available, skipping pitch deck processing.")
        all_collected_data['pitch_deck_analysis'] = {"status": "skipped", "message": "PDF extractor not available"}


    # 3. Perform Research Tasks (Founder & Market)
    # Using ThreadPoolExecutor for concurrent research tasks
    founder_profiles_results = []
    market_analysis_result = None
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_task = {}

        # Submit Founder Research tasks
        if all_collected_data['hubspot_deal_data'] and all_collected_data['hubspot_deal_data'].get('associated_contacts'):
            contacts = all_collected_data['hubspot_deal_data']['associated_contacts']
            logger.info(f"Found {len(contacts)} associated contacts. Submitting founder research tasks...")
            for contact in contacts:
                properties = contact.get("properties", {})
                first_name = properties.get("firstname")
                last_name = properties.get("lastname")
                linkedin_url = properties.get("hs_linkedin_url") # Confirm this is the correct property name

                if first_name and last_name and linkedin_url:
                    full_name = f"{first_name} {last_name}"
                    logger.info(f"Submitting founder research for: {full_name} ({linkedin_url})")
                    future_to_task[executor.submit(run_founder_research_cli, full_name, linkedin_url)] = f"Founder: {full_name}"
                elif first_name and last_name:
                    logger.info(f"Contact {first_name} {last_name} found, but missing LinkedIn URL. Skipping founder research for this contact.")
                else:
                    logger.info(f"Contact ID {contact.get('id')} missing name or LinkedIn URL. Skipping.")
        else:
            logger.info("No associated contacts found in HubSpot data for founder research.")

        # Submit Market Intelligence task (example: based on company industry)
        # Extract sector/industry from company data
        company_sector = None
        if all_collected_data['hubspot_deal_data'] and all_collected_data['hubspot_deal_data'].get('associated_companies'):
            primary_company = all_collected_data['hubspot_deal_data']['associated_companies'][0] # Assuming first is primary
            company_props = primary_company.get("properties", {})
            # Try specific field first, then fallback to general 'industry'
            company_sector = company_props.get("what_sector_is_your_business_product_", company_props.get("industry"))
            if company_sector:
                logger.info(f"Submitting market intelligence research for sector: {company_sector}")
                future_to_task[executor.submit(run_market_intelligence_cli, company_sector)] = f"Market: {company_sector}"
            else:
                logger.warning("Could not determine company sector for market intelligence research.")
                all_collected_data["errors"].append("Market research skipped: No sector found.")
        else:
            logger.info("No associated company found for market research.")
            all_collected_data["errors"].append("Market research skipped: No company data.")

        # Collect results from completed tasks
        for future in as_completed(future_to_task):
            task_name = future_to_task[future]
            try:
                result = future.result()
                if result:
                    logger.info(f"Successfully completed task: {task_name}")
                    if task_name.startswith("Founder:"):
                        founder_profiles_results.append(result)
                    elif task_name.startswith("Market:"):
                        market_analysis_result = result
                else:
                    logger.warning(f"Task {task_name} returned no result or failed.")
                    all_collected_data["errors"].append(f"Task {task_name} failed or returned empty.")
            except Exception as exc:
                logger.error(f"Task {task_name} generated an exception: {exc}", exc_info=True)
                all_collected_data["errors"].append(f"Exception in task {task_name}: {str(exc)}")
    
    all_collected_data['founder_profiles'] = founder_profiles_results
    all_collected_data['market_analysis'] = market_analysis_result

    # 4. Run Decision Support
    logger.info("Running decision support analysis...")
    try:
        decision_output = run_decision_support_cli(
            hubspot_data=all_collected_data['hubspot_deal_data'], 
            pitch_deck_data=all_collected_data['pitch_deck_analysis'],
            founder_profiles=all_collected_data['founder_profiles'],
            market_analysis=all_collected_data['market_analysis']
        )
        all_collected_data['decision_support_output'] = decision_output
        if decision_output and decision_output.get("status") != "error": # Assuming decision agent returns a dict with status
             logger.info("Decision support analysis completed.")
        else:
            logger.warning(f"Decision support analysis may have failed or returned an error: {decision_output}")
            all_collected_data["errors"].append(f"Decision support issue: {decision_output.get('message', 'Unknown') if isinstance(decision_output, dict) else 'Format error'}")

    except Exception as e:
        logger.error(f"Exception during decision support analysis: {e}", exc_info=True)
        all_collected_data["errors"].append(f"Exception in decision support: {str(e)}")
        all_collected_data['decision_support_output'] = {"status": "error", "message": str(e)}


    # 5. Generate Final Report (JSON file)
    all_collected_data["orchestration_end_time"] = datetime.datetime.utcnow().isoformat()
    report_filename = f"investment_report_deal_{deal_id}_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(report_filename, 'w') as f:
            json.dump(all_collected_data, f, indent=2)
        logger.info(f"Investment analysis report saved to: {report_filename}")
    except Exception as e:
        logger.error(f"Failed to save investment report {report_filename}: {e}", exc_info=True)
        # Still return the filename, as some data might have been collected
        return report_filename 

    # 6. (Optional) Update HubSpot Deal with Assessment Summary
    if all_collected_data.get('decision_support_output') and isinstance(all_collected_data['decision_support_output'], dict) and all_collected_data['decision_support_output'].get("investment_assessment"):
        # Assuming the decision_support_output is the InvestmentResearch model dict
        # and it contains an 'investment_assessment' key which itself is a dict.
        assessment_details_for_hubspot = all_collected_data['decision_support_output'] # Pass the whole research object
        
        # We need to map fields from our InvestmentResearch/InvestmentAssessment model
        # to what update_hubspot_deal_with_assessment expects or what HubSpot properties we have.
        # For now, let's create a simplified assessment summary for HubSpot.
        
        # Example: Extracting recommendation from the InvestmentResearch model structure
        recommendation = all_collected_data['decision_support_output'].get('overall_summary_and_recommendation', "N/A")
        confidence = all_collected_data['decision_support_output'].get('confidence_score_overall', 0.0)
        
        # The investment_assessment sub-dictionary might contain more details
        ia_details = all_collected_data['decision_support_output'].get('investment_assessment', {})
        criteria_summary = ia_details.get('overall_criteria_summary', 'Details in report.')

        simplified_assessment_for_hs = {
            "recommendation": recommendation,
            "confidence_score": confidence, # Assuming this is 0.0-1.0
            "summary": f"Overall Recommendation: {recommendation}. Criteria Summary: {criteria_summary}. See full report: {report_filename}",
            # Add other key fields you want to push to HubSpot
        }
        update_hubspot_deal_with_assessment(deal_id, simplified_assessment_for_hs)
    else:
        logger.warning("Decision support output not available or not in expected format for HubSpot update.")

    return report_filename


if __name__ == "__main__":
    # Ensure essential clients are available
    if not hubspot_client:
        logger.error("HubSpot client could not be initialized. Exiting test run.")
        exit(1)
    # PDF extractor is optional, orchestrator should handle its absence.
    # if not pdf_extractor:
    #     logger.error("PDF Extractor could not be initialized. Exiting test run.")
    #     exit(1)

    # --- Use a TEST Deal ID that exists in your HubSpot instance ---
    # Replace with a Deal ID that has:
    # 1. Associated contacts (some with hs_linkedin_url)
    # 2. Associated company (with 'what_sector_is_your_business_product_' or 'industry' property)
    # 3. A form submission with a pitch deck URL in a field like 'please_attach_your_pitch_deck' (optional)
    
    TEST_DEAL_ID = "227710582988" # <<< From your deal_data_export.json example
    
    if not TEST_DEAL_ID or TEST_DEAL_ID == "YOUR_TEST_DEAL_ID_HERE": # Basic check
        logger.warning("Please set a valid TEST_DEAL_ID in the __main__ block for testing.")
    else:
        logger.info(f"Attempting to analyze test Deal ID: {TEST_DEAL_ID}")
        report_output_path = analyze_investment_opportunity(TEST_DEAL_ID)

        if report_output_path:
            logger.info(f"\n--- Orchestrator Test Run Complete ---")
            logger.info(f"Final Report saved to: {report_output_path}")
            logger.info(f"Check the report file and console logs for detailed steps and outcomes.")
        else:
            logger.error("\n--- Orchestrator Test Run Failed ---")
            logger.error("Analysis did not complete successfully or failed to produce a report path. Check logs for errors.") 