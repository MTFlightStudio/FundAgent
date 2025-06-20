import os
import json
import logging
import datetime
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import sys
import re # Added for safe filenames
import tempfile # Added for decision support agent call
from pathlib import Path # Added for decision support agent call

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

RUN_LOGS_DIR = "run_logs"

def _ensure_log_dir_and_get_safe_filename(deal_id: str, agent_type: str, entity_name: str) -> str:
    """Ensures RUN_LOGS_DIR exists and creates a safe, unique filename for agent logs."""
    os.makedirs(RUN_LOGS_DIR, exist_ok=True)
    
    safe_entity_name = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', entity_name)[:50] # Sanitize and shorten
    timestamp = datetime.datetime.now(datetime.UTC).strftime('%Y%m%d_%H%M%S_%f')
    filename = f"deal_{deal_id}_{agent_type}_{safe_entity_name}_{timestamp}.log"
    return os.path.join(RUN_LOGS_DIR, filename)

# --- Placeholder CLI/Tool Functions ---
# Replace these with actual calls to your scripts or functions

def run_founder_research_cli(deal_id: str, founder_name: str, linkedin_url: str) -> Optional[Dict[str, Any]]:
    logger.info(f"Executing founder research for {founder_name} ({linkedin_url}) for Deal ID {deal_id}.")
    log_file_path = _ensure_log_dir_and_get_safe_filename(deal_id, "founder", founder_name)
    
    try:
        command = [
            sys.executable,
            "-m", "ai_agents.agents.founder_research_agent",
            "--linkedin_url", linkedin_url,
            "--name", founder_name
        ]
        logger.debug(f"Executing command: {' '.join(command)}")
        process = subprocess.run(command, capture_output=True, text=True, check=False, timeout=300)

        with open(log_file_path, 'w') as f:
            f.write(f"--- Command ---\n{' '.join(command)}\n\n")
            f.write(f"--- Return Code ---\n{process.returncode}\n\n")
            f.write(f"--- STDOUT ---\n{process.stdout}\n\n")
            f.write(f"--- STDERR ---\n{process.stderr}\n")
        logger.info(f"Full log for founder {founder_name} saved to: {log_file_path}")

        if process.returncode != 0:
            logger.error(f"Founder research script for {founder_name} failed. See log: {log_file_path}")
            return None
        
        try:
            result = json.loads(process.stdout)
            logger.info(f"Successfully completed founder research for {founder_name}.")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON output from founder research for {founder_name}: {e}. See log: {log_file_path}")
            return None

    except subprocess.TimeoutExpired:
        logger.error(f"Founder research for {founder_name} timed out. Log: {log_file_path}")
        with open(log_file_path, 'a') as f: f.write("\n--- ERROR: Process Timed Out ---\n")
        return None
    except FileNotFoundError:
        logger.error(f"Founder research script not found. Log: {log_file_path}")
        with open(log_file_path, 'a') as f: f.write("\n--- ERROR: Script Not Found ---\n")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during founder research for {founder_name}: {e}. Log: {log_file_path}", exc_info=True)
        with open(log_file_path, 'a') as f: f.write(f"\n--- ERROR: Unexpected Exception ---\n{str(e)}\n")
        return None

def run_company_research_cli(deal_id: str, company_name: str) -> Optional[Dict[str, Any]]:
    logger.info(f"Executing company research for: {company_name} for Deal ID {deal_id}...")
    log_file_path = _ensure_log_dir_and_get_safe_filename(deal_id, "company", company_name)
    try:
        command = [
            sys.executable,
            "-m", "ai_agents.agents.company_research_agent",
            "--company_name", company_name
        ]
        logger.debug(f"Executing command: {' '.join(command)}")
        process = subprocess.run(command, capture_output=True, text=True, check=False, timeout=600)

        with open(log_file_path, 'w') as f:
            f.write(f"--- Command ---\n{' '.join(command)}\n\n")
            f.write(f"--- Return Code ---\n{process.returncode}\n\n")
            f.write(f"--- STDOUT ---\n{process.stdout}\n\n")
            f.write(f"--- STDERR ---\n{process.stderr}\n")
        logger.info(f"Full log for company {company_name} saved to: {log_file_path}")

        if process.returncode != 0:
            logger.error(f"Company research script for '{company_name}' failed. See log: {log_file_path}")
            return None
        
        try:
            result = json.loads(process.stdout)
            logger.info(f"Successfully completed company research for '{company_name}'.")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON output from company research for '{company_name}': {e}. See log: {log_file_path}")
            return None
            
    except subprocess.TimeoutExpired:
        logger.error(f"Company research for '{company_name}' timed out. Log: {log_file_path}")
        with open(log_file_path, 'a') as f: f.write("\n--- ERROR: Process Timed Out ---\n")
        return None
    except FileNotFoundError:
        logger.error(f"Company research script not found. Log: {log_file_path}")
        with open(log_file_path, 'a') as f: f.write("\n--- ERROR: Script Not Found ---\n")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during company research for '{company_name}': {e}. Log: {log_file_path}", exc_info=True)
        with open(log_file_path, 'a') as f: f.write(f"\n--- ERROR: Unexpected Exception ---\n{str(e)}\n")
        return None

def run_market_intelligence_cli(deal_id: str, sector: str, hubspot_data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Run market intelligence analysis for a specific sector with optional HubSpot data for enhanced targeting.
    
    Args:
        deal_id: The deal ID for logging and tracking
        sector: The market or industry sector (can be generic if HubSpot data provides specificity)
        hubspot_data: Optional HubSpot deal data for targeted market definition
    
    Returns:
        Dictionary containing market analysis results or None if failed
    """
    safe_filename = _ensure_log_dir_and_get_safe_filename(deal_id, "market_research", sector)
    log_file_path = safe_filename  # This is the full path including directory

    try:
        # Import here to avoid circular import
        from ai_agents.agents.market_intelligence_agent import run_market_intelligence_cli as market_agent_cli
        
        if hubspot_data:
            logger.info(f"InvestmentOrchestrator: Running ENHANCED market intelligence for deal {deal_id} with HubSpot context")
            result = market_agent_cli(sector, hubspot_data=hubspot_data)
        else:
            logger.info(f"InvestmentOrchestrator: Running standard market intelligence for deal {deal_id}, sector: {sector}")
            result = market_agent_cli(sector)
        
        if result and hasattr(result, 'model_dump'):
            result_dict = result.model_dump()
            logger.info(f"InvestmentOrchestrator: Market intelligence completed successfully for deal {deal_id}")
            return result_dict
        elif result:
            logger.info(f"InvestmentOrchestrator: Market intelligence completed for deal {deal_id}")
            return result
        else:
            logger.warning(f"InvestmentOrchestrator: Market intelligence returned no results for deal {deal_id}, sector: {sector}")
            return None
            
    except Exception as e:
        logger.error(f"InvestmentOrchestrator: Market intelligence failed for deal {deal_id}, sector: {sector}. Error: {e}", exc_info=True)
        return None

def run_decision_support_cli(
    deal_id: str, # Added deal_id for logging
    hubspot_data: Dict[str, Any],
    pitch_deck_data: Optional[Dict[str, Any]],
    founder_profiles: List[Dict[str, Any]],
    company_profile: Optional[Dict[str, Any]],
    market_analysis: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Runs the decision support agent CLI by creating temporary files for complex inputs.
    """
    logger.info("Preparing to run decision support analysis via CLI...")
    log_file_path = _ensure_log_dir_and_get_safe_filename(deal_id, "decision_support", "analysis")

    # Use a temporary directory to stage the input files
    with tempfile.TemporaryDirectory(prefix=f"decision_support_{deal_id}_") as temp_dir:
        temp_path = Path(temp_dir)
        
        # --- File Preparation ---
        company_file = None
        if company_profile:
            company_file = temp_path / "company_profile.json"
            with open(company_file, 'w') as f:
                json.dump(company_profile, f)

        market_file = None
        if market_analysis:
            market_file = temp_path / "market_analysis.json"
            with open(market_file, 'w') as f:
                json.dump(market_analysis, f)

        founder_files = []
        if founder_profiles:
            for i, profile in enumerate(founder_profiles):
                founder_file = temp_path / f"founder_{i+1}.json"
                with open(founder_file, 'w') as f:
                    json.dump(profile, f)
                founder_files.append(str(founder_file))
        
        # Create a single additional context file
        additional_context_file = temp_path / "additional_context.json"
        additional_context_data = {
            "hubspot_deal_data": hubspot_data,
            "pitch_deck_analysis": pitch_deck_data
        }
        with open(additional_context_file, 'w') as f:
            json.dump(additional_context_data, f)
            
        # --- Command Construction ---
        command = [
            sys.executable, "-m", "ai_agents.agents.decision_support_agent",
        ]
        if not company_file:
            logger.error("Decision support requires a company profile, but none was provided. Aborting.")
            return {"status": "error", "error_message": "Company profile is a required input."}
        
        command.extend(["--company_file", str(company_file)])

        if founder_files:
            command.append("--founder_files")
            command.extend(founder_files)

        if market_file:
            command.extend(["--market_file", str(market_file)])

        command.extend(["--additional_context_file", str(additional_context_file)])

        # --- Subprocess Execution ---
        logger.debug(f"Executing decision support command: {' '.join(command)}")
        try:
            process = subprocess.run(command, capture_output=True, text=True, check=False, timeout=600)

            with open(log_file_path, 'w') as f:
                f.write(f"--- Command ---\n{' '.join(command)}\n\n")
                f.write(f"--- Return Code ---\n{process.returncode}\n\n")
                f.write(f"--- STDOUT ---\n{process.stdout}\n\n")
                f.write(f"--- STDERR ---\n{process.stderr}\n")
            logger.info(f"Full log for decision support saved to: {log_file_path}")
            
            if process.returncode != 0:
                logger.error(f"Decision support script failed. See log: {log_file_path}")
                return None
            
            try:
                result = json.loads(process.stdout)
                logger.info("Successfully completed decision support analysis.")
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON output from decision support: {e}. See log: {log_file_path}")
                return None

        except subprocess.TimeoutExpired:
            logger.error(f"Decision support analysis timed out. Log: {log_file_path}")
            with open(log_file_path, 'a') as f: f.write("\n--- ERROR: Process Timed Out ---\n")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during decision support: {e}. Log: {log_file_path}", exc_info=True)
            with open(log_file_path, 'a') as f: f.write(f"\n--- ERROR: Unexpected Exception ---\n{str(e)}\n")
            return None

def generate_investment_report(deal_id: str, all_data: Dict[str, Any]) -> str:
    logger.info(f"Generating comprehensive investment report for deal {deal_id}...")
    # This would format all collected data into a structured report (e.g., PDF, HTML, Markdown)
    report_path = f"investment_report_deal_{deal_id}_{datetime.datetime.now(datetime.UTC).strftime('%Y%m%d_%H%M%S')}.json"
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
        "last_investment_analysis_date": datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%S.%fZ') # HubSpot expects ISO format UTC
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
    # Ensure the main run_logs directory exists (though _ensure_log_dir_and_get_safe_filename handles it too)
    os.makedirs(RUN_LOGS_DIR, exist_ok=True) 

    if not hubspot_client:
        logger.error("HubSpot client is not available. Cannot proceed with analysis.")
        return None
    if not pdf_extractor:
        logger.warning("PDF extractor is not available. Pitch deck processing will be skipped.")
    
    all_collected_data: Dict[str, Any] = {
        "deal_id": deal_id,
        "orchestration_start_time": datetime.datetime.now(datetime.UTC).isoformat(),
        "hubspot_deal_data": None,
        "pitch_deck_analysis": None,
        "founder_profiles": [],
        "company_profile": None,
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
            report_filename = f"investment_report_deal_{deal_id}_error_{datetime.datetime.now(datetime.UTC).strftime('%Y%m%d_%H%M%S')}.json"
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
        report_filename = f"investment_report_deal_{deal_id}_error_{datetime.datetime.now(datetime.UTC).strftime('%Y%m%d_%H%M%S')}.json"
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
    company_profile_result = None
    
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
                    logger.info(f"Submitting founder research for: {full_name} ({linkedin_url}) for deal {deal_id}")
                    future_to_task[executor.submit(run_founder_research_cli, deal_id, full_name, linkedin_url)] = f"Founder: {full_name}"
                elif first_name and last_name:
                    logger.info(f"Contact {first_name} {last_name} found, but missing LinkedIn URL. Skipping founder research for this contact.")
                else:
                    logger.info(f"Contact ID {contact.get('id')} missing name or LinkedIn URL. Skipping.")
        else:
            logger.info("No associated contacts found in HubSpot data for founder research.")

        # Submit Company Research task
        primary_company_name = None
        if all_collected_data['hubspot_deal_data'] and all_collected_data['hubspot_deal_data'].get('associated_companies'):
            primary_company_data = all_collected_data['hubspot_deal_data']['associated_companies'][0] # Assuming first is primary
            primary_company_name = primary_company_data.get("properties", {}).get("name")
            if primary_company_name:
                logger.info(f"Submitting company research for: {primary_company_name} for deal {deal_id}")
                future_to_task[executor.submit(run_company_research_cli, deal_id, primary_company_name)] = f"Company: {primary_company_name}"
            else:
                logger.warning("Could not determine primary company name for company research.")
                all_collected_data["errors"].append("Company research skipped: No company name found.")
        else:
            logger.info("No associated company found for company research.")
            all_collected_data["errors"].append("Company research skipped: No company data for name extraction.")

        # Submit Market Intelligence task (example: based on company industry)
        # Extract sector/industry from company data
        company_sector = None
        if all_collected_data['hubspot_deal_data'] and all_collected_data['hubspot_deal_data'].get('associated_companies'):
            primary_company = all_collected_data['hubspot_deal_data']['associated_companies'][0] # Assuming first is primary
            company_props = primary_company.get("properties", {})
            # Try specific field first, then fallback to general 'industry'
            company_sector = company_props.get("what_sector_is_your_business_product_", company_props.get("industry"))
            if company_sector:
                logger.info(f"Submitting market intelligence research for sector: {company_sector} for deal {deal_id}")
                future_to_task[executor.submit(run_market_intelligence_cli, deal_id, company_sector, all_collected_data['hubspot_deal_data'])] = f"Market: {company_sector}"
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
                    elif task_name.startswith("Company:"):
                        company_profile_result = result
                else:
                    logger.warning(f"Task {task_name} returned no result or failed.")
                    all_collected_data["errors"].append(f"Task {task_name} failed or returned empty.")
            except Exception as exc:
                logger.error(f"Task {task_name} generated an exception: {exc}", exc_info=True)
                all_collected_data["errors"].append(f"Exception in task {task_name}: {str(exc)}")
    
    all_collected_data['founder_profiles'] = founder_profiles_results
    all_collected_data['market_analysis'] = market_analysis_result
    all_collected_data['company_profile'] = company_profile_result

    # 4. Run Decision Support
    logger.info("Running decision support analysis...")
    try:
        decision_output = run_decision_support_cli(
            deal_id=deal_id, # Pass deal_id for logging
            hubspot_data=all_collected_data['hubspot_deal_data'], 
            pitch_deck_data=all_collected_data['pitch_deck_analysis'],
            founder_profiles=all_collected_data['founder_profiles'],
            company_profile=all_collected_data['company_profile'],
            market_analysis=all_collected_data['market_analysis']
        )
        all_collected_data['decision_support_output'] = decision_output
        
        # Check based on the structure of the InvestmentResearch model
        if decision_output and decision_output.get("status") != "Error":
             logger.info("Decision support analysis completed.")
        else:
            logger.warning(f"Decision support analysis may have failed or returned an error: {decision_output}")
            error_msg = "Unknown error"
            if isinstance(decision_output, dict):
                error_msg = decision_output.get('error_message', 'No error message provided')
            all_collected_data["errors"].append(f"Decision support issue: {error_msg}")

    except Exception as e:
        logger.error(f"Exception during decision support analysis: {e}", exc_info=True)
        all_collected_data["errors"].append(f"Exception in decision support: {str(e)}")
        all_collected_data['decision_support_output'] = {"status": "error", "message": str(e)}


    # 5. Generate Final Report (JSON file)
    all_collected_data["orchestration_end_time"] = datetime.datetime.now(datetime.UTC).isoformat()
    report_filename = f"investment_report_deal_{deal_id}_{datetime.datetime.now(datetime.UTC).strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(report_filename, 'w') as f:
            json.dump(all_collected_data, f, indent=2)
        logger.info(f"Investment analysis report saved to: {report_filename}")
    except Exception as e:
        logger.error(f"Failed to save investment report {report_filename}: {e}", exc_info=True)
        # Still return the filename, as some data might have been collected
        return report_filename 

    # 6. (Optional) Update HubSpot Deal with Assessment Summary
    if all_collected_data.get('decision_support_output') and isinstance(all_collected_data['decision_support_output'], dict) and all_collected_data['decision_support_output'].get("status") == "Complete":
        # The output is the full InvestmentResearch model as a dict.
        
        # Extracting recommendation from the InvestmentResearch model structure
        recommendation = all_collected_data['decision_support_output'].get('overall_summary_and_recommendation', "N/A")
        confidence = all_collected_data['decision_support_output'].get('confidence_score_overall', 0.0)
        
        # The investment_assessment sub-dictionary might contain more details
        ia_details = all_collected_data['decision_support_output'].get('investment_assessment', {})
        criteria_summary = ia_details.get('overall_criteria_summary', 'Details in report.')

        simplified_assessment_for_hs = {
            "recommendation": recommendation,
            "confidence_score": confidence, # Assuming this is 0.0-1.0
            "summary": f"Overall Recommendation: {recommendation}. Confidence: {confidence*100:.0f}%. Criteria Summary: {criteria_summary}. See full report: {report_filename}",
        }
        update_hubspot_deal_with_assessment(deal_id, simplified_assessment_for_hs)
    else:
        logger.warning("Decision support output not available or not in expected 'Complete' status for HubSpot update.")

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
    
    TEST_DEAL_ID = "239412768969" # <<< From your deal_data_export.json example
    
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