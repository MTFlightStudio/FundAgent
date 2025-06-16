import os
import json
import re
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from dotenv import load_dotenv
import argparse
import sys
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.language_models.chat_models import BaseChatModel

from ai_agents.models.investment_research import (
    InvestmentAssessment,
    CompanyProfile,
    FounderProfile,
    MarketAnalysis,
    InvestmentResearch
)

from ai_agents.config.model_config import get_llm_for_agent
from ai_agents.utils.retry_handler import with_smart_retry, RetryConfig

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

investment_assessment_parser = PydanticOutputParser(pydantic_object=InvestmentAssessment)

DECISION_SUPPORT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert investment analyst for Flight Story, a venture capital fund with very specific and strict investment criteria. Your official thesis is 'visionary founders who are contributing to a healthier, happier humanity through innovation'. In practice, this means you look for exceptional founders with cool, world-changing ideas that Steven Bartlett (SB), our founder, would be personally excited about.\n"
            "Your task is to synthesize research data and make an investment recommendation based on the following framework.\n\n"
            "FLIGHT STORY'S 6 MUST-HAVE INVESTMENT CRITERIA:\n\n"
            "--- Industry/Mission/Idea ---\n"
            "1.  **Focus Industry Fit**: The company MUST primarily operate in one of these sectors:\n"
            "    - Consumer Products, Retail & Ecommerce\n"
            "    - Consumer Services\n"
            "    - Creative Industries\n"
            "    - Education/Ed-tech\n"
            "    - Environmental & Sustainability\n"
            "    - Financial Services/Fintech\n"
            "    - Food & Beverage\n"
            "    - Healthcare & Life Sciences/Health-tech\n"
            "    - Hospitality & Tourism\n"
            "    - Information Technology & Telecommunications\n"
            "    - Low Alcoholic Alternatives\n"
            "    - Media, Marketing, Entertainment, Gaming\n"
            "    - Professional & Business Services\n"
            "    - Public Sector & Non-Profit\n"
            "    - Science & Technology (including Space, Biotech, and AI)\n"
            "    - Wellness/Wellbeing\n\n"
            "2.  **Mission Alignment**: The company's mission MUST align with creating a 'healthier, happier humanity'. It must be genuinely impact-driven.\n\n"
            "3.  **Exciting Solution (The 'SB Test')**: This is a critical, qualitative test. The idea must be genuinely exciting. Ask yourself:\n"
            "    - Is it cool, interesting, or 'sexy'?\n"
            "    - Is it something SB would personally use, be proud to be associated with, or post about on his social media?\n"
            "    - Does it leverage SB's unique skill set and personal brand in a way that provides more value than just a capital injection?\n"
            "    - Is it a visionary product that enough people would want to be associated with?\n"
            "    - Does it have the potential for serious economic impact?\n\n"
            "--- Founder Potential ---\n"
            "4.  **Founded Something Relevant Before**: The founder(s) MUST have founded another company *before* the one being evaluated. The previous venture should be impressive or relevant. Do NOT count the current company as prior experience.\n\n"
            "5.  **Impressive, Relevant Past Experience**: The founder(s) should have worked at impressive or highly relevant companies that would have prepared them for success (e.g., FAANG, major brands, other successful VC-backed startups).\n\n"
            "6.  **Exceptionally Smart or Strategic**: There must be strong evidence of the founder(s) being exceptionally intelligent or strategic. Look for signals such as education from top-tier institutions, experience at elite companies, or outstanding thought leadership. While not an exhaustive list, here are prime examples of strong signals:\n"
            "    - **Universities**: Cambridge, Oxford, Harvard, MIT, LSE, Durham, Imperial, St Andrews, Yale, Stanford, Princeton, Brown, Cornell, Penn, Dartmouth, Columbia.\n"
            "    - **Companies**: McKinsey, BCG, Bain, Morgan Stanley, JP Morgan, Goldman Sachs, Bank of America, Allen & Overy, Kirkland & Ellis, Clifford Chance, Freshfields, Linklaters, Slaughter and May, Amazon, Google, Meta, TikTok, Microsoft.\n\n"
            "SCORING GUIDELINES:\n"
            "- Be VERY conservative and strict. Only mark 'true' if there is clear, undeniable evidence. When in doubt, score 'false' or 'null'.\n"
            "- Consider all founders collectively for criteria 4-6.\n\n"
            "RECOMMENDATION THRESHOLDS:\n"
            "- PASS (Proceed to partner meeting): 5-6 out of 6 criteria met\n"
            "- EXPLORE (Gather more information): 3-4 out of 6 criteria met\n"
            "- DECLINE (Pass on investment): Less than 3 criteria met\n\n"
            "ANALYSIS APPROACH:\n"
            "1. Carefully review the company profile, founder profiles, and market analysis, and any additional context (like pitch deck summary or deal info from HubSpot).\n"
            "2. Evaluate each of the 6 criteria based on the evidence provided across all data sources.\n"
            "3. Identify key risks and opportunities.\n"
            "4. Formulate an investment thesis if applicable.\n"
            "5. Recommend clear next steps.\n\n"
            "Your output MUST be a valid JSON object conforming to the InvestmentAssessment schema.\n"
            "Base your assessment ONLY on the provided data. If information is missing, note it in your analysis.\n"
            "Schema:\n{format_instructions}"
        ),
        (
            "human", 
            "Please analyze this investment opportunity:\n\n"
            "COMPANY DATA:\n{company_data}\n\n"
            "FOUNDER DATA (Array of profiles):\n{founder_data}\n\n"
            "MARKET DATA:\n{market_data}\n\n"
            "ADDITIONAL CONTEXT (e.g., Pitch Deck Summary, HubSpot Deal Info):\n{additional_context}"
        ),
    ]
).partial(format_instructions=investment_assessment_parser.get_format_instructions())

def load_research_data(
    company_file: Optional[str] = None,
    founder_files: Optional[List[str]] = None,
    market_file: Optional[str] = None,
    additional_context_file: Optional[str] = None
) -> Tuple[Optional[CompanyProfile], List[FounderProfile], Optional[MarketAnalysis], Optional[Dict[str, Any]]]:
    """
    Load research data from JSON files.
    Returns tuple of (company_profile, founder_profiles, market_analysis, additional_context)
    """
    company_profile = None
    founder_profiles = []
    market_analysis = None
    additional_context_data = None
    
    # Load company data
    if company_file and os.path.exists(company_file):
        try:
            with open(company_file, 'r') as f:
                company_data = json.load(f)
            company_profile = CompanyProfile.model_validate(company_data)
            logger.info(f"Loaded company profile: {company_profile.company_name}")
        except Exception as e:
            logger.error(f"Error loading company file {company_file}: {e}")
    
    # Load founder data
    if founder_files:
        for founder_file in founder_files:
            if os.path.exists(founder_file):
                try:
                    with open(founder_file, 'r') as f:
                        founder_data_item = json.load(f)
                    # Ensure we handle both single founder dict and list of founders in a file
                    if isinstance(founder_data_item, list):
                        for fd_item in founder_data_item:
                            founder_profile = FounderProfile.model_validate(fd_item)
                            founder_profiles.append(founder_profile)
                            logger.info(f"Loaded founder profile: {founder_profile.name}")
                    else: # Single founder object in file
                        founder_profile = FounderProfile.model_validate(founder_data_item)
                        founder_profiles.append(founder_profile)
                        logger.info(f"Loaded founder profile: {founder_profile.name}")
                except Exception as e:
                    logger.error(f"Error loading founder file {founder_file}: {e}")
    
    # Load market data
    if market_file and os.path.exists(market_file):
        try:
            with open(market_file, 'r') as f:
                market_data = json.load(f)
            market_analysis = MarketAnalysis.model_validate(market_data)
            logger.info(f"Loaded market analysis for: {market_analysis.industry_overview[:50]}...")
        except Exception as e:
            logger.error(f"Error loading market file {market_file}: {e}")

    # Load additional context data (for CLI testing)
    if additional_context_file and os.path.exists(additional_context_file):
        try:
            with open(additional_context_file, 'r') as f:
                additional_context_data = json.load(f)
            logger.info(f"Loaded additional context data from {additional_context_file}")
        except Exception as e:
            logger.error(f"Error loading additional context file {additional_context_file}: {e}")
            additional_context_data = {"error": f"Failed to load {additional_context_file}"}
    elif not additional_context_file:
        additional_context_data = {"info": "No additional context file provided for CLI run."}

    
    return company_profile, founder_profiles, market_analysis, additional_context_data

def calculate_recommendation(assessment: InvestmentAssessment) -> str:
    """
    Calculate the investment recommendation based on criteria scores.
    Returns: 'PASS', 'EXPLORE', or 'DECLINE'
    """
    criteria_scores = [
        assessment.fs_focus_industry_fit,
        assessment.fs_mission_alignment,
        assessment.fs_exciting_solution_to_problem,
        assessment.fs_founded_something_relevant_before,
        assessment.fs_impressive_relevant_past_experience,
        assessment.fs_exceptionally_smart_or_strategic
    ]
    
    # Count only True values (not None/null)
    true_count = sum(1 for score in criteria_scores if score is True)
    
    if true_count >= 5:
        return "PASS"
    elif true_count >= 3:
        return "EXPLORE"
    else:
        return "DECLINE"

@with_smart_retry(
    "decision_support",
    retry_config_override=RetryConfig(switch_model_on_rate_limit=True, max_retries=2),
    model_selector_func=lambda agent_name, prefer_fast=False: get_llm_for_agent(agent_name, prefer_fast=prefer_fast)
)
def run_decision_support_analysis(
    company_data_dict: Dict[str, Any],
    founder_data_list: List[Dict[str, Any]],
    market_data_dict: Optional[Dict[str, Any]],
    additional_context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Optional[InvestmentResearch]:
    """
    Runs the decision support analysis using provided data dictionaries and an injected LLM.
    """
    llm: Optional[BaseChatModel] = kwargs.get("llm")

    if not llm:
        logger.error("LLM not available (not provided by decorator or failed selection). Aborting decision support.")
        # Create a fallback InvestmentResearch object indicating the error
        return InvestmentResearch(
            query_or_target_entity=company_data_dict.get("company_name", "Unknown"),
            primary_analyst="AI System Error",
            status="Error",
            overall_summary_and_recommendation="ERROR_LLM_UNAVAILABLE",
            investment_assessment=InvestmentAssessment(overall_criteria_summary="LLM was not available for analysis."),
            error_message="LLM instance was not provided to the analysis function.",
            timestamp=datetime.utcnow().isoformat()
        )

    try:
        company_profile: Optional[CompanyProfile] = None
        if company_data_dict:
            try:
                company_profile = CompanyProfile(**company_data_dict)
            except Exception as e:
                logger.error(f"Error parsing company data: {e}. Data: {company_data_dict}", exc_info=True)
                # Create a fallback or return error InvestmentResearch
                return InvestmentResearch(
                    query_or_target_entity=company_data_dict.get("company_name", "Unknown"),
                    primary_analyst="AI System Error", status="Error",
                    overall_summary_and_recommendation="ERROR_INVALID_COMPANY_DATA",
                    error_message=f"Invalid company data: {e}",
                    investment_assessment=InvestmentAssessment(overall_criteria_summary=f"Invalid company data: {e}"),
                    timestamp=datetime.utcnow().isoformat()
                )
        else: # Company data is mandatory
            logger.error("Company data dictionary is missing or empty.")
            return InvestmentResearch(
                query_or_target_entity="Unknown", primary_analyst="AI System Error", status="Error",
                overall_summary_and_recommendation="ERROR_MISSING_COMPANY_DATA",
                error_message="Company data is mandatory and was not provided.",
                investment_assessment=InvestmentAssessment(overall_criteria_summary="Company data missing."),
                timestamp=datetime.utcnow().isoformat()
            )

        parsed_founder_profiles = []
        if founder_data_list:
            for i, fd in enumerate(founder_data_list):
                try:
                    parsed_founder_profiles.append(FounderProfile(**fd))
                except Exception as e:
                    logger.warning(f"Error parsing founder data for founder {i+1}, skipping. Error: {e}. Data: {fd}", exc_info=True)
        
        # It's okay if parsed_founder_profiles is empty if founder_data_list was empty/None.
        # If founder_data_list was provided but all failed parsing, it will be empty here.

        parsed_market_analysis = None
        if market_data_dict:
            try:
                parsed_market_analysis = MarketAnalysis(**market_data_dict)
            except Exception as e:
                logger.warning(f"Error parsing market data, proceeding without. Error: {e}. Data: {market_data_dict}", exc_info=True)
        
        # Prepare data for the LLM prompt
        company_data_str = company_profile.model_dump_json(indent=2)
        # Ensure founder_data_str is "N/A" or similar if empty, not just "[]"
        founder_data_str = json.dumps([fp.model_dump(mode='json') for fp in parsed_founder_profiles], indent=2) if parsed_founder_profiles else "N/A - No valid founder profiles provided or parsed."
        market_data_str = parsed_market_analysis.model_dump_json(indent=2) if parsed_market_analysis else "N/A - No market data provided or parsed."
        additional_context_str = json.dumps(additional_context, indent=2) if additional_context else "N/A - No additional context provided."

        chain = DECISION_SUPPORT_PROMPT_TEMPLATE | llm
        raw_llm_output = chain.invoke({
            "company_data": company_data_str,
            "founder_data": founder_data_str,
            "market_data": market_data_str,
            "additional_context": additional_context_str,
        })
        raw_llm_output_str = raw_llm_output.content if hasattr(raw_llm_output, 'content') else str(raw_llm_output)
        
        logger.debug(f"Raw LLM Output for Decision Support (first 1000 chars):\n{raw_llm_output_str[:1000]}")

        json_match = re.search(r"```json\s*([\s\S]*?)\s*```", raw_llm_output_str, re.DOTALL)
        json_to_parse = ""
        if json_match:
            json_to_parse = json_match.group(1).strip()
            logger.info("Successfully extracted JSON using markdown code block regex.")
        else:
            logger.warning("Markdown JSON block not found. Attempting fallback brace extraction.")
            try:
                first_brace = raw_llm_output_str.index('{')
                last_brace = raw_llm_output_str.rindex('}')
                if last_brace > first_brace:
                    json_to_parse = raw_llm_output_str[first_brace : last_brace+1].strip()
                    logger.info(f"Extracted potential JSON using brace finding (first 100 chars): {json_to_parse[:100]}...")
                else:
                    logger.error("Error: Last brace found before or at first brace. Cannot extract JSON.")
                    raise ValueError("Failed to extract JSON: Invalid brace positions.")
            except ValueError:
                logger.error(f"Error: Could not find opening or closing braces for JSON extraction. Raw output (first 500 chars): {raw_llm_output_str[:500]}...", exc_info=True)
                raise ValueError("Failed to extract JSON from LLM response for InvestmentAssessment (braces not found).")

        if not json_to_parse:
            logger.error(f"Error: JSON string to parse is empty after extraction attempts. Raw output (first 500 chars): {raw_llm_output_str[:500]}...")
            raise ValueError("Failed to extract JSON: Resulting string is empty.")
        
        assessment = investment_assessment_parser.parse(json_to_parse)
        recommendation = calculate_recommendation(assessment)

        # Calculate confidence score (simplified example from reference)
        confidence = 0.0
        if company_profile: confidence += 0.2
        if parsed_founder_profiles: confidence += 0.2 # If any founder data was successfully parsed
        if parsed_market_analysis: confidence += 0.1
        # Based on assessment completeness (Flight Story criteria)
        fs_criteria_fields = [
            assessment.fs_focus_industry_fit, assessment.fs_mission_alignment, assessment.fs_exciting_solution_to_problem,
            assessment.fs_founded_something_relevant_before, assessment.fs_impressive_relevant_past_experience,
            assessment.fs_exceptionally_smart_or_strategic
        ]
        assessed_criteria_count = sum(1 for f in fs_criteria_fields if f is not None)
        if assessed_criteria_count > 0:
             confidence += (assessed_criteria_count / len(fs_criteria_fields)) * 0.3 # Max 0.3 for criteria assessment
        if recommendation == "PASS": confidence += 0.2
        elif recommendation == "EXPLORE": confidence += 0.1
        confidence = min(round(confidence, 2), 1.0)

        investment_research = InvestmentResearch(
            query_or_target_entity=company_profile.company_name,
            primary_analyst="AI Investment Analyst (Decision Support Agent)",
            company_profile=company_profile,
            founder_profiles=parsed_founder_profiles,
            market_analysis=parsed_market_analysis,
            investment_assessment=assessment,
            overall_summary_and_recommendation=recommendation,
            confidence_score_overall=confidence,
            status="Complete",
            sources_consulted=[
                {"type": "structured_input", "source": "company_profile_data_input"},
                {"type": "structured_input", "source": "founder_profiles_data_input"},
                {"type": "structured_input", "source": "market_analysis_data_input"},
                {"type": "structured_input", "source": "additional_context_input"}
            ],
            timestamp=datetime.utcnow().isoformat()
        )
        
        logger.info(f"Decision support analysis completed. Recommendation: {recommendation}, Confidence: {confidence}")
        return investment_research

    except Exception as e:
        logger.error(f"Critical error during investment decision analysis: {e}", exc_info=True)
        # Fallback InvestmentResearch object
        company_name_for_error = "Unknown"
        if company_data_dict and isinstance(company_data_dict, dict): # Check if dict before access
            company_name_for_error = company_data_dict.get("company_name", "Unknown")

        return InvestmentResearch(
            query_or_target_entity=company_name_for_error,
            primary_analyst="AI System Error",
            status="Error",
            overall_summary_and_recommendation="ERROR_ANALYSIS_FAILED",
            investment_assessment=InvestmentAssessment(overall_criteria_summary=f"Analysis failed due to: {type(e).__name__}"),
            error_message=f"Unhandled exception in decision support: {str(e)}",
            timestamp=datetime.utcnow().isoformat()
        )

def generate_investment_recommendation(
    founder_profile_data: Dict[str, Any],
    company_profile_data: Dict[str, Any],
    market_analysis_data: Dict[str, Any]
) -> Dict[str, Any]:
    logger.warning("Deprecated function 'generate_investment_recommendation' called. The main logic is in 'run_decision_support_analysis'.")
    # ... (rest of the old simple logic) ...
    return {"recommendation": "DEPRECATED_FUNCTION", "confidence_score": 0.0, "reasoning": "This function is deprecated."}

def main():
    logger.info("Decision support agent CLI starting...")
    parser = argparse.ArgumentParser(description="Investment Decision Support Agent CLI")
    parser.add_argument("--company_file", type=str, required=True, help="Path to the company profile JSON file.")
    # Allow multiple founder files or a single file containing a list
    parser.add_argument("--founder_files", type=str, nargs='*', help="Path(s) to founder profile JSON file(s). Each file can be a single profile or a list of profiles.")
    parser.add_argument("--market_file", type=str, help="Path to the market analysis JSON file.")
    parser.add_argument("--additional_context_file", type=str, help="Path to an optional JSON file for additional context (e.g., pitch deck summary).")
    parser.add_argument("--output_file", type=str, help="Optional path to save the full InvestmentResearch JSON output.")

    args = parser.parse_args()

    company_profile_obj, founder_profiles_list, market_analysis_obj, additional_context_dict = load_research_data(
        args.company_file,
        args.founder_files,
        args.market_file,
        args.additional_context_file
    )

    # Convert Pydantic models back to dicts for run_decision_support_analysis, or adapt run_decision_support_analysis to take models
    # For now, converting back to dicts to match current run_decision_support_analysis signature
    company_data_for_analysis = company_profile_obj.model_dump(mode='json') if company_profile_obj else None
    founders_data_for_analysis = [fp.model_dump(mode='json') for fp in founder_profiles_list] if founder_profiles_list else []
    market_data_for_analysis = market_analysis_obj.model_dump(mode='json') if market_analysis_obj else None
    
    if not company_data_for_analysis:
        logger.critical("Company data could not be loaded or is invalid. Exiting.")
        error_output = {"status": "error", "error_message": "Failed to load valid company data from file."}
        print(json.dumps(error_output, indent=None), file=sys.stdout)
        sys.exit(1)

    # Call the main analysis function
    # The llm and model_config will be injected by the decorator
    investment_research_result = run_decision_support_analysis(
        company_data_dict=company_data_for_analysis,
        founder_data_list=founders_data_for_analysis,
        market_data_dict=market_data_for_analysis,
        additional_context=additional_context_dict # This is already a dict
    )

    if investment_research_result:
        # Output the full InvestmentResearch object as JSON to stdout for the orchestrator
        # The orchestrator expects a single JSON object from stdout.
        print(investment_research_result.model_dump_json(indent=None))
        logger.info("Decision support agent CLI: Successfully generated InvestmentResearch JSON output to stdout.")

        if args.output_file:
            try:
                with open(args.output_file, 'w') as f:
                    f.write(investment_research_result.model_dump_json(indent=2))
                logger.info(f"Full InvestmentResearch output also saved to: {args.output_file}")
            except Exception as e:
                logger.error(f"Error saving full InvestmentResearch to {args.output_file}: {e}")
    else:
        # run_decision_support_analysis should ideally return a InvestmentResearch object even on error
        # but as a fallback:
        error_output = {"status": "error", "error_message": "Decision support analysis failed to produce a result."}
        print(json.dumps(error_output, indent=None), file=sys.stdout)
        logger.error("Decision support agent CLI: Analysis failed to produce a result.")
        sys.exit(1)

if __name__ == "__main__":
    main()