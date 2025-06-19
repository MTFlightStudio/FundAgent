import os
import json
import re
import logging # Added
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import argparse
import sys
from urllib.parse import urlparse

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor

# Import models and tools from your project structure
from ai_agents.models.investment_research import CompanyProfile
from ai_agents.tools import search_tool, save_tool 
from ai_agents.tools import tavily_search_tool_instance

# Import the new model selection and retry systems
try:
    from ai_agents.config.model_config import get_llm_for_agent, ModelConfig, ModelSelectionError
    from ai_agents.utils.retry_handler import with_smart_retry, RetryConfig
    MODEL_SYSTEM_AVAILABLE = True
except ImportError:
    MODEL_SYSTEM_AVAILABLE = False
    logging.warning("CompanyResearchAgent: Model selection or retry system not available. Functionality may be limited or use fallbacks.")


# Load .env file from the project root
load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)

# --- Output Parser for CompanyProfile ---
company_profile_parser = PydanticOutputParser(pydantic_object=CompanyProfile)

# --- Prompt Template for Company Research ---
COMPANY_RESEARCH_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a specialized company research AI. Your primary goal is to gather comprehensive information about a given company and structure it into a JSON object adhering to the CompanyProfile schema provided below.\n"
            "You MUST use the 'search_tool' to find the required information. Make multiple targeted search queries if necessary to cover all aspects of the CompanyProfile model.\n"
            "Focus on finding:\n"
            "- Official company website and LinkedIn profile URL.\n"
            "- A clear description of the company, its industry, and founding year.\n"
            "- Detailed funding history: total raised, stages, key investors, and dates. Be precise with amounts and dates.\n"
            "- Key products/services, business model, and target customers.\n"
            "- Team size (e.g., '11-50 employees') and headquarters location.\n"
            "- Recent news, press releases, and any available key metrics (e.g., ARR, user growth).\n"
            "- Company mission statement if available.\n\n"
            "After gathering information using the search_tool, your FINAL output MUST be a single, valid JSON object that strictly conforms to the following CompanyProfile schema. Do NOT include any other text, explanations, or markdown outside of this JSON object.\n"
            "If specific information cannot be found for a field, set its value to null or omit it if it's an optional field in the schema, but try your best to populate all fields.\n"
            "Prioritize recent and reliable sources (e.g., official company site, reputable news outlets, Crunchbase, PitchBook summaries found via search).\n"
            "Schema:\n{format_instructions}"
        ),
        ("human", "Please research the company: {input}"), # 'input' will be the company name
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
).partial(format_instructions=company_profile_parser.get_format_instructions())

# Enhanced prompt template for when HubSpot data is available
COMPANY_RESEARCH_WITH_HUBSPOT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a specialized company research AI. Your primary goal is to gather comprehensive information about a given company and structure it into a JSON object adhering to the CompanyProfile schema provided below.\n"
            "You have been provided with VERIFIED FINANCIAL AND BUSINESS DATA from HubSpot CRM which should be PRIORITIZED over any conflicting information found via web search.\n"
            "Use the 'search_tool' to find additional information not provided in the HubSpot data, but ALWAYS prioritize the HubSpot data for:\n"
            "- Financial metrics (revenue, funding amounts, valuation, team size)\n"
            "- Business description, sector, and stage\n"
            "- Founder information and team details\n"
            "- Specific business details like USP, customer base, etc.\n\n"
            "When searching, use the HubSpot context to make more targeted queries. For example, if you know the sector and location, search for 'CompanyName health tech London' rather than just 'CompanyName' to avoid finding wrong companies with similar names.\n\n"
            "HubSpot Data Context:\n{hubspot_context}\n\n"
            "Focus on finding via search:\n"
            "- Official company website and LinkedIn profile URL (if not in HubSpot data)\n"
            "- Founding year and detailed company history\n"
            "- External funding history from investors (beyond what's in HubSpot)\n"
            "- Recent news, press releases, and market position\n"
            "- Detailed product/service information and competitive landscape\n"
            "- Any additional key metrics not captured in HubSpot\n\n"
            "After gathering information using the search_tool, your FINAL output MUST be a single, valid JSON object that strictly conforms to the following CompanyProfile schema. Do NOT include any other text, explanations, or markdown outside of this JSON object.\n"
            "CRITICAL: For any financial fields (revenue, funding, team_size, etc.), use the HubSpot data as the primary source and only supplement with web search if HubSpot data is missing.\n"
            "Schema:\n{format_instructions}"
        ),
        ("human", "Please research the company: {input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
).partial(format_instructions=company_profile_parser.get_format_instructions())

def _ensure_url_scheme(url: Optional[str]) -> Optional[str]:
    """Prepends https:// to a URL if it's missing a scheme."""
    if url and not urlparse(url).scheme and (url.startswith("www.") or "." in url.split("/")[0]):
        return f"https://{url}"
    return url

def _get_fallback_llm():
    """Fallback LLM selection logic (simplified)."""
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        logger.info("CompanyResearchAgent: Using OpenAI GPT model (fallback)")
        return ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.1, max_tokens=4000)
    elif os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        logger.info("CompanyResearchAgent: Using Anthropic Claude model (fallback)")
        return ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.2, max_tokens=4000)
    else:
        logger.error("CompanyResearchAgent: FATAL: No LLM API keys available for fallback.")
        raise ValueError("No LLM API keys available for fallback LLM initialization for Company Research.")

# Define the core research logic that can be decorated
def _core_company_research_logic(company_name: str, llm: Any, model_config: Optional[ModelConfig], hubspot_data: Optional[Dict[str, Any]] = None) -> CompanyProfile:
    """
    The core logic for company research, designed to be wrapped by the retry decorator.
    It expects an LLM instance to be passed in, and optionally HubSpot data for enhanced context.
    """
    logger.info(f"CompanyResearchAgent: Executing core research logic for \"{company_name}\" with model {model_config.model_name if model_config else 'Unknown'}.")
    
    if hubspot_data:
        logger.info(f"CompanyResearchAgent: Using HubSpot data for enhanced research context for \"{company_name}\".")

    if not tavily_search_tool_instance:
        logger.error("CompanyResearchAgent: FATAL: Tavily search_tool is not available. Cannot proceed.")
        # This should ideally be caught before even calling this core logic,
        # but as a safeguard within the core execution path.
        raise ValueError("Tavily search_tool not available for Company Research.")

    available_tools = [search_tool]

    # Choose prompt template based on whether we have HubSpot data
    if hubspot_data:
        hubspot_context = _extract_hubspot_context(hubspot_data)
        company_research_agent_prompt = COMPANY_RESEARCH_WITH_HUBSPOT_PROMPT_TEMPLATE.partial(hubspot_context=hubspot_context)
        logger.debug(f"CompanyResearchAgent: Using enhanced prompt with HubSpot context: {hubspot_context[:200]}...")
    else:
        company_research_agent_prompt = COMPANY_RESEARCH_PROMPT_TEMPLATE
        logger.debug(f"CompanyResearchAgent: Using standard prompt template (no HubSpot data).")

    agent = create_tool_calling_agent(
        llm=llm, 
        prompt=company_research_agent_prompt,
        tools=available_tools
    )
    agent_executor = AgentExecutor(
        agent=agent,
        tools=available_tools,
        verbose=False, 
        handle_parsing_errors=True,
        max_iterations=8 # Sensible default
    )

    raw_agent_output_container = None
    json_to_parse = ""
    try:
        # Create enhanced input with HubSpot context if available
        if hubspot_data:
            # Use HubSpot context to create a more targeted search input
            companies = hubspot_data.get('associated_companies', [])
            if companies:
                company_props = companies[0].get('properties', {})
                sector = company_props.get('what_sector_is_your_business_product_', '')
                location = company_props.get('where_is_your_business_based_', '')
                # Create enhanced search query
                search_input = f"{company_name}"
                if sector:
                    search_input += f" {sector}"
                if location:
                    search_input += f" {location}"
                search_input += " company"
            else:
                search_input = company_name
        else:
            search_input = company_name
        
        logger.debug(f"CompanyResearchAgent: Search input: '{search_input}'")
        raw_agent_output_container = agent_executor.invoke({"input": search_input})

        llm_response_content_str = None
        if isinstance(raw_agent_output_container, dict) and "output" in raw_agent_output_container:
            output_payload = raw_agent_output_container["output"]
            if isinstance(output_payload, str):
                llm_response_content_str = output_payload
            elif isinstance(output_payload, list) and output_payload:
                first_block = output_payload[0]
                if isinstance(first_block, dict) and "text" in first_block and isinstance(first_block["text"], str):
                    llm_response_content_str = first_block["text"]
                else:
                    llm_response_content_str = str(output_payload)
            else:
                llm_response_content_str = str(output_payload)
        elif isinstance(raw_agent_output_container, str):
             llm_response_content_str = raw_agent_output_container
        else:
            llm_response_content_str = str(raw_agent_output_container)

        if not llm_response_content_str:
            logger.error(f"CompanyResearchAgent: Agent for \"{company_name}\" did not produce a parsable output string.")
            raise ValueError(f"Agent for {company_name} produced no parsable output string for CompanyProfile.")

        logger.debug(f"CompanyResearchAgent: Raw LLM content for \"{company_name}\": {llm_response_content_str[:500]}...")

        # Enhanced JSON extraction with better error handling
        json_match = re.search(r"\{[\s\S]*\}", llm_response_content_str)
        if json_match:
            json_to_parse = json_match.group(0).strip()
        else:
            temp_str = llm_response_content_str.strip()
            if temp_str.startswith("```json"):
                temp_str = temp_str[len("```json"):].strip()
            if temp_str.endswith("```"):
                temp_str = temp_str[:-len("```")].strip()
            if temp_str.startswith("{") and temp_str.endswith("}"):
                json_to_parse = temp_str
                logger.warning(f"CompanyResearchAgent: Used fallback JSON extraction for \"{company_name}\".")
            else:
                cleaned_further = re.sub(r"<[^>]+>", "", temp_str).strip()
                if cleaned_further.startswith("{") and cleaned_further.endswith("}"):
                    json_to_parse = cleaned_further
                    logger.warning(f"CompanyResearchAgent: Used fallback JSON extraction (XML tag stripping) for \"{company_name}\".")
                else:
                    logger.error(f"CompanyResearchAgent: Could not extract JSON from LLM response for \"{company_name}\": {llm_response_content_str[:500]}...")
                    raise ValueError(f"Could not extract valid JSON for {company_name} from LLM response.")
        
        # Validate JSON before parsing - check for truncation
        if not json_to_parse.endswith('}'):
            logger.warning(f"CompanyResearchAgent: JSON appears truncated for \"{company_name}\". Attempting to fix...")
            # Try to find the last complete field and close the JSON
            lines = json_to_parse.split('\n')
            fixed_lines = []
            for line in lines:
                if ':' in line or line.strip() in ['{', '}', '[', ']']:
                    fixed_lines.append(line)
                else:
                    break  # Stop at the first incomplete line
            
            # Ensure proper closing
            fixed_json = '\n'.join(fixed_lines)
            if not fixed_json.endswith('}'):
                fixed_json += '\n}'
            json_to_parse = fixed_json
            logger.info(f"CompanyResearchAgent: Attempted to fix truncated JSON for \"{company_name}\".")
        
        logger.debug(f"CompanyResearchAgent: Extracted JSON for \"{company_name}\": {json_to_parse[:300]}...")
        
        try:
            parsed_dict = json.loads(json_to_parse, strict=False)
        except json.JSONDecodeError as e:
            logger.error(f"CompanyResearchAgent: JSON parsing failed for \"{company_name}\". Error: {e}")
            logger.error(f"CompanyResearchAgent: Problematic JSON: {json_to_parse}")
            # Try one more time with a simpler approach - extract just core fields
            logger.warning(f"CompanyResearchAgent: Attempting minimal JSON extraction for \"{company_name}\"...")
            minimal_json = {
                "company_name": company_name,
                "description": "Research failed - using minimal data",
                "industry": "Unknown"
            }
            parsed_dict = minimal_json
            logger.info(f"CompanyResearchAgent: Using minimal fallback data for \"{company_name}\".")

        for url_field in ["website", "linkedin_url"]:
            if url_field in parsed_dict and isinstance(parsed_dict[url_field], str):
                parsed_dict[url_field] = _ensure_url_scheme(parsed_dict[url_field])
        
        # Fix business_model if it's a dict before any further processing
        if 'business_model' in parsed_dict and isinstance(parsed_dict['business_model'], dict):
            logger.warning(f"CompanyResearchAgent: Converting business_model from dict to string for \"{company_name}\".")
            bm_dict = parsed_dict['business_model']
            if 'description' in bm_dict:
                parsed_dict['business_model'] = bm_dict['description']
            else:
                # Join all values in the dict
                parsed_dict['business_model'] = '. '.join(str(v) for v in bm_dict.values() if v)
        
        # Merge with HubSpot data if available (prioritizing HubSpot financials)
        if hubspot_data:
            logger.info(f"CompanyResearchAgent: Merging web search results with HubSpot data for \"{company_name}\".")
            try:
                parsed_dict = _merge_hubspot_and_web_data(parsed_dict, hubspot_data)
            except Exception as merge_error:
                logger.error(f"CompanyResearchAgent: Error merging HubSpot data for \"{company_name}\": {merge_error}")
                logger.warning(f"CompanyResearchAgent: Continuing with web-only data for \"{company_name}\".")
        
        # Final check to ensure business_model is a string after merge
        if 'business_model' in parsed_dict and isinstance(parsed_dict['business_model'], dict):
            logger.warning(f"CompanyResearchAgent: Final conversion of business_model from dict to string for \"{company_name}\".")
            bm_dict = parsed_dict['business_model']
            if 'description' in bm_dict:
                parsed_dict['business_model'] = bm_dict['description']
            else:
                parsed_dict['business_model'] = '. '.join(str(v) for v in bm_dict.values() if v)
        
        company_profile_obj = CompanyProfile.model_validate(parsed_dict)
        logger.info(f"CompanyResearchAgent: Successfully researched and validated profile for company: {company_name}")
        return company_profile_obj

    except json.JSONDecodeError as json_err:
        logger.error(f"CompanyResearchAgent: JSONDecodeError for \"{company_name}\". Error: {json_err}. String: {json_to_parse[:1000]}", exc_info=True)
        if raw_agent_output_container: logger.debug(f"Raw agent output at time of JSON error: {raw_agent_output_container}")
        raise # Re-raise for retry handler
    except Exception as e:
        logger.error(f"CompanyResearchAgent: Error during core research for \"{company_name}\": {e}", exc_info=True)
        if raw_agent_output_container: logger.debug(f"Raw agent output at time of error: {raw_agent_output_container}")
        raise # Re-raise for retry handler

# --- Apply Decorator ---
_decorated_core_research_logic = _core_company_research_logic

if MODEL_SYSTEM_AVAILABLE:
    _model_selector_for_retry = None
    try:
        from ai_agents.config.model_config import get_llm_for_agent as get_llm_func_for_decorator
        _model_selector_for_retry = get_llm_func_for_decorator
    except ImportError:
        logger.warning("CompanyResearchAgent: Could not import get_llm_for_agent for retry decorator. Model switching disabled.")

    _retry_config = RetryConfig(
        max_retries=3, 
        base_delay=2.0, 
        switch_model_on_rate_limit=bool(_model_selector_for_retry) # Only True if selector is available
    )
    
    _decorated_core_research_logic = with_smart_retry(
        agent_name="company_research",
        retry_config_override=_retry_config,
        model_selector_func=_model_selector_for_retry
    )(_core_company_research_logic)
else:
    logger.warning("CompanyResearchAgent: Retry system not fully available. Research will not have advanced retry/model switching.")
    # _decorated_core_research_logic remains the original undecorated function if no retry system

def _extract_hubspot_context(hubspot_data: Dict[str, Any]) -> str:
    """Extract key context from HubSpot data to help the agent make better search queries"""
    if not hubspot_data:
        return "No HubSpot data available."
    
    context_parts = []
    
    # Company information
    companies = hubspot_data.get('associated_companies', [])
    if companies:
        company_props = companies[0].get('properties', {})
        
        # Basic company info
        company_name = company_props.get('name')
        if company_name:
            context_parts.append(f"Company Name: {company_name}")
        
        # Business description and sector
        business_desc = company_props.get('describe_the_business_product_in_one_sentence')
        if business_desc:
            context_parts.append(f"Business Description: {business_desc}")
        
        sector = company_props.get('what_sector_is_your_business_product_')
        if sector:
            context_parts.append(f"Sector: {sector}")
        
        # Location
        location = company_props.get('where_is_your_business_based_')
        if location:
            context_parts.append(f"Location: {location}")
        
        # Stage
        stage = company_props.get('what_best_describes_your_stage_of_business_')
        if stage:
            context_parts.append(f"Business Stage: {stage}")
        
        # Financial data (THIS IS THE KEY PRIORITY DATA)
        ltm_revenue = company_props.get('what_is_your_ltm__last_12_months__revenue_')
        if ltm_revenue:
            context_parts.append(f"LTM Revenue: {ltm_revenue}")
        
        monthly_revenue = company_props.get('what_is_your_current_monthly_revenue_')
        if monthly_revenue:
            context_parts.append(f"Monthly Revenue: {monthly_revenue}")
        
        raising_amount = company_props.get('how_much_are_you_raising_at_this_stage_')
        if raising_amount:
            context_parts.append(f"Current Raise Amount: {raising_amount}")
        
        valuation = company_props.get('what_valuation_are_you_raising_at_')
        if valuation:
            context_parts.append(f"Valuation: {valuation}")
        
        prior_funding = company_props.get('how_much_have_you_raised_prior_to_this_round_')
        if prior_funding:
            context_parts.append(f"Prior Funding: {prior_funding}")
        
        equity_split = company_props.get('how_much_of_the_equity_do_you_your_team_have_')
        if equity_split:
            context_parts.append(f"Team Equity: {equity_split}")
        
        team_size = company_props.get('how_many_employees_do_you_have__full_time_equivalents_')
        if team_size:
            context_parts.append(f"Team Size: {team_size}")
        
        # USP and differentiation
        usp = company_props.get('what_is_your_usp__what_makes_you_different_from_your_competitors_')
        if usp:
            context_parts.append(f"USP: {usp}")
        
        # Website (if available)
        website = company_props.get('website')
        if website:
            context_parts.append(f"Website: {website}")
    
    # Contact/founder information
    contacts = hubspot_data.get('associated_contacts', [])
    if contacts:
        founder_names = []
        for contact in contacts:
            props = contact.get('properties', {})
            first_name = props.get('firstname', '')
            last_name = props.get('lastname', '')
            if first_name and last_name:
                founder_names.append(f"{first_name} {last_name}")
        
        if founder_names:
            context_parts.append(f"Key Founders: {', '.join(founder_names)}")
    
    return "\n".join(context_parts) if context_parts else "Limited HubSpot data available."

def _merge_hubspot_and_web_data(web_result: Dict[str, Any], hubspot_data: Dict[str, Any]) -> Dict[str, Any]:
    """Merge web search results with HubSpot data, prioritizing HubSpot for financial and verified data"""
    if not hubspot_data or not hubspot_data.get('associated_companies'):
        return web_result
    
    # Start with web search result
    merged_result = web_result.copy()
    
    # Extract HubSpot company data
    company_props = hubspot_data['associated_companies'][0].get('properties', {})
    
    # Extract HubSpot contact data (founders)
    contacts = hubspot_data.get('associated_contacts', [])
    founder_info = []
    founder_linkedin_urls = []
    
    for contact in contacts:
        contact_props = contact.get('properties', {})
        
        # Extract founder name
        firstname = contact_props.get('firstname', '')
        lastname = contact_props.get('lastname', '')
        if firstname or lastname:
            full_name = f"{firstname} {lastname}".strip()
            if full_name:
                founder_info.append(full_name)
        
        # Extract founder LinkedIn URL
        linkedin_url = contact_props.get('hs_linkedin_url')
        if linkedin_url:
            founder_linkedin_urls.append(linkedin_url)
        
        # Extract email for additional context
        email = contact_props.get('email')
        if email and full_name:
            founder_info[-1] = f"{full_name} ({email})"
    
    # Override with HubSpot data where available (prioritizing financial accuracy)
    
    # Financial data - ALWAYS prioritize HubSpot
    ltm_revenue = company_props.get('what_is_your_ltm__last_12_months__revenue_')
    if ltm_revenue:
        # Ensure we have a funding_history structure
        if 'funding_history' not in merged_result:
            merged_result['funding_history'] = {}
        if 'current_metrics' not in merged_result['funding_history']:
            merged_result['funding_history']['current_metrics'] = {}
        merged_result['funding_history']['current_metrics']['annual_revenue'] = ltm_revenue
    
    monthly_revenue = company_props.get('what_is_your_current_monthly_revenue_')
    if monthly_revenue:
        if 'funding_history' not in merged_result:
            merged_result['funding_history'] = {}
        if 'current_metrics' not in merged_result['funding_history']:
            merged_result['funding_history']['current_metrics'] = {}
        merged_result['funding_history']['current_metrics']['monthly_revenue'] = monthly_revenue
    
    # Team size
    team_size = company_props.get('how_many_employees_do_you_have__full_time_equivalents_')
    if team_size:
        merged_result['team_size'] = team_size
    
    # Business description - prefer HubSpot if more detailed
    business_desc = company_props.get('describe_the_business_product_in_one_sentence')
    if business_desc and len(business_desc) > 20:  # Only if substantial
        merged_result['description'] = business_desc
    
    # Industry/sector
    sector = company_props.get('what_sector_is_your_business_product_')
    if sector:
        merged_result['industry'] = sector
    
    # Location/headquarters
    location = company_props.get('where_is_your_business_based_')
    if location:
        merged_result['headquarters'] = location
    
    # Founder LinkedIn URL - prioritize HubSpot contact data
    if founder_linkedin_urls:
        # Use the first founder's LinkedIn URL as the primary company LinkedIn
        merged_result['linkedin_url'] = founder_linkedin_urls[0]
    
    # Add founder information to key_metrics or a dedicated field
    if founder_info:
        if 'key_metrics' not in merged_result:
            merged_result['key_metrics'] = {}
        merged_result['key_metrics']['Founders'] = ', '.join(founder_info)
    
    # Store all founder LinkedIn URLs for potential use
    if founder_linkedin_urls:
        merged_result['founder_linkedin_urls'] = founder_linkedin_urls
    
    # USP and business model details - Keep as string to match schema
    usp = company_props.get('what_is_your_usp__what_makes_you_different_from_your_competitors_')
    if usp:
        # business_model should be a string according to CompanyProfile schema
        existing_bm = merged_result.get('business_model', '')
        if isinstance(existing_bm, dict):
            # Convert existing dict to string
            existing_bm = existing_bm.get('description', str(existing_bm))
        
        # Combine existing business model with USP
        if existing_bm:
            merged_result['business_model'] = f"{existing_bm}. USP: {usp}"
        else:
            merged_result['business_model'] = f"USP: {usp}"
    
    # Current fundraising details
    raising_amount = company_props.get('how_much_are_you_raising_at_this_stage_')
    valuation = company_props.get('what_valuation_are_you_raising_at_')
    prior_funding = company_props.get('how_much_have_you_raised_prior_to_this_round_')
    
    if raising_amount or valuation or prior_funding:
        if 'funding_history' not in merged_result:
            merged_result['funding_history'] = {}
        if 'current_round' not in merged_result['funding_history']:
            merged_result['funding_history']['current_round'] = {}
        
        if raising_amount:
            merged_result['funding_history']['current_round']['amount_raising'] = raising_amount
        if valuation:
            merged_result['funding_history']['current_round']['valuation'] = valuation
        if prior_funding:
            merged_result['funding_history']['total_raised'] = prior_funding
    
    # Additional HubSpot-specific fields that don't fit standard CompanyProfile schema
    # Store these in key_metrics or as separate fields for the UI to display
    
    # Customer base information
    customer_base = company_props.get('what_best_describes_your_customer_base_')
    if customer_base:
        merged_result['target_customer'] = customer_base
    
    # UN SDG Goals
    un_sdg_goals = company_props.get('which__if_any__of_the_un_sdg_17_goals_does_your_business_address_')
    if un_sdg_goals:
        merged_result['un_sdg_goals'] = un_sdg_goals
    
    # Business stage
    business_stage = company_props.get('what_best_describes_your_stage_of_business_')
    if business_stage:
        merged_result['funding_stage'] = business_stage
    
    # Health/happiness contribution
    health_contribution = company_props.get('how_does_your_product_contribute_to_a_healthier__happier_whole_human_experience_')
    if not health_contribution:
        health_contribution = company_props.get('does_your_product_contribute_to_a_healthier__happier_whole_human_experience_')
    if health_contribution:
        merged_result['health_happiness_contribution'] = health_contribution
    
    # Innovation and technology use
    innovation_use = company_props.get('how_does_your_company_use_innovation__through_technology_or_to_differentiate_the_business_model__')
    if innovation_use:
        merged_result['innovation_use'] = innovation_use
    
    # Equity and ownership information
    equity_split = company_props.get('how_much_of_the_equity_do_you_your_team_have_')
    if equity_split:
        # Add to key_metrics for display
        if 'key_metrics' not in merged_result:
            merged_result['key_metrics'] = {}
        merged_result['key_metrics']['Team Equity'] = f"{equity_split}%"
    
    # Partnership goals and Flight-specific information
    partnership_goals = company_props.get('what_is_it_that_you_re_looking_for_with_a_partnership_from_flight_')
    partnership_expansion = company_props.get('please_expand')
    
    if partnership_goals or partnership_expansion:
        partnership_info = []
        if partnership_goals:
            partnership_info.append(f"Partnership Goals: {partnership_goals}")
        if partnership_expansion:
            partnership_info.append(f"Additional Details: {partnership_expansion}")
        
        merged_result['partnership_objectives'] = "; ".join(partnership_info)
    
    # Add any additional key metrics from HubSpot
    if 'key_metrics' not in merged_result:
        merged_result['key_metrics'] = {}
    
    # Add business stage to key metrics if available
    if business_stage:
        merged_result['key_metrics']['Business Stage'] = business_stage
    
    return merged_result

def run_company_research_cli(company_name: str, hubspot_data: Optional[Dict[str, Any]] = None) -> Optional[CompanyProfile]:
    """
    Sets up and runs the company research, utilizing the (potentially) decorated core logic.
    Now accepts optional HubSpot data for enhanced context and financial accuracy.
    """
    logger.info(f"CompanyResearchAgent: Starting comprehensive research for company: \"{company_name}\" via CLI entry point.")

    if not tavily_search_tool_instance: # Initial check before involving LLMs or retries
        logger.error("CompanyResearchAgent: FATAL: Tavily search_tool is not available (check TAVILY_API_KEY in .env). Cannot proceed.")
        return None

    # Initial LLM selection for the first attempt (or if no retry system)
    # The decorator will handle subsequent selections if it's active and configured for model switching.
    llm_instance = None
    model_config_instance = None

    if MODEL_SYSTEM_AVAILABLE:
        try:
            llm_instance, model_config_instance = get_llm_for_agent("company_research")
            # get_llm_for_agent already logs the selection.
            # logger.info(f"CompanyResearchAgent: Initial model for '{company_name}': {model_config_instance.model_name}")
        except ModelSelectionError as e_mse:
            logger.error(f"CompanyResearchAgent: FATAL - Initial model selection failed for '{company_name}': {e_mse}. Attempting fallback.")
            try:
                llm_instance = _get_fallback_llm()
                # We don't have a full ModelConfig for fallback LLMs here, but the LLM object itself is key.
                model_config_instance = ModelConfig(provider="Unknown", model_name="fallback", temperature=0.1, max_tokens=4000, cost_per_1k_tokens=0, rate_limit_rpm=0, best_for=[]) 
            except ValueError as e_fb: # Fallback also failed
                 logger.error(f"CompanyResearchAgent: FATAL - Fallback LLM also failed for '{company_name}': {e_fb}")
                 return None

        except Exception as e_llm_init:
            logger.error(f"CompanyResearchAgent: FATAL - Unexpected error during initial LLM init for '{company_name}': {e_llm_init}")
            return None
    else: # No model system, try direct fallback
        try:
            llm_instance = _get_fallback_llm()
            model_config_instance = ModelConfig(provider="Unknown", model_name="fallback_direct", temperature=0.1, max_tokens=4000, cost_per_1k_tokens=0, rate_limit_rpm=0, best_for=[])
        except ValueError as e_fb_direct:
            logger.error(f"CompanyResearchAgent: FATAL - Fallback LLM failed (no model system): {e_fb_direct}")
            return None


    if not llm_instance:
        logger.error(f"CompanyResearchAgent: FATAL - LLM instance is None after all selection attempts for '{company_name}'. Cannot proceed.")
        return None

    try:
        # Call the (potentially) decorated core research logic with HubSpot data
        # Pass the initially selected LLM and its config. The decorator will use these for the first attempt
        # and can switch them for subsequent attempts if `model_selector_func` is provided to it.
        company_profile_obj = _decorated_core_research_logic(
            company_name=company_name, 
            llm=llm_instance, # Pass the LLM instance
            model_config=model_config_instance, # Pass its config
            hubspot_data=hubspot_data # Pass HubSpot data if available
        )

        if company_profile_obj:
            # Save the structured data
            safe_company_name = "".join(c if c.isalnum() else "_" for c in company_name)[:50].rstrip("_")
            output_filename = f"company_research_{safe_company_name}.json"
            try:
                save_tool.invoke({
                    "filename": output_filename,
                    "text": company_profile_obj.model_dump_json(indent=2)
                })
                logger.info(f"CompanyResearchAgent: Profile for \"{company_name}\" saved to {output_filename}")
            except Exception as e_save:
                logger.error(f"CompanyResearchAgent: Error saving profile for \"{company_name}\" to file: {e_save}")
        return company_profile_obj

    except Exception as e: # This will catch errors if retries are exhausted or if the error is non-retryable
        logger.error(f"CompanyResearchAgent: Research for \"{company_name}\" ultimately failed after all attempts: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    # Ensure logging is set up for CLI execution.
    if not logging.getLogger().hasHandlers(): # Check if handlers are already configured
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    logger.info("Company research agent CLI starting...")
    
    parser = argparse.ArgumentParser(description="Company Research Agent CLI - Researches a company by name.")
    parser.add_argument("--company_name", type=str, required=True, help="Name of the company to research.")
    
    args = parser.parse_args()

    try:
        profile = run_company_research_cli(args.company_name)
        
        if profile:
            print(profile.model_dump_json(indent=None)) 
            logger.info(f"Company research agent: Successfully generated JSON output for {args.company_name}")
        else:
            error_output = {
                "status": "error",
                "company_name": args.company_name,
                "error_message": f"Failed to retrieve or generate profile for {args.company_name} after all attempts."
            }
            print(json.dumps(error_output, indent=None))
            logger.error(f"Company research agent: Failed to generate profile for {args.company_name}.")
            sys.exit(1)
            
    except Exception as e_main:
        error_output = {
            "status": "error",
            "company_name": args.company_name,
            "error_message": f"Critical error in company_research_agent CLI for {args.company_name}: {str(e_main)}"
        }
        print(json.dumps(error_output, indent=None))
        logger.critical(f"Critical error in company_research_agent CLI for {args.company_name}: {e_main}", exc_info=True)
        sys.exit(1) 