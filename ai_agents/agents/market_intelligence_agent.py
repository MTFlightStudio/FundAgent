import os
import json
import re
import logging
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import argparse
import sys

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor

# Import models and tools from your project structure
from ai_agents.models.investment_research import MarketAnalysis, CompetitorInfo
from ai_agents.tools import search_tool, wiki_tool, save_tool
from ai_agents.tools import tavily_search_tool_instance, wiki_tool_instance

# Import the new model selection and retry systems
try:
    from ai_agents.config.model_config import get_llm_for_agent, ModelConfig, ModelSelectionError
    from ai_agents.utils.retry_handler import with_smart_retry, RetryConfig
    MODEL_SYSTEM_AVAILABLE = True
except ImportError:
    MODEL_SYSTEM_AVAILABLE = False
    logging.warning("MarketIntelligenceAgent: Model selection or retry system not available. Functionality will be limited.")

# Load .env file from the project root
load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)

def _define_specific_market_from_hubspot_data(hubspot_data: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """
    Analyzes HubSpot company data to define a specific, targeted market rather than generic industry sectors.
    
    Returns:
        Dict with 'market_focus', 'search_terms', 'geographic_scope', and 'reasoning'
    """
    if not hubspot_data or not hubspot_data.get('associated_companies'):
        return {
            'market_focus': 'Unknown Market',
            'search_terms': ['technology market'],
            'geographic_scope': 'Global',
            'reasoning': 'No HubSpot data available for market definition'
        }
    
    company_props = hubspot_data['associated_companies'][0].get('properties', {})
    
    # Extract key data points
    business_desc = company_props.get('describe_the_business_product_in_one_sentence', '')
    usp = company_props.get('what_is_your_usp__what_makes_you_different_from_your_competitors_', '')
    sector = company_props.get('what_sector_is_your_business_product_', '')
    customer_base = company_props.get('what_best_describes_your_customer_base_', '')
    location = company_props.get('where_is_your_business_based_', '')
    company_name = company_props.get('name', '')
    
    # Combine all text for analysis
    combined_text = f"{business_desc} {usp} {customer_base}".lower()
    
    # Define specific market patterns and their corresponding targeted markets
    market_patterns = [
        # Food & Beverage Specific Markets
        {'keywords': ['kefir', 'probiotic drink', 'fermented beverage'], 'market': 'Kefir and Probiotic Beverages Market', 'terms': ['kefir market', 'probiotic beverages market', 'fermented drinks market']},
        {'keywords': ['plant-based', 'vegan', 'dairy-free', 'alternative protein'], 'market': 'Plant-Based Food and Alternative Protein Market', 'terms': ['plant-based food market', 'vegan food market', 'alternative protein market']},
        {'keywords': ['organic food', 'organic produce', 'organic farming'], 'market': 'Organic Food Market', 'terms': ['organic food market', 'organic agriculture market']},
        {'keywords': ['meal kit', 'food delivery', 'recipe box'], 'market': 'Meal Kit Delivery Market', 'terms': ['meal kit market', 'food delivery services market']},
        
        # FinTech Specific Markets
        {'keywords': ['neobank', 'digital bank', 'challenger bank'], 'market': 'Digital Banking and Neobanking Market', 'terms': ['neobank market', 'digital banking market', 'challenger bank market']},
        {'keywords': ['buy now pay later', 'bnpl', 'installment payment'], 'market': 'Buy Now Pay Later (BNPL) Market', 'terms': ['BNPL market', 'installment payments market', 'alternative credit market']},
        {'keywords': ['cryptocurrency', 'crypto', 'digital asset', 'blockchain wallet'], 'market': 'Cryptocurrency and Digital Assets Market', 'terms': ['cryptocurrency market', 'digital assets market', 'crypto trading market']},
        {'keywords': ['robo advisor', 'automated investing', 'algorithm trading'], 'market': 'Robo-Advisory and Automated Investment Market', 'terms': ['robo-advisor market', 'automated investment market']},
        {'keywords': ['insurtech', 'digital insurance', 'insurance technology'], 'market': 'InsurTech and Digital Insurance Market', 'terms': ['insurtech market', 'digital insurance market']},
        
        # Healthcare Specific Markets  
        {'keywords': ['telemedicine', 'telehealth', 'virtual consultation'], 'market': 'Telemedicine and Digital Health Market', 'terms': ['telemedicine market', 'telehealth market', 'digital health market']},
        {'keywords': ['mental health', 'therapy app', 'mindfulness', 'meditation'], 'market': 'Digital Mental Health and Wellness Market', 'terms': ['digital mental health market', 'therapy apps market', 'wellness technology market']},
        {'keywords': ['fitness tracker', 'wearable', 'health monitoring'], 'market': 'Wearable Health Technology Market', 'terms': ['wearable health market', 'fitness tracker market', 'health monitoring devices market']},
        
        # EdTech Specific Markets
        {'keywords': ['online learning', 'e-learning', 'educational platform'], 'market': 'Online Learning and EdTech Market', 'terms': ['online learning market', 'e-learning market', 'educational technology market']},
        {'keywords': ['coding bootcamp', 'programming education', 'software training'], 'market': 'Programming Education and Coding Bootcamp Market', 'terms': ['coding bootcamp market', 'programming education market']},
        {'keywords': ['language learning', 'language app', 'language course'], 'market': 'Digital Language Learning Market', 'terms': ['language learning apps market', 'digital language education market']},
        
        # Transportation Specific Markets
        {'keywords': ['public transit', 'transit app', 'bus tracking', 'train schedule'], 'market': 'Public Transit Technology Market', 'terms': ['transit technology market', 'public transportation apps market', 'mobility-as-a-service market']},
        {'keywords': ['ride sharing', 'rideshare', 'car sharing'], 'market': 'Ride-Sharing and Car-Sharing Market', 'terms': ['ride-sharing market', 'car-sharing market', 'shared mobility market']},
        {'keywords': ['electric vehicle', 'ev charging', 'electric car'], 'market': 'Electric Vehicle and EV Charging Market', 'terms': ['electric vehicle market', 'EV charging market', 'electric mobility market']},
        
        # E-commerce Specific Markets
        {'keywords': ['social commerce', 'social selling', 'influencer marketplace'], 'market': 'Social Commerce and Influencer Marketing Market', 'terms': ['social commerce market', 'influencer marketing market', 'creator economy market']},
        {'keywords': ['subscription box', 'subscription service'], 'market': 'Subscription Box and Recurring Commerce Market', 'terms': ['subscription box market', 'subscription commerce market']},
        {'keywords': ['marketplace', 'peer-to-peer', 'p2p marketplace'], 'market': 'Peer-to-Peer Marketplace Market', 'terms': ['P2P marketplace market', 'peer-to-peer commerce market']},
        
        # PropTech Specific Markets
        {'keywords': ['property management', 'rental platform', 'real estate tech'], 'market': 'PropTech and Real Estate Technology Market', 'terms': ['proptech market', 'real estate technology market', 'property management software market']},
        {'keywords': ['short-term rental', 'vacation rental', 'airbnb'], 'market': 'Short-Term Rental and Vacation Rental Market', 'terms': ['short-term rental market', 'vacation rental market', 'alternative accommodation market']},
        
        # SaaS Specific Markets
        {'keywords': ['crm', 'customer relationship management'], 'market': 'CRM Software Market', 'terms': ['CRM software market', 'customer relationship management market']},
        {'keywords': ['project management', 'task management', 'workflow'], 'market': 'Project Management Software Market', 'terms': ['project management software market', 'task management tools market']},
        {'keywords': ['hr software', 'human resources', 'payroll'], 'market': 'HR Technology and Workforce Management Market', 'terms': ['HR software market', 'workforce management market', 'payroll software market']},
        
        # Add more patterns as needed...
    ]
    
    # Find the most specific market match
    best_match = None
    max_matches = 0
    
    for pattern in market_patterns:
        matches = sum(1 for keyword in pattern['keywords'] if keyword in combined_text)
        if matches > max_matches:
            max_matches = matches
            best_match = pattern
    
    # Determine geographic scope
    geographic_scope = 'Global'  # Default
    if location:
        location_lower = location.lower()
        if any(country in location_lower for country in ['uk', 'united kingdom', 'britain', 'england', 'scotland', 'wales']):
            geographic_scope = 'UK'
        elif any(country in location_lower for country in ['usa', 'united states', 'america', 'us']):
            geographic_scope = 'US'
        elif any(region in location_lower for region in ['europe', 'eu', 'european']):
            geographic_scope = 'Europe'
        elif any(region in location_lower for region in ['asia', 'singapore', 'hong kong', 'japan', 'korea']):
            geographic_scope = 'Asia'
        elif any(region in location_lower for region in ['africa', 'south africa', 'nigeria', 'kenya']):
            geographic_scope = 'Africa'
        elif any(region in location_lower for region in ['caribbean', 'barbados', 'jamaica', 'trinidad']):
            geographic_scope = 'Caribbean'
        elif location:
            geographic_scope = location  # Use specific location if not in predefined regions
    
    if best_match and max_matches > 0:
        market_focus = best_match['market']
        search_terms = best_match['terms']
        reasoning = f"Identified specific market based on {max_matches} keyword matches in company description and USP"
    else:
        # Fallback to analyzing business description for specific terms
        if business_desc:
            # Extract potential market indicators from business description
            desc_words = business_desc.lower().split()
            specific_terms = []
            
            # Look for specific product/service mentions
            for word in desc_words:
                if len(word) > 4 and word not in ['platform', 'service', 'company', 'business', 'solution', 'technology', 'system']:
                    specific_terms.append(word)
            
            if specific_terms:
                market_focus = f"{' '.join(specific_terms[:3]).title()} Market"
                search_terms = [f"{term} market" for term in specific_terms[:2]]
                reasoning = f"Derived market focus from business description: '{business_desc[:100]}...'"
            else:
                # Last resort - use sector with business model qualifier
                if sector and customer_base:
                    market_focus = f"{sector} for {customer_base}"
                    search_terms = [f"{sector} market", f"{customer_base} {sector}"]
                    reasoning = f"Combined sector '{sector}' with customer base '{customer_base}'"
                elif sector:
                    market_focus = sector
                    search_terms = [f"{sector} market"]
                    reasoning = f"Used generic sector: {sector}"
                else:
                    market_focus = "Technology Market"
                    search_terms = ["technology market"]
                    reasoning = "No specific market indicators found, defaulting to technology"
        else:
            market_focus = sector if sector else "Technology Market"
            search_terms = [f"{sector} market"] if sector else ["technology market"]
            reasoning = f"Limited data available, using sector: {sector or 'Technology'}"
    
    return {
        'market_focus': market_focus,
        'search_terms': search_terms,
        'geographic_scope': geographic_scope,
        'reasoning': reasoning
    }

# --- Output Parser for MarketAnalysis ---
market_analysis_parser = PydanticOutputParser(pydantic_object=MarketAnalysis)

# --- Prompt Template for Market Intelligence ---
MARKET_INTELLIGENCE_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a specialized market intelligence AI analyst. Your goal is to gather comprehensive information about a SPECIFIC MARKET using targeted research that has been pre-defined based on company analysis.\n"
            "IMPORTANT: You have been provided with a SPECIFIC MARKET FOCUS and TARGETED SEARCH TERMS based on the company's actual business model, products, and positioning. Use these specific terms rather than generic industry categories.\n\n"
            "MARKET RESEARCH CONTEXT:\n"
            "Target Market: {market_focus}\n"
            "Geographic Scope: {geographic_scope}\n"
            "Suggested Search Terms: {search_terms}\n"
            "Market Definition Reasoning: {market_reasoning}\n\n"
            "🚨 CRITICAL: You MUST provide actual data values for ALL fields - do NOT leave fields as null unless absolutely no information can be found. Use your research results to synthesize meaningful values.\n\n"
            "You MUST use the 'search_tool' and 'wiki_tool' to find the required information. Make multiple targeted search queries using the SPECIFIC SEARCH TERMS provided above:\n\n"
            "TARGETED RESEARCH STRATEGY:\n"
            "1. Market Size and Growth (use SPECIFIC market terms):\n"
            "   - Search for '{market_focus} market size {geographic_scope}'\n"
            "   - Search each specific term: {search_terms} with 'market size', 'TAM', 'revenue'\n"
            "   - Look for market research reports specific to this market (not generic industry)\n"
            "   - Find CAGR and growth projections for this SPECIFIC market\n"
            "   - Example: Instead of 'health market size', search 'kefir market size' or 'probiotic beverages market size'\n"
            "2. Market Trends and Dynamics:\n"
            "   - Search for '{market_focus} trends 2024-2025'\n"
            "   - Search each term: {search_terms} combined with 'trends', 'growth drivers', 'consumer behavior'\n"
            "   - Look for regulatory changes, technological advances, consumer shifts specific to this market\n"
            "   - MUST provide at least 3-5 specific trends from your targeted research\n"
            "3. Competitive Landscape:\n"
            "   - Search for 'top companies {market_focus}'\n"
            "   - Search 'competitors in {market_focus}' and each specific search term\n"
            "   - Find 3-5 key competitors specifically in this market (not broader industry)\n"
            "   - Look for market share, funding, and positioning within this specific market\n"
            "4. Market Maturity and Timing:\n"
            "   - Assess market stage: emerging/growth/mature for this SPECIFIC market\n"
            "   - Look for venture capital activity, new entrants, M&A specifically in this market\n"
            "   - Search for 'investment in {market_focus}' and funding trends\n"
            "5. Market Entry Barriers:\n"
            "   - Search for 'challenges starting in {market_focus}'\n"
            "   - Look for regulatory, technical, and competitive barriers specific to this market\n"
            "   - Consider capital requirements, expertise needed, market access challenges\n"
            "6. Regulatory Environment:\n"
            "   - Search for 'regulations {market_focus} {geographic_scope}'\n"
            "   - Look for compliance requirements, licensing, safety standards specific to this market\n\n"
            "ENHANCED SEARCH STRATEGY:\n"
            "- Use the provided specific search terms rather than generic industry terms\n"
            "- Start with Wikipedia for market overview using the specific market focus\n"
            "- Cross-reference multiple sources for market data specific to this niche\n"
            "- Look for specialized market research reports (Grand View Research, IBISWorld, Technavio) for this specific market\n"
            "- Search for recent news and press releases using the specific market terms\n\n"
            "CRITICAL REQUIREMENTS:\n"
            "🎯 For each field in the JSON schema:\n"
            "- target_market_segment: Describe the specific customer segments for this targeted market\n"
            "- market_size_tam: Find TAM specifically for this market (e.g., 'Kefir market: $2.1B globally')\n"
            "- market_size_sam: Estimate SAM within the geographic scope\n"
            "- market_size_som: Estimate realistic SOM for a new entrant in this market\n"
            "- market_growth_rate_cagr: Find CAGR specific to this market, not broader industry\n"
            "- key_market_trends: List trends specific to this market\n"
            "- competitors: Identify competitors specifically in this market niche\n"
            "- barriers_to_entry: List barriers specific to entering this market\n"
            "- regulatory_environment: Describe regulations specific to this market and geography\n\n"
            "Your FINAL output MUST be a single, valid JSON object that strictly conforms to the MarketAnalysis schema below.\n"
            "🚨 CRITICAL: Every field must contain actual data synthesized from your TARGETED research - no null values unless absolutely no information exists.\n"
            "Use the jurisdiction field to specify the geographic scope: {geographic_scope}\n"
            "Schema:\n{format_instructions}"
        ),
        ("human", "Please research the specific market: {market_focus}\n\nFocus on these specific search terms: {search_terms}\nGeographic scope: {geographic_scope}\nOriginal input context: {input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
).partial(format_instructions=market_analysis_parser.get_format_instructions())


def _get_fallback_llm():
    """Fallback LLM selection logic (simplified)."""
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        logger.info("MarketIntelligenceAgent: Using OpenAI GPT model (fallback)")
        return ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.2, max_tokens=4000)
    elif os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        logger.info("MarketIntelligenceAgent: Using Anthropic Claude model (fallback)")
        return ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.3, max_tokens=4000)
    else:
        logger.error("MarketIntelligenceAgent: FATAL: No LLM API keys available for fallback.")
        raise ValueError("No LLM API keys for Market Intelligence Agent fallback.")

def _core_market_intelligence_logic(market_or_industry: str, llm: Any, model_config: Optional[ModelConfig], hubspot_data: Optional[Dict[str, Any]] = None) -> MarketAnalysis:
    """
    The core logic for market intelligence research, designed to be wrapped by the retry decorator.
    It expects an LLM instance and its config to be passed in, along with optional HubSpot data for targeted market definition.
    """
    logger.info(f"MarketIntelligenceAgent: Executing core research for \"{market_or_industry}\" with model {model_config.model_name if model_config else 'Unknown'}.")

    if not tavily_search_tool_instance:
        logger.error("MarketIntelligenceAgent: FATAL: Tavily search_tool is not available. Cannot proceed.")
        raise ValueError("Tavily search_tool not available for Market Intelligence.")

    if not wiki_tool_instance:
        logger.warning("MarketIntelligenceAgent: Wikipedia tool is not available. Research will rely solely on web search.")

    available_tools = [search_tool]
    if wiki_tool_instance:
        available_tools.append(wiki_tool)
    
    # Define specific market using HubSpot data if available
    if hubspot_data:
        market_definition = _define_specific_market_from_hubspot_data(hubspot_data)
        logger.info(f"MarketIntelligenceAgent: Using specific market definition - {market_definition['market_focus']} ({market_definition['reasoning']})")
        
        # Use enhanced prompt with market-specific context
        agent_prompt = MARKET_INTELLIGENCE_PROMPT_TEMPLATE.partial(
            market_focus=market_definition['market_focus'],
            geographic_scope=market_definition['geographic_scope'],
            search_terms=', '.join(market_definition['search_terms']),
            market_reasoning=market_definition['reasoning']
        )
        
        # Create enhanced input that includes the specific market context
        enhanced_input = f"Specific Market: {market_definition['market_focus']}\nSearch Terms: {', '.join(market_definition['search_terms'])}\nGeographic Scope: {market_definition['geographic_scope']}\nOriginal Context: {market_or_industry}"
        
    else:
        # Fallback to original behavior for backward compatibility
        logger.info(f"MarketIntelligenceAgent: No HubSpot data available, using generic market research for: {market_or_industry}")
        market_definition = {
            'market_focus': market_or_industry,
            'geographic_scope': 'Global',
            'search_terms': [f"{market_or_industry} market"],
            'reasoning': 'No HubSpot data available, using generic approach'
        }
        
        agent_prompt = MARKET_INTELLIGENCE_PROMPT_TEMPLATE.partial(
            market_focus=market_or_industry,
            geographic_scope='Global',
            search_terms=f"{market_or_industry} market",
            market_reasoning='Generic market research without company-specific context'
        )
        
        enhanced_input = market_or_industry

    agent = create_tool_calling_agent(
        llm=llm,
        prompt=agent_prompt,
        tools=available_tools
    )
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=available_tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=15
    )

    raw_agent_output_container = None
    json_to_parse = ""
    try:
        # Use the enhanced input with specific market context
        raw_agent_output_container = agent_executor.invoke({
            "input": enhanced_input,
            "market_focus": market_definition['market_focus'],
            "search_terms": ', '.join(market_definition['search_terms']),
            "geographic_scope": market_definition['geographic_scope']
        })
        
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
            logger.error(f"MarketIntelligenceAgent: Agent for \"{market_or_industry}\" produced no parsable output.")
            raise ValueError(f"Agent for {market_or_industry} produced no parsable output for MarketAnalysis.")

        logger.debug(f"MarketIntelligenceAgent: Raw LLM content for \"{market_or_industry}\": {llm_response_content_str[:500]}...")

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
                logger.warning(f"MarketIntelligenceAgent: Used fallback JSON extraction for \"{market_or_industry}\".")
            else:
                logger.error(f"MarketIntelligenceAgent: Could not extract JSON from LLM response for \"{market_or_industry}\": {llm_response_content_str[:500]}...")
                raise ValueError(f"Could not extract valid JSON for {market_or_industry} from LLM response.")
        
        logger.debug(f"MarketIntelligenceAgent: Extracted JSON for \"{market_or_industry}\": {json_to_parse[:300]}...")
        parsed_dict = json.loads(json_to_parse, strict=False)
        market_analysis_obj = MarketAnalysis.model_validate(parsed_dict)

        logger.info(f"MarketIntelligenceAgent: Successfully researched and validated market analysis for: {market_definition['market_focus']} (from input: {market_or_industry})")
        return market_analysis_obj

    except json.JSONDecodeError as json_err:
        logger.error(f"MarketIntelligenceAgent: JSONDecodeError for \"{market_or_industry}\". Error: {json_err}. String: {json_to_parse[:1000]}", exc_info=True)
        if raw_agent_output_container: logger.debug(f"Raw agent output at time of JSON error: {raw_agent_output_container}")
        raise # Re-raise for retry handler
    except Exception as e:
        logger.error(f"MarketIntelligenceAgent: Error during core research for \"{market_or_industry}\": {e}", exc_info=True)
        if raw_agent_output_container: logger.debug(f"Raw agent output at time of error: {raw_agent_output_container}")
        raise # Re-raise for retry handler

# --- Apply Decorator ---
_decorated_core_market_logic = _core_market_intelligence_logic

if MODEL_SYSTEM_AVAILABLE:
    _model_selector_for_retry = None
    try:
        from ai_agents.config.model_config import get_llm_for_agent as get_llm_func_for_decorator
        _model_selector_for_retry = get_llm_func_for_decorator
    except ImportError:
        logger.warning("MarketIntelligenceAgent: Could not import get_llm_for_agent for retry decorator. Model switching disabled.")

    _retry_config = RetryConfig(
        max_retries=3, 
        base_delay=2.5, # Slightly longer base for potentially more complex market queries
        switch_model_on_rate_limit=bool(_model_selector_for_retry)
    )
    
    _decorated_core_market_logic = with_smart_retry(
        agent_name="market_intelligence",
        retry_config_override=_retry_config,
        model_selector_func=_model_selector_for_retry
    )(_core_market_intelligence_logic)
else:
    logger.warning("MarketIntelligenceAgent: Retry system not fully available. Research will not have advanced retry/model switching.")


def run_market_intelligence_cli(market_or_industry: str, hubspot_data: Optional[Dict[str, Any]] = None) -> Optional[MarketAnalysis]:
    """
    Sets up and runs the market intelligence research with optional HubSpot data for enhanced targeting.
    
    Args:
        market_or_industry: The market or industry to research (can be generic if HubSpot data provides specificity)
        hubspot_data: Optional HubSpot deal data containing company information for targeted market definition
    
    Returns:
        MarketAnalysis object or None if research fails
    """
    if hubspot_data:
        logger.info(f"MarketIntelligenceAgent: Starting ENHANCED research for \"{market_or_industry}\" with HubSpot data via CLI.")
    else:
        logger.info(f"MarketIntelligenceAgent: Starting standard research for \"{market_or_industry}\" via CLI.")

    if not tavily_search_tool_instance: # Initial check
        logger.error("MarketIntelligenceAgent: FATAL: Tavily search_tool is not available. Cannot proceed.")
        return None

    llm_instance = None
    model_config_instance = None

    if MODEL_SYSTEM_AVAILABLE:
        try:
            llm_instance, model_config_instance = get_llm_for_agent("market_intelligence")
        except ModelSelectionError as e_mse:
            logger.error(f"MarketIntelligenceAgent: Initial model selection failed for '{market_or_industry}': {e_mse}. Attempting fallback.")
            try:
                llm_instance = _get_fallback_llm()
                model_config_instance = ModelConfig(provider="Unknown", model_name="fallback", temperature=0.1, max_tokens=4000, cost_per_1k_tokens=0, rate_limit_rpm=0, best_for=[]) 
            except ValueError as e_fb:
                 logger.error(f"MarketIntelligenceAgent: Fallback LLM also failed for '{market_or_industry}': {e_fb}")
                 return None
        except Exception as e_llm_init:
            logger.error(f"MarketIntelligenceAgent: Unexpected error during initial LLM init for '{market_or_industry}': {e_llm_init}")
            return None
    else: # No model system, try direct fallback
        try:
            llm_instance = _get_fallback_llm()
            model_config_instance = ModelConfig(provider="Unknown", model_name="fallback_direct", temperature=0.1, max_tokens=4000, cost_per_1k_tokens=0, rate_limit_rpm=0, best_for=[])
        except ValueError as e_fb_direct:
            logger.error(f"MarketIntelligenceAgent: Fallback LLM failed (no model system): {e_fb_direct}")
            return None

    if not llm_instance:
        logger.error(f"MarketIntelligenceAgent: LLM instance is None for '{market_or_industry}'. Cannot proceed.")
        return None

    try:
        analysis_obj = _decorated_core_market_logic(
            market_or_industry=market_or_industry,
            llm=llm_instance,
            model_config=model_config_instance,
            hubspot_data=hubspot_data
        )

        if analysis_obj:
            # Generate filename based on specific market focus if available
            if hubspot_data:
                market_def = _define_specific_market_from_hubspot_data(hubspot_data)
                safe_market_name = "".join(c if c.isalnum() else "_" for c in market_def['market_focus'])[:50].rstrip("_")
                output_filename = f"market_analysis_enhanced_{safe_market_name}.json"
                logger.info(f"MarketIntelligenceAgent: Enhanced analysis completed for specific market: {market_def['market_focus']}")
            else:
                safe_market_name = "".join(c if c.isalnum() else "_" for c in market_or_industry)[:50].rstrip("_")
                output_filename = f"market_analysis_{safe_market_name}.json"
                
            try:
                save_tool.invoke({
                    "filename": output_filename,
                    "text": analysis_obj.model_dump_json(indent=2)
                })
                logger.info(f"MarketIntelligenceAgent: Analysis saved to {output_filename}")
            except Exception as e_save:
                logger.error(f"MarketIntelligenceAgent: Error saving analysis to file: {e_save}")
            
            # Print summary of key findings to stderr for orchestrator logs
            market_name = market_or_industry
            if hubspot_data:
                market_def = _define_specific_market_from_hubspot_data(hubspot_data)
                market_name = market_def['market_focus']
                
            logger.info(f"--- Market Intelligence Summary for {market_name} ---")
            if analysis_obj.industry_overview: logger.info(f"Overview: {analysis_obj.industry_overview[:200]}...")
            if analysis_obj.market_size_tam: logger.info(f"TAM: {analysis_obj.market_size_tam}")
            if analysis_obj.market_growth_rate_cagr: logger.info(f"CAGR: {analysis_obj.market_growth_rate_cagr}")
            if analysis_obj.market_timing_assessment: logger.info(f"Timing: {analysis_obj.market_timing_assessment}")
            if analysis_obj.competitors: logger.info(f"Competitors: {len(analysis_obj.competitors)}")

        return analysis_obj

    except Exception as e:
        logger.error(f"MarketIntelligenceAgent: Research for \"{market_or_industry}\" ultimately failed: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    logger.info("Market intelligence agent CLI starting...")
    
    parser = argparse.ArgumentParser(description="Market Intelligence Agent CLI - Researches a market/industry sector with optional HubSpot data for enhanced targeting.")
    parser.add_argument("--sector", type=str, required=True, help="The market or industry sector to research.")
    parser.add_argument("--hubspot-data", type=str, help="Optional path to JSON file containing HubSpot deal data for enhanced market targeting.")
    
    args = parser.parse_args()

    # Load HubSpot data if provided
    hubspot_data = None
    if args.hubspot_data:
        try:
            with open(args.hubspot_data, 'r') as f:
                hubspot_data = json.load(f)
            logger.info(f"Loaded HubSpot data from {args.hubspot_data} for enhanced market research")
        except Exception as e:
            logger.error(f"Failed to load HubSpot data from {args.hubspot_data}: {e}")
            logger.info("Proceeding with standard market research")

    try:
        analysis = run_market_intelligence_cli(args.sector, hubspot_data=hubspot_data)
        
        if analysis:
            print(analysis.model_dump_json(indent=None))
            if hubspot_data:
                market_def = _define_specific_market_from_hubspot_data(hubspot_data)
                logger.info(f"Market intelligence agent: Successfully generated ENHANCED JSON output for specific market: {market_def['market_focus']} (from sector: {args.sector})")
            else:
                logger.info(f"Market intelligence agent: Successfully generated JSON output for sector: {args.sector}")
        else:
            error_output = {
                "status": "error",
                "sector": args.sector,
                "error_message": f"Failed to retrieve or generate market analysis for sector: {args.sector}."
            }
            print(json.dumps(error_output, indent=None))
            logger.error(f"Market intelligence agent: Failed to generate analysis for sector: {args.sector}.")
            sys.exit(1)
            
    except Exception as e_main:
        error_output = {
            "status": "error",
            "sector": args.sector,
            "error_message": f"Critical error in market_intelligence_agent CLI for sector {args.sector}: {str(e_main)}"
        }
        print(json.dumps(error_output, indent=None))
        logger.critical(f"Critical error in market_intelligence_agent CLI for sector {args.sector}: {e_main}", exc_info=True)
        sys.exit(1)