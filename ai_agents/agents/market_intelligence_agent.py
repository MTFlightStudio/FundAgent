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

# --- Output Parser for MarketAnalysis ---
market_analysis_parser = PydanticOutputParser(pydantic_object=MarketAnalysis)

# --- Prompt Template for Market Intelligence ---
MARKET_INTELLIGENCE_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a specialized market intelligence AI analyst. Your goal is to gather comprehensive information about a given market or industry, focusing on a specific GEOGRAPHIC JURISDICTION if provided or implied in the user's query. Structure the output into a JSON object adhering to the MarketAnalysis schema provided below.\n"
            "If the user's query specifies a region (e.g., 'US market for X', 'European Y industry'), focus your research and analysis on that region. Populate the 'jurisdiction' field in your JSON output accordingly (e.g., 'USA', 'Europe', 'UK'). If no specific jurisdiction is given, attempt to determine the most relevant one from your search results or default to 'Global' and note this in the 'jurisdiction' field.\n\n"
            "You MUST use the 'search_tool' and 'wiki_tool' to find the required information. Make multiple targeted search queries to cover all aspects for the SPECIFIED JURISDICTION:\n\n"
            "Required Research Areas (for the specified jurisdiction):\n"
            "1. Market Size and Growth:\n"
            "   - Search for '[jurisdiction] [market name] market size', '[market name] market size [jurisdiction] billion'\n"
            "   - Find Total Addressable Market (TAM), Serviceable Addressable Market (SAM), and Serviceable Obtainable Market (SOM) for the jurisdiction.\n"
            "   - Look for CAGR (Compound Annual Growth Rate) and growth projections within the jurisdiction.\n"
            "2. Key Market Trends (within the jurisdiction):\n"
            "   - Search for '[jurisdiction] [market name] trends 2024-2025', 'future of [market] in [jurisdiction]'\n"
            "   - Look for technological shifts, consumer behavior changes, emerging segments\n"
            "3. Competitive Landscape (within the jurisdiction):\n"
            "   - Search for 'top companies [market name] [jurisdiction]', '[market name] competitive landscape [jurisdiction]'\n"
            "   - Find 3-5 key competitors active in the jurisdiction with their funding, strengths, and positioning\n"
            "   - Look for market share data, competitive advantages, and differentiation\n"
            "4. Market Timing and Maturity:\n"
            "   - Assess if the market is in early/growth/mature/declining phase\n"
            "   - Look for indicators like number of new entrants, M&A activity, funding trends\n"
            "5. Barriers to Entry:\n"
            "   - Search for 'barriers to entry [market name]', 'challenges [market name] startups'\n"
            "   - Consider regulatory, capital, technical, and network effect barriers\n"
            "6. Regulatory Environment (specific to the jurisdiction):\n"
            "   - Search for '[jurisdiction] [market name] regulations', 'compliance requirements [market] [jurisdiction]'\n"
            "   - Look for government policies, licensing requirements, data protection laws\n\n"
            "Search Strategy Tips:\n"
            "- Start with Wikipedia for industry overview and established facts\n"
            "- Use specific search queries for recent market data (include years like 2024, 2025)\n"
            "- Look for credible sources: research firms (Gartner, McKinsey, CB Insights), industry reports, news outlets\n"
            "- Cross-reference multiple sources for market size estimates\n"
            "- Be specific with competitor searches to get funding and positioning data\n\n"
            "After gathering comprehensive information, your FINAL output MUST be a single, valid JSON object that strictly conforms to the MarketAnalysis schema below, including the 'jurisdiction' field. Do NOT include any other text, explanations, or markdown outside of this JSON object.\n"
            "If specific information cannot be found, set its value to null or provide your best estimate with a note about uncertainty.\n"
            "For market size figures, always include the currency (e.g., '$10B', '€5M').\n"
            "For growth rates, use percentages (e.g., '25%').\n"
            "Schema:\n{format_instructions}"
        ),
        ("human", "Please research the market/industry: {input}"),
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

def _core_market_intelligence_logic(market_or_industry: str, llm: Any, model_config: Optional[ModelConfig]) -> MarketAnalysis:
    """
    The core logic for market intelligence research, designed to be wrapped by the retry decorator.
    It expects an LLM instance and its config to be passed in.
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
    
    agent_prompt = MARKET_INTELLIGENCE_PROMPT_TEMPLATE 
    # market_analysis_parser is globally defined

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
        raw_agent_output_container = agent_executor.invoke({"input": market_or_industry})
        
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

        logger.info(f"MarketIntelligenceAgent: Successfully researched and validated market analysis for: {market_or_industry}")
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


def run_market_intelligence_cli(market_or_industry: str) -> Optional[MarketAnalysis]:
    """
    Sets up and runs the market intelligence research.
    """
    logger.info(f"MarketIntelligenceAgent: Starting research for sector: \"{market_or_industry}\" via CLI.")

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
            model_config=model_config_instance
        )

        if analysis_obj:
            safe_market_name = "".join(c if c.isalnum() else "_" for c in market_or_industry)[:50].rstrip("_")
            output_filename = f"market_analysis_{safe_market_name}.json"
            try:
                save_tool.invoke({
                    "filename": output_filename,
                    "text": analysis_obj.model_dump_json(indent=2)
                })
                logger.info(f"MarketIntelligenceAgent: Analysis for \"{market_or_industry}\" saved to {output_filename}")
            except Exception as e_save:
                logger.error(f"MarketIntelligenceAgent: Error saving analysis for \"{market_or_industry}\" to file: {e_save}")
            
            # Print summary of key findings to stderr for orchestrator logs
            logger.info(f"--- Market Intelligence Summary for {market_or_industry} ---")
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
    
    parser = argparse.ArgumentParser(description="Market Intelligence Agent CLI - Researches a market/industry sector.")
    parser.add_argument("--sector", type=str, required=True, help="The market or industry sector to research.")
    
    args = parser.parse_args()

    try:
        analysis = run_market_intelligence_cli(args.sector)
        
        if analysis:
            print(analysis.model_dump_json(indent=None))
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