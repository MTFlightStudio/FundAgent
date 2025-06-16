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
def _core_company_research_logic(company_name: str, llm: Any, model_config: Optional[ModelConfig]) -> CompanyProfile:
    """
    The core logic for company research, designed to be wrapped by the retry decorator.
    It expects an LLM instance to be passed in.
    """
    logger.info(f"CompanyResearchAgent: Executing core research logic for \"{company_name}\" with model {model_config.model_name if model_config else 'Unknown'}.")

    if not tavily_search_tool_instance:
        logger.error("CompanyResearchAgent: FATAL: Tavily search_tool is not available. Cannot proceed.")
        # This should ideally be caught before even calling this core logic,
        # but as a safeguard within the core execution path.
        raise ValueError("Tavily search_tool not available for Company Research.")

    available_tools = [search_tool]

    company_research_agent_prompt = COMPANY_RESEARCH_PROMPT_TEMPLATE 
    # company_profile_parser is globally defined

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
        raw_agent_output_container = agent_executor.invoke({"input": company_name})

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
        
        logger.debug(f"CompanyResearchAgent: Extracted JSON for \"{company_name}\": {json_to_parse[:300]}...")
        parsed_dict = json.loads(json_to_parse, strict=False)

        for url_field in ["website", "linkedin_url"]:
            if url_field in parsed_dict and isinstance(parsed_dict[url_field], str):
                parsed_dict[url_field] = _ensure_url_scheme(parsed_dict[url_field])
        
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

def run_company_research_cli(company_name: str) -> Optional[CompanyProfile]:
    """
    Sets up and runs the company research, utilizing the (potentially) decorated core logic.
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
        # Call the (potentially) decorated core research logic
        # Pass the initially selected LLM and its config. The decorator will use these for the first attempt
        # and can switch them for subsequent attempts if `model_selector_func` is provided to it.
        company_profile_obj = _decorated_core_research_logic(
            company_name=company_name, 
            llm=llm_instance, # Pass the LLM instance
            model_config=model_config_instance # Pass its config
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