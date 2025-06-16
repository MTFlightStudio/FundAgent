import os
import json
import re
import sys
import argparse
import logging
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor

# Import models and tools from your project structure
from ai_agents.models.investment_research import (
    FounderProfile,
    FounderCriteriaAssessment,
    PreviousCompany,
    EducationDetail
)
from ai_agents.tools import (
    search_tool,
    research_prospect_tool,
    save_tool,
    tavily_search_tool_instance,
    relevance_ai_tool_configured
)

# Import the new model selection and retry systems
try:
    from ai_agents.config.model_config import get_llm_for_agent, SmartModelSelector # SmartModelSelector might not be directly used if get_llm_for_agent is primary
    from ai_agents.utils.retry_handler import with_smart_retry, RetryConfig
    MODEL_SYSTEM_AVAILABLE = True # Combined flag
except ImportError:
    MODEL_SYSTEM_AVAILABLE = False
    logging.warning("Model selection or retry system not available, functionality will be limited or use fallbacks.")

load_dotenv()

# Setup logging
# logging.basicConfig(level=logging.INFO) # Avoid reconfiguring if already set by orchestrator
logger = logging.getLogger(__name__) # Use __name__ for module-specific logger

class EnhancedFounderResearchAgent:
    def __init__(self):
        self.agent_name = "founder_research"
        # self.model_selector = SmartModelSelector() if MODEL_SYSTEM_AVAILABLE else None # Not directly used if get_llm_for_agent is the entry point
        self.current_llm = None # Stores the LLM for the current research_founder call, potentially updated by _get_llm
        self.current_model_config = None # Stores its config
        
    def _get_llm(self, prefer_fast: bool = False):
        """Get the best available LLM for this agent for the current attempt."""
        if MODEL_SYSTEM_AVAILABLE:
            try:
                # get_llm_for_agent is the main interface now
                llm, config = get_llm_for_agent(self.agent_name, prefer_fast=prefer_fast)
                self.current_llm = llm # Update class state if needed elsewhere, though primarily used in current call
                self.current_model_config = config
                # The get_llm_for_agent function already logs the selected model
                # logger.info(f"Using model: {config.model_name} for founder research (via _get_llm)")
                return llm
            except Exception as e:
                logger.warning(f"Model selection via get_llm_for_agent failed for '{self.agent_name}', attempting fallback: {e}")
        
        # Fallback to original simple logic if advanced system fails or not available
        return self._get_fallback_llm()
    
    def _get_fallback_llm(self):
        """Fallback LLM selection logic (simplified)."""
        # This method is only called if MODEL_SYSTEM_AVAILABLE is False or get_llm_for_agent fails.
        if os.getenv("OPENAI_API_KEY"):
            from langchain_openai import ChatOpenAI
            logger.info("Using OpenAI GPT model for founder research (fallback)")
            # Ensure max_tokens is set for fallback models too
            return ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.1, max_tokens=4000)
        elif os.getenv("ANTHROPIC_API_KEY"):
            from langchain_anthropic import ChatAnthropic
            logger.info("Using Anthropic Claude model for founder research (fallback)")
            return ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.2, max_tokens=4000)
        else:
            # If no keys, this will prevent agent execution.
            logger.error("FATAL: No LLM API keys available for fallback.")
            # Raising an error here is better than returning None and failing later.
            raise ValueError("No LLM API keys available for fallback LLM initialization.")

    def _ensure_url_scheme(self, url: Optional[str]) -> Optional[str]:
        """Prepends https:// to a URL if it's missing a scheme."""
        if url and not urlparse(url).scheme and (url.startswith("www.") or "." in url.split("/")[0]):
            return f"https://{url}"
        return url

    # Note: The @with_smart_retry decorator will manage retries.
    # If MODEL_SYSTEM_AVAILABLE is False, the retry_handler itself might not be available.
    # The decorator is applied below the class definition to correctly pass the model_selector_func.
    def research_founder(self, founder_name: str, linkedin_url: Optional[str] = None, llm_override: Optional[Any] = None, model_config_override: Optional[Any] = None, **kwargs) -> Optional[FounderProfile]:
        """
        Enhanced founder research with retry logic and model switching.
        `llm_override` and `model_config_override` can be passed by the retry decorator.
        The decorator will also pass 'llm' and 'model_config' in kwargs.
        """
        logger.info(f"Starting research for founder: \"{founder_name}\" using EnhancedFounderResearchAgent.")
        
        processed_linkedin_url = self._ensure_url_scheme(linkedin_url)
        if processed_linkedin_url:
            logger.info(f"Processed LinkedIn URL for research: {processed_linkedin_url}")

        # Tool availability checks
        if not tavily_search_tool_instance:
            logger.warning("Tavily search_tool is not available (check TAVILY_API_KEY). Web search capabilities will be limited.")
        
        if processed_linkedin_url and not relevance_ai_tool_configured:
            logger.warning("LinkedIn URL provided, but Relevance AI 'Research Prospect' tool is not configured. Will rely on general web search for LinkedIn details.")

        # Determine LLM for this attempt:
        # 1. Use llm_override if provided by retry decorator (means a model switch happened)
        # 2. Else, check kwargs for 'llm' also injected by the decorator
        # 3. Else, call _get_llm() to select/fallback.
        active_llm = llm_override if llm_override else kwargs.get("llm")
        # active_model_config = model_config_override if model_config_override else kwargs.get("model_config")

        if not active_llm:
            active_llm = self._get_llm() # This will use get_llm_for_agent or fallback

        if not active_llm:
            # _get_llm or _get_fallback_llm should raise if they fail, but as a safeguard:
            logger.error(f"FATAL: Could not obtain an LLM for founder research agent '{self.agent_name}'.")
            raise ValueError(f"No LLM available for founder research agent '{self.agent_name}'.")

        # Setup tools
        available_tools = [search_tool] # Tavily is primary general search
        if relevance_ai_tool_configured: # Relevance AI for LinkedIn specific prospect research
            available_tools.append(research_prospect_tool)

        if not available_tools: # Should not happen if search_tool is always at least attempted
            logger.error("FATAL: No research tools are available. Cannot proceed.")
            raise ValueError("No research tools available for founder research.")

        # Create prompt template (simplified as per user's latest version)
        founder_profile_parser = PydanticOutputParser(pydantic_object=FounderProfile)
        
        prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a specialized founder research AI. Your goal is to take raw data from a LinkedIn scrape, enrich it with supplemental information, and structure it into a final, detailed JSON object adhering to the FounderProfile schema.\n\n"
                "**Research Strategy:**\n\n"
                "1.  **Primary Tool (`research_prospect_tool`)**: Your first and most important action is to call the `research_prospect_tool`. This tool returns a **JSON object** which is your primary source of truth for the founder's history.\n\n"
                "2.  **Map Directly from Tool Output**: The JSON from the tool contains `linkedin_profile_data`. This data is rich and should be mapped directly to your final `FounderProfile` schema.\n"
                "    -   The `about` text should be used for the `background_summary`.\n"
                "    -   The `experiences` list is the primary source for the `previous_companies` field. Iterate through it and map the fields (e.g., map `title` to `role`, `company` to `company_name`, etc.). You must also determine the `was_founder` boolean for each role.\n"
                "    -   The `education` list is the primary source for the `education` field in your final output. Map the fields directly.\n\n"
                "3.  **Enrich for Supplemental Data ONLY**: Use the `search_tool` ONLY to find information **not** present in the LinkedIn data. This includes:\n"
                "    -   `key_skills_and_expertise`\n"
                "    -   `public_speaking_or_content`\n"
                "    -   Any other relevant facts to help with the assessment.\n"
                "    **Do NOT re-search for work experience or education that is already provided by the primary tool.**\n\n"
                "4.  **Perform Investment Criteria Assessment**: Based on ALL the compiled data, assess the founder against the 6 investment criteria and populate the `investment_criteria_assessment` object.\n\n"
                "5.  **Synthesize and Output Final JSON**: Combine all the information into a single, valid `FounderProfile` JSON object.\n\n"
                "Your FINAL output MUST be a single, valid JSON object that strictly conforms to the FounderProfile schema below. Do NOT include any other text, explanations, or markdown outside of this JSON object.\n{format_instructions}"
            ),
            ("human", "Please research the founder: {founder_name}. LinkedIn URL (if provided): {linkedin_url_input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]).partial(format_instructions=founder_profile_parser.get_format_instructions())

        # Create and run agent
        # The llm instance (active_llm) is now determined per attempt (potentially switched by retry logic via llm_override)
        agent = create_tool_calling_agent(llm=active_llm, prompt=prompt_template, tools=available_tools)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=available_tools,
            verbose=False, # Set to True for dev debugging
            handle_parsing_errors=True, # Important for robust JSON extraction
            max_iterations=10 # Increased to allow more detailed research steps
        )

        agent_input = {
            "founder_name": founder_name,
            "linkedin_url_input": processed_linkedin_url if processed_linkedin_url else "Not provided"
        }
        
        raw_agent_output_container = None
        json_to_parse = ""
        try:
            # This is the part that will be retried by the decorator
            raw_agent_output_container = agent_executor.invoke(agent_input)
            
            content_str = self._extract_content_string(raw_agent_output_container)
            if not content_str:
                logger.error("Agent did not produce a parsable output string for FounderProfile.")
                raise ValueError("Agent produced no parsable output string for FounderProfile.")

            logger.debug(f"Raw LLM content string (for JSON extraction): {content_str[:500]}...")

            json_to_parse = self._extract_json_from_content(content_str)
            logger.debug(f"Extracted JSON block for FounderProfile parsing: {json_to_parse[:300]}...")
            
            parsed_dict = json.loads(json_to_parse, strict=False)
            
            if "linkedin_url" in parsed_dict and isinstance(parsed_dict["linkedin_url"], str):
                parsed_dict["linkedin_url"] = self._ensure_url_scheme(parsed_dict["linkedin_url"])
            
            founder_profile = FounderProfile.model_validate(parsed_dict)
            
            logger.info(f"Successfully researched and validated profile for founder: {founder_name}")
            return founder_profile
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed for founder '{founder_name}'. Error: {e}. String that failed: '{json_to_parse[:1000]}'")
            # Add raw output to log for debugging
            if raw_agent_output_container: logger.debug(f"Raw agent output at time of JSON error: {raw_agent_output_container}")
            # This error will be caught by the retry handler
            raise ValueError(f"Failed to parse agent output as JSON for {founder_name}: {e}")
        except Exception as e:
            # Other exceptions during agent execution or validation
            logger.error(f"Research execution or validation failed for founder '{founder_name}': {e}", exc_info=True)
            if raw_agent_output_container: logger.debug(f"Raw agent output at time of error: {raw_agent_output_container}")
            # This error will also be caught by the retry handler
            raise # Re-raise the original error for the retry handler

    def _extract_content_string(self, raw_output: Any) -> Optional[str]:
        """Robustly extract content string from various agent output formats."""
        if isinstance(raw_output, dict) and "output" in raw_output:
            output_payload = raw_output["output"]
            if isinstance(output_payload, str):
                return output_payload
            elif isinstance(output_payload, list) and output_payload:
                # Handle cases like Anthropic's list of content blocks
                first_block = output_payload[0]
                if isinstance(first_block, dict) and "text" in first_block and isinstance(first_block["text"], str):
                    return first_block["text"]
                else: # Fallback for unexpected list structure
                    logger.warning(f"Unexpected structure in first block of agent output list: {first_block}")
                    return str(output_payload) 
            else: # Fallback for other unexpected structures within output['output']
                logger.warning(f"Unexpected structure in agent output['output'] payload: {output_payload}")
                return str(output_payload)
        elif isinstance(raw_output, str): # If the agent directly returns a string
             return raw_output
        else: # Fallback for other unexpected overall output structures
            logger.warning(f"Unexpected overall output structure from agent: {raw_output}")
            return str(raw_output)

    def _extract_json_from_content(self, content_str: str) -> str:
        """Extract JSON object from content string, attempting to clean it if necessary."""
        # Primary method: find a JSON object using regex
        json_match = re.search(r"\{[\s\S]*\}", content_str)
        if json_match:
            return json_match.group(0).strip()
        
        # Fallback: attempt to strip markdown and retry
        temp_str = content_str.strip()
        if temp_str.startswith("```json"):
            temp_str = temp_str[len("```json"):].strip()
        if temp_str.endswith("```"):
            temp_str = temp_str[:-len("```")].strip()
        
        if temp_str.startswith("{") and temp_str.endswith("}"):
            logger.warning("Used fallback JSON extraction (markdown stripping).")
            return temp_str
            
        logger.error(f"Could not extract a valid JSON object from LLM response: {content_str[:500]}...")
        raise ValueError("No valid JSON object found in agent output after attempting cleaning.")

    def save_results(self, founder_profile: FounderProfile, founder_name: str):
        """Save research results to file using the save_tool."""
        safe_name = "".join(c if c.isalnum() else "_" for c in founder_name)[:50].rstrip("_")
        filename = f"founder_research_{safe_name}.json"
        
        try:
            save_tool.invoke({
                "filename": filename,
                "text": founder_profile.model_dump_json(indent=2)
            })
            logger.info(f"Founder profile research results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save founder profile research results to file {filename}: {e}")

# --- Decorator Application ---
# We need to define the model_selector_func to pass to the decorator.
# This should be get_llm_for_agent if the system is available.

_model_selector_for_retry = None
if MODEL_SYSTEM_AVAILABLE:
    try:
        # Ensure get_llm_for_agent is imported correctly for this scope
        from ai_agents.config.model_config import get_llm_for_agent as get_llm_func_for_decorator
        _model_selector_for_retry = get_llm_func_for_decorator
    except ImportError:
        logger.warning("Could not import get_llm_for_agent for retry decorator's model switching.")
        MODEL_SYSTEM_AVAILABLE = False # Force false if this specific import fails

# Conditionally apply the decorator with model switching if available
if MODEL_SYSTEM_AVAILABLE and _model_selector_for_retry:
    EnhancedFounderResearchAgent.research_founder = with_smart_retry(
        agent_name="founder_research",
        retry_config_override=RetryConfig(max_retries=3, base_delay=2.0, switch_model_on_rate_limit=True),
        model_selector_func=_model_selector_for_retry # Pass the actual function
    )(EnhancedFounderResearchAgent.research_founder)
else:
    # Apply decorator without model switching if system or selector func is not available
    # Fallback: Retry without model switching
    if MODEL_SYSTEM_AVAILABLE: # Retry handler might still be available
        logger.warning("Applying retry decorator for founder_research_agent WITHOUT model switching capability (model_selector_func not available).")
        EnhancedFounderResearchAgent.research_founder = with_smart_retry(
            agent_name="founder_research",
            retry_config_override=RetryConfig(max_retries=3, base_delay=2.0, switch_model_on_rate_limit=False), # Set switch to False
            model_selector_func=None
        )(EnhancedFounderResearchAgent.research_founder)
    else:
        logger.warning("Retry system not available for EnhancedFounderResearchAgent. research_founder will not have retry capabilities.")
        # The method remains undecorated in this case.

# --- CLI Execution ---
def run_founder_research_cli_entrypoint(founder_name: str, linkedin_url: Optional[str] = None) -> Optional[FounderProfile]:
    """CLI wrapper for the enhanced founder research agent."""
    # This function is what the orchestrator or __main__ will call.
    agent_instance = EnhancedFounderResearchAgent()
    
    try:
        # The research_founder method is now potentially decorated.
        # It accepts llm_override and model_config_override for the decorator.
        profile = agent_instance.research_founder(founder_name, linkedin_url) # Initial call won't pass these overrides.
        # The save_results call is removed from the main CLI path to avoid duplicated output
        # when using stdout redirection. The __main__ block handles printing the output.
        # if profile:
        #     agent_instance.save_results(profile, founder_name)
        return profile
    except Exception as e:
        # If retries are exhausted or a non-retryable error occurs, it will be raised here.
        logger.error(f"Founder research for '{founder_name}' ultimately failed after potential retries: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    # Ensure logging is set up for CLI execution.
    # Basic config if not already set by an orchestrator/caller.
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    logger.info("Founder research agent CLI starting...")
    
    parser = argparse.ArgumentParser(description="Enhanced Founder Research Agent CLI")
    parser.add_argument("--name", type=str, required=True, help="Full name of the founder.")
    parser.add_argument("--linkedin_url", type=str, required=False, default=None, help="LinkedIn URL of the founder (optional).")
    
    args = parser.parse_args()
    
    try:
        # Call the refactored entry point
        profile_object = run_founder_research_cli_entrypoint(args.name, linkedin_url=args.linkedin_url)
        
        if profile_object:
            # Print the final FounderProfile JSON to stdout for the orchestrator
            print(profile_object.model_dump_json(indent=None)) # indent=None for cleaner capture by orchestrator
            logger.info(f"Founder research agent: Successfully generated and printed JSON output for {args.name}")
        else:
            # If research failed even after retries (or no retries configured/available).
            error_output = {
                "status": "error",
                "founder_name": args.name,
                "linkedin_url": args.linkedin_url,
                "error_message": f"Failed to retrieve or generate profile for {args.name} after all attempts."
            }
            print(json.dumps(error_output, indent=None))
            logger.error(f"Founder research agent: Failed to generate profile for {args.name}.")
            sys.exit(1) # Exit with error code
            
    except Exception as e_main:
        # Catch any other unexpected errors during the main execution flow that weren't handled by retries
        error_output = {
            "status": "error",
            "founder_name": args.name,
            "linkedin_url": args.linkedin_url,
            "error_message": f"Critical unhandled error in founder_research_agent CLI for {args.name}: {str(e_main)}"
        }
        print(json.dumps(error_output, indent=None)) # Print error JSON to stdout
        logger.critical(f"Critical unhandled error in founder_research_agent CLI for {args.name}: {e_main}", exc_info=True)
        sys.exit(1) 