import os
import json
import re
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import argparse
import sys
from urllib.parse import urlparse

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor

# Import models and tools from your project structure
from ai_agents.models.investment_research import CompanyProfile
from ai_agents.tools import search_tool, save_tool # Assuming Tavily is your primary search_tool
from ai_agents.tools import tavily_search_tool_instance # For API key check

# Load .env file from the project root
load_dotenv()

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

def get_llm():
    """Initializes and returns the appropriate LLM based on available API keys."""
    if os.getenv("ANTHROPIC_API_KEY"):
        print("Using Anthropic Claude model for company research.", file=sys.stderr)
        return ChatAnthropic(
            model="claude-3-haiku-20240307", 
            temperature=0.3,
            max_tokens=4000  # Increased max tokens
        )
    elif os.getenv("OPENAI_API_KEY"):
        print("Using OpenAI GPT model for company research.", file=sys.stderr)
        return ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.2) # Or gpt-4-turbo-preview
    else:
        print("FATAL: Neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set. Cannot initialize LLM.", file=sys.stderr)
        return None

def run_company_research_cli(company_name: str) -> Optional[CompanyProfile]:
    """
    Researches a company comprehensively and returns a CompanyProfile object.
    Saves the result to a JSON file.
    """
    print(f"Starting comprehensive research for company: \"{company_name}\"", file=sys.stderr)

    # --- API Key and Tool Availability Checks ---
    if not tavily_search_tool_instance:
        print("FATAL: Tavily search_tool is not available (check TAVILY_API_KEY in .env). Cannot proceed with company research.", file=sys.stderr)
        return None

    llm = get_llm()
    if not llm:
        return None

    available_tools = [search_tool] # Primarily uses web search

    # --- Agent and Executor Setup ---
    company_research_agent = create_tool_calling_agent(
        llm=llm,
        prompt=COMPANY_RESEARCH_PROMPT_TEMPLATE,
        tools=available_tools
    )
    agent_executor = AgentExecutor(
        agent=company_research_agent,
        tools=available_tools,
        verbose=False,
        handle_parsing_errors=True # Handles errors if LLM output is not perfect JSON for tool calls
    )

    raw_agent_output_container = None
    json_to_parse = ""
    try:
        raw_agent_output_container = agent_executor.invoke({"input": company_name})

        # Robustly extract the string content from the agent's output
        llm_response_content_str = None # This will hold the string that potentially contains the JSON
        if isinstance(raw_agent_output_container, dict) and "output" in raw_agent_output_container:
            output_payload = raw_agent_output_container["output"]
            if isinstance(output_payload, str):
                llm_response_content_str = output_payload
            elif isinstance(output_payload, list) and output_payload:
                # Handle cases where output is a list of content blocks (e.g., from Anthropic)
                first_block = output_payload[0]
                if isinstance(first_block, dict) and "text" in first_block and isinstance(first_block["text"], str):
                    llm_response_content_str = first_block["text"]
                    print(f"Extracted text content from agent output list's first block: {llm_response_content_str[:200]}...", file=sys.stderr)
                else:
                    # Fallback if the list structure is not as expected
                    print(f"Warning: Unexpected structure in first block of agent output list: {first_block}", file=sys.stderr)
                    llm_response_content_str = str(output_payload) 
            else:
                # Fallback for other unexpected structures within output['output']
                print(f"Warning: Unexpected structure in agent output['output'] payload: {output_payload}", file=sys.stderr)
                llm_response_content_str = str(output_payload)
        elif isinstance(raw_agent_output_container, str): # If the agent directly returns a string
             llm_response_content_str = raw_agent_output_container
        else:
            # Fallback for other unexpected overall output structures
            print(f"Warning: Unexpected overall output structure from agent: {raw_agent_output_container}", file=sys.stderr)
            llm_response_content_str = str(raw_agent_output_container)

        if not llm_response_content_str:
            print("Agent did not produce a parsable output string for CompanyProfile.", file=sys.stderr)
            return None

        print(f"\nRaw LLM content string (for JSON extraction): {llm_response_content_str[:500]}...", file=sys.stderr)

        # Attempt to extract the JSON block from the content string
        # The LLM is expected to return JSON, possibly wrapped in some text (e.g. <result> or ```json)
        json_match = re.search(r"\{[\s\S]*\}", llm_response_content_str) # Use [\s\S]* for robust multiline match
        if json_match:
            json_to_parse = json_match.group(0).strip()
            print(f"Extracted JSON block for CompanyProfile parsing (stripped): {json_to_parse[:300]}...", file=sys.stderr)
        else:
            # Fallback if regex doesn't find a clear JSON object.
            temp_str = llm_response_content_str.strip()
            # Remove ```json ... ``` markdown
            if temp_str.startswith("```json"):
                temp_str = temp_str[len("```json"):].strip()
            if temp_str.endswith("```"):
                temp_str = temp_str[:-len("```")].strip()
            
            if temp_str.startswith("{") and temp_str.endswith("}"):
                json_to_parse = temp_str
                print(f"Warning: No clear JSON block found by primary regex, attempting to parse potentially cleaned string: {json_to_parse[:300]}...", file=sys.stderr)
            else:
                cleaned_further = re.sub(r"<[^>]+>", "", temp_str).strip() # Remove all XML-like tags
                if cleaned_further.startswith("{") and cleaned_further.endswith("}"):
                    json_to_parse = cleaned_further
                    print(f"Warning: Cleaned XML-like tags, attempting to parse: {json_to_parse[:300]}...", file=sys.stderr)
                else:
                    print(f"Error: Could not extract a valid JSON object from LLM response: {llm_response_content_str[:500]}...", file=sys.stderr)
                    if raw_agent_output_container: print("Raw agent output at time of error:", raw_agent_output_container, file=sys.stderr)
                    return None
        
        # Parse the JSON string into CompanyProfile object
        parsed_dict = json.loads(json_to_parse, strict=False) # strict=False for leniency with newlines in strings

        # Ensure HttpUrl fields have a scheme before validation
        for url_field in ["website", "linkedin_url"]: # Add other HttpUrl fields if any
            if url_field in parsed_dict and isinstance(parsed_dict[url_field], str):
                parsed_dict[url_field] = _ensure_url_scheme(parsed_dict[url_field])
                print(f"Ensured scheme for {url_field} from LLM output: {parsed_dict[url_field]}", file=sys.stderr)
        
        company_profile_obj = CompanyProfile.model_validate(parsed_dict)

        print("\n--- Company Profile Research Output (to be saved to file) ---", file=sys.stderr)
        print(company_profile_obj.model_dump_json(indent=2), file=sys.stderr)

        # Save the structured data
        safe_company_name = "".join(c if c.isalnum() else "_" for c in company_name)[:50].rstrip("_")
        output_filename = f"company_research_{safe_company_name}.json"
        
        try:
            save_tool.invoke({
                "filename": output_filename,
                "text": company_profile_obj.model_dump_json(indent=2)
            })
            print(f"Company profile research saved to {output_filename}", file=sys.stderr)
        except Exception as e_save:
            print(f"Error saving company profile research to file: {e_save}", file=sys.stderr)

        return company_profile_obj

    except json.JSONDecodeError as json_err:
        print(f"JSONDecodeError: Failed to parse JSON string for CompanyProfile. Error: {json_err}", file=sys.stderr)
        print(f"String that failed parsing (up to 1000 chars): {json_to_parse[:1000]}", file=sys.stderr)
        if raw_agent_output_container: print("Raw agent output at time of error:", raw_agent_output_container, file=sys.stderr)
        return None
    except Exception as e:
        print(f"An error occurred during company research agent execution or response parsing: {e}", file=sys.stderr)
        if raw_agent_output_container: print("Raw agent output at time of error:", raw_agent_output_container, file=sys.stderr)
        return None

if __name__ == "__main__":
    print("Company research agent CLI starting...", file=sys.stderr)
    
    parser = argparse.ArgumentParser(description="Company Research Agent CLI - Researches a company by name.")
    # The orchestrator calls this with --company_name, but the agent prompt uses "input" for company_name.
    # Let's use --company_name for the CLI argument for consistency with the orchestrator.
    parser.add_argument("--company_name", type=str, required=True, help="Name of the company to research.")
    
    args = parser.parse_args()

    try:
        profile = run_company_research_cli(args.company_name)
        
        if profile:
            # Print the final CompanyProfile JSON to stdout for the orchestrator
            print(profile.model_dump_json(indent=None)) # indent=None for cleaner capture
            print(f"Company research agent: Successfully generated JSON output for {args.company_name}", file=sys.stderr)
        else:
            # If research failed, print an error JSON to stdout and exit with error code
            error_output = {
                "status": "error",
                "company_name": args.company_name,
                "error_message": f"Failed to retrieve or generate profile for {args.company_name}."
            }
            print(json.dumps(error_output, indent=None))
            print(f"Company research agent: Failed to generate profile for {args.company_name}.", file=sys.stderr)
            sys.exit(1)
            
    except Exception as e_main:
        # Catch any other unexpected errors during the main execution
        error_output = {
            "status": "error",
            "company_name": args.company_name,
            "error_message": f"Critical error in company_research_agent CLI for {args.company_name}: {str(e_main)}"
        }
        print(json.dumps(error_output, indent=None)) # Print error JSON to stdout
        print(f"Critical error in company_research_agent CLI for {args.company_name}: {e_main}", file=sys.stderr) # To stderr
        sys.exit(1) 