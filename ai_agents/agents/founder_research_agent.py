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
from ai_agents.models.investment_research import (
    FounderProfile,
    FounderCriteriaAssessment,
    PreviousCompany,
    EducationDetail
)
from ai_agents.tools import (
    search_tool, # Tavily
    research_prospect_tool, # Relevance AI
    save_tool
)
from ai_agents.tools import tavily_search_tool_instance, relevance_ai_tool_configured # For API key checks

# Load .env file from the project root
load_dotenv()

# --- Output Parser for FounderProfile ---
founder_profile_parser = PydanticOutputParser(pydantic_object=FounderProfile)

# --- Prompt Template for Founder Research ---
FOUNDER_RESEARCH_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a specialized founder research AI. Your goal is to gather comprehensive information about a given founder and structure it into a JSON object adhering to the FounderProfile schema provided below.\n"
            "You have access to 'search_tool' for general web research and 'research_prospect_tool' for detailed LinkedIn profile analysis.\n\n"
            "Research Strategy:\n"
            "1. First, use 'search_tool' to find general information about the founder. A key objective of this initial search is to locate the founder's LinkedIn profile URL if it was not provided in the initial input. If you find a LinkedIn URL, ensure it is captured.\n"
            "2. **Crucial Step for LinkedIn Data**: If a LinkedIn URL is now known (either because it was provided in the input OR you found it in step 1 using 'search_tool'), you MUST then use the 'research_prospect_tool' with that LinkedIn URL. This tool is the primary source for detailed LinkedIn profile information (like work experience duration, education details, skills) and the 'Investment Criteria' assessment. Do NOT skip this step if a LinkedIn URL is available and the tool is configured. The output from this tool is vital.\n"
            "3. After attempting to use 'research_prospect_tool' (if a LinkedIn URL was available), or if no LinkedIn URL could be found/used, use 'search_tool' again if necessary to find any remaining supplementary information to complete the FounderProfile (e.g., public speaking engagements, articles not found on LinkedIn, or to fill gaps if 'research_prospect_tool' failed or its data was incomplete for some reason).\n"
            "4. Synthesize information from ALL tool outputs (responses from 'search_tool' and 'research_prospect_tool') to populate the FounderProfile. When data is available from 'research_prospect_tool', prioritize it for LinkedIn-specific fields (education, experience, skills) and the 'investment_criteria_assessment'.\n\n"
            "Key Information to Extract for FounderProfile:\n"
            "- Full name, role in their current primary company.\n"
            "- LinkedIn profile URL (verify or find if not directly provided, and ensure this is populated if found).\n"
            "- Comprehensive background summary.\n"
            "- Detailed list of previous companies: name, role, duration, description, and whether they were a founder. (Prioritize 'research_prospect_tool' for this if used)\n"
            "- Detailed list of education: institution, degree, field, graduation year, achievements. (Prioritize 'research_prospect_tool' for this if used)\n"
            "- Key skills and expertise. (Prioritize 'research_prospect_tool' for this if used)\n"
            "- Links to or descriptions of public content (articles, talks).\n\n"
            "CRITICAL: Flight Story Investment Criteria Assessment (populate 'investment_criteria_assessment' field):\n"
            "   This assessment is BEST obtained from the 'research_prospect_tool'. If that tool was used and provided an 'Investment Criteria' section in its output, meticulously extract the answers to these 6 questions and map them to the boolean fields in the 'FounderCriteriaAssessment' sub-model. These are:\n"
            "   - Focus Industry Fit (Media, Brand, Tech, Creator Economy for their venture)\n"
            "   - Mission Alignment (venture aligned to healthier, happier humanity; impact-driven)\n"
            "   - Exciting Solution to a Problem (idea is a good solution to a real problem, would excite S. Steven Bartlett)\n"
            "   - Founded Something Relevant Before (founded something impressive and relevant previously)\n"
            "   - Impressive, Relevant Past Experience (worked somewhere impressive/relevant, making them a better founder)\n"
            "   - Exceptionally Smart or Strategic (evidence of being super smart/strategic from education, job, content)\n"
            "   If 'research_prospect_tool' could not be used (e.g., no LinkedIn URL was found or provided, or the tool itself reported an error) or if it was used but did not provide a structured 'Investment Criteria' section, only then should you attempt to infer these answers based on general web search results from 'search_tool'. When inferring, be conservative and use null or false if unsure. Clearly state in 'assessment_summary' how the criteria were assessed (e.g., 'Based on structured output from research_prospect_tool', 'Inferred from web research as research_prospect_tool was not used due to no LinkedIn URL', 'Inferred from web research as research_prospect_tool output was missing this section', or 'research_prospect_tool failed').\n\n"
            "Your FINAL output MUST be a single, valid JSON object that strictly conforms to the FounderProfile schema below. Do NOT include any other text, explanations, or markdown outside of this JSON object.\n"
            "If specific information cannot be found, set its value to null or omit it if optional. Be careful with URLs and dates.\n"
            "Schema:\n{format_instructions}"
        ),
        # Input will be a dictionary: {"founder_name": "...", "linkedin_url": "..." or None}
        # We'll format this into a string for the human message.
        ("human", "Please research the founder: {founder_name}. LinkedIn URL (if provided): {linkedin_url_input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
).partial(format_instructions=founder_profile_parser.get_format_instructions())


def _ensure_url_scheme(url: Optional[str]) -> Optional[str]:
    """Prepends https:// to a URL if it's missing a scheme."""
    if url and not urlparse(url).scheme and (url.startswith("www.") or "." in url.split("/")[0]):
        return f"https://{url}"
    return url

def get_llm():
    """Initializes and returns the appropriate LLM based on available API keys."""
    if os.getenv("ANTHROPIC_API_KEY"):
        print("Using Anthropic Claude model for founder research.", file=sys.stderr)
        return ChatAnthropic(
            model="claude-3-haiku-20240307", 
            temperature=0.2, 
            max_tokens=4000  # Increased max tokens
        )
    elif os.getenv("OPENAI_API_KEY"):
        print("Using OpenAI GPT model for founder research.", file=sys.stderr)
        return ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.1)
    else:
        print("FATAL: Neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set. Cannot initialize LLM.", file=sys.stderr)
        return None

def run_founder_research_cli(founder_name: str, linkedin_url: Optional[str] = None) -> Optional[FounderProfile]:
    """
    Researches a founder comprehensively and returns a FounderProfile object.
    Saves the result to a JSON file.
    """
    print(f"Starting comprehensive research for founder: \"{founder_name}\"", file=sys.stderr)
    
    processed_linkedin_url = _ensure_url_scheme(linkedin_url) # Ensure scheme for tool usage

    if processed_linkedin_url:
        print(f"Provided LinkedIn URL (processed): {processed_linkedin_url}", file=sys.stderr)

    # --- API Key and Tool Availability Checks ---
    if not tavily_search_tool_instance:
        print("Warning: Tavily search_tool is not available (check TAVILY_API_KEY). Web search capabilities will be limited.", file=sys.stderr)
    
    can_use_prospect_tool = bool(processed_linkedin_url and relevance_ai_tool_configured) # Use processed URL
    if processed_linkedin_url and not relevance_ai_tool_configured: # Use processed URL
        print("Warning: LinkedIn URL provided, but Relevance AI 'Research Prospect' tool is not configured. Will rely on general web search.", file=sys.stderr)
    
    llm = get_llm()
    if not llm:
        return None

    available_tools = [search_tool]
    if relevance_ai_tool_configured: # Only add if configured, agent will decide to use it based on prompt
        available_tools.append(research_prospect_tool)
    
    if not available_tools:
        print("FATAL: No research tools are available. Cannot proceed.", file=sys.stderr)
        return None

    # --- Agent and Executor Setup ---
    founder_research_agent = create_tool_calling_agent(
        llm=llm,
        prompt=FOUNDER_RESEARCH_PROMPT_TEMPLATE,
        tools=available_tools
    )
    agent_executor = AgentExecutor(
        agent=founder_research_agent,
        tools=available_tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=10 # Allow for a few steps of tool use and synthesis
    )

    agent_input = {
        "founder_name": founder_name,
        "linkedin_url_input": processed_linkedin_url if processed_linkedin_url else "Not provided" # Pass processed URL to agent
    }
    
    raw_agent_output_container = None
    json_to_parse = "" # Initialize to ensure it's defined for error messages
    try:
        raw_agent_output_container = agent_executor.invoke(agent_input)
        
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
            print("Agent did not produce a parsable output string for FounderProfile.", file=sys.stderr)
            return None

        print(f"\nRaw LLM content string (for JSON extraction): {llm_response_content_str[:500]}...", file=sys.stderr)

        # Attempt to extract the JSON block from the content string
        json_match = re.search(r"\{[\s\S]*\}", llm_response_content_str) # Use [\s\S]* for robust multiline match
        if json_match:
            json_to_parse = json_match.group(0).strip()
            print(f"Extracted JSON block for FounderProfile parsing (stripped): {json_to_parse[:300]}...", file=sys.stderr)
        else:
            # Fallback if regex doesn't find a clear JSON object.
            temp_str = llm_response_content_str.strip()
            if temp_str.startswith("```json"):
                temp_str = temp_str[len("```json"):].strip()
            if temp_str.endswith("```"):
                temp_str = temp_str[:-len("```")].strip()
            
            if temp_str.startswith("{") and temp_str.endswith("}"):
                json_to_parse = temp_str
                print(f"Warning: No clear JSON block found by primary regex, attempting to parse potentially cleaned string: {json_to_parse[:300]}...", file=sys.stderr)
            else:
                print(f"Error: Could not extract a valid JSON object from LLM response: {llm_response_content_str[:500]}...", file=sys.stderr)
                if raw_agent_output_container: print("Raw agent output at time of error:", raw_agent_output_container, file=sys.stderr)
                return None
        
        # Parse the JSON string into FounderProfile object
        parsed_dict = json.loads(json_to_parse, strict=False) # strict=False for leniency with newlines in strings
        
        # Ensure linkedin_url in the parsed_dict has a scheme before validation
        if "linkedin_url" in parsed_dict and isinstance(parsed_dict["linkedin_url"], str):
            parsed_dict["linkedin_url"] = _ensure_url_scheme(parsed_dict["linkedin_url"])
            print(f"Ensured scheme for linkedin_url from LLM output: {parsed_dict['linkedin_url']}", file=sys.stderr)

        founder_profile_obj = FounderProfile.model_validate(parsed_dict)

        print("\n--- Founder Profile Research Output (to be saved to file) ---", file=sys.stderr)
        print(founder_profile_obj.model_dump_json(indent=2), file=sys.stderr)

        safe_founder_name = "".join(c if c.isalnum() else "_" for c in founder_name)[:50].rstrip("_")
        output_filename = f"founder_research_{safe_founder_name}.json"
        
        try:
            save_tool.invoke({
                "filename": output_filename,
                "text": founder_profile_obj.model_dump_json(indent=2)
            })
            print(f"Founder profile research saved to {output_filename}", file=sys.stderr)
        except Exception as e_save:
            print(f"Error saving founder profile research to file: {e_save}", file=sys.stderr)

        return founder_profile_obj

    except json.JSONDecodeError as json_err:
        print(f"JSONDecodeError: Failed to parse JSON string for FounderProfile. Error: {json_err}", file=sys.stderr)
        print(f"String that failed parsing (up to 1000 chars): {json_to_parse[:1000]}", file=sys.stderr)
        if raw_agent_output_container: print("Raw agent output at time of error:", raw_agent_output_container, file=sys.stderr)
        return None
    except Exception as e:
        print(f"An error occurred during founder research agent execution or response parsing: {e}", file=sys.stderr)
        if raw_agent_output_container: print("Raw agent output at time of error:", raw_agent_output_container, file=sys.stderr)
        return None

if __name__ == "__main__":
    # All prints in __main__ should go to stderr except the final JSON output
    print("Founder research agent CLI starting...", file=sys.stderr)
    
    parser = argparse.ArgumentParser(description="Founder Research Agent CLI - Processes founder name and LinkedIn URL.")
    parser.add_argument("--name", type=str, required=True, help="Full name of the founder.")
    parser.add_argument("--linkedin_url", type=str, required=False, default=None, help="LinkedIn URL of the founder (optional).")
    
    args = parser.parse_args()

    try:
        profile_object = run_founder_research_cli(args.name, linkedin_url=args.linkedin_url)
        
        if profile_object:
            # Print the final FounderProfile JSON to stdout for the orchestrator
            print(profile_object.model_dump_json(indent=None)) # indent=None for cleaner capture
            print(f"Founder research agent: Successfully generated JSON output for {args.name}", file=sys.stderr)
        else:
            # If research failed, print an error JSON to stdout and exit with error code
            error_output = {
                "status": "error",
                "founder_name": args.name,
                "linkedin_url": args.linkedin_url,
                "error_message": f"Failed to retrieve or generate profile for {args.name}."
            }
            print(json.dumps(error_output, indent=None))
            print(f"Founder research agent: Failed to generate profile for {args.name}.", file=sys.stderr)
            sys.exit(1)
            
    except Exception as e_main:
        # Catch any other unexpected errors during the main execution
        error_output = {
            "status": "error",
            "founder_name": args.name,
            "linkedin_url": args.linkedin_url,
            "error_message": f"Critical error in founder_research_agent CLI for {args.name}: {str(e_main)}"
        }
        print(json.dumps(error_output, indent=None)) # Print error JSON to stdout
        print(f"Critical error in founder_research_agent CLI for {args.name}: {e_main}", file=sys.stderr)
        sys.exit(1) 