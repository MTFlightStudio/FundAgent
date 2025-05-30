import os
import json
import re
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

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


def get_llm():
    """Initializes and returns the appropriate LLM based on available API keys."""
    if os.getenv("ANTHROPIC_API_KEY"):
        print("Using Anthropic Claude model for company research.")
        return ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.3)
    elif os.getenv("OPENAI_API_KEY"):
        print("Using OpenAI GPT model for company research.")
        return ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.2) # Or gpt-4-turbo-preview
    else:
        print("FATAL: Neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set. Cannot initialize LLM.")
        return None

def run_company_research_cli(company_name: str) -> Optional[CompanyProfile]:
    """
    Researches a company comprehensively and returns a CompanyProfile object.
    Saves the result to a JSON file.
    """
    print(f"Starting comprehensive research for company: \"{company_name}\"")

    # --- API Key and Tool Availability Checks ---
    if not tavily_search_tool_instance:
        print("FATAL: Tavily search_tool is not available (check TAVILY_API_KEY in .env). Cannot proceed with company research.")
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
        verbose=True,
        handle_parsing_errors=True # Handles errors if LLM output is not perfect JSON for tool calls
    )

    raw_agent_output_container = None # To store raw output for debugging
    try:
        raw_agent_output_container = agent_executor.invoke({"input": company_name})
        
        llm_response_str = None
        if isinstance(raw_agent_output_container, dict) and "output" in raw_agent_output_container:
            output_content = raw_agent_output_container["output"]
            if isinstance(output_content, str):
                llm_response_str = output_content
            # Add other extraction logic if needed, similar to research_agent.py
            else:
                print(f"Warning: Unexpected structure in agent output['output']: {output_content}")
                llm_response_str = str(output_content) # Fallback
        else:
            print(f"Warning: Unexpected output structure from agent: {raw_agent_output_container}")
            llm_response_str = str(raw_agent_output_container)

        if not llm_response_str:
            print("Agent did not produce a parsable output string for CompanyProfile.")
            return None

        print(f"\nRaw LLM response string (before JSON extraction for CompanyProfile): {llm_response_str[:500]}...")

        # Attempt to extract the JSON block
        json_match = re.search(r"\{.*\}", llm_response_str, re.DOTALL)
        if json_match:
            json_to_parse = json_match.group(0).strip()
            print(f"Extracted JSON block for CompanyProfile parsing (stripped): {json_to_parse[:300]}...")
        else:
            json_to_parse = llm_response_str.strip() # Fallback if no clear JSON block
            print(f"Warning: No clear JSON block found, attempting to parse whole string for CompanyProfile (stripped): {json_to_parse[:300]}...")

        # Parse the JSON string into CompanyProfile object
        parsed_dict = json.loads(json_to_parse, strict=False) # strict=False for leniency with newlines in strings
        company_profile_obj = CompanyProfile.model_validate(parsed_dict)

        print("\n--- Company Profile Research Output ---")
        print(company_profile_obj.model_dump_json(indent=2))

        # Save the structured data
        safe_company_name = "".join(c if c.isalnum() else "_" for c in company_name)[:50].rstrip("_")
        output_filename = f"company_research_{safe_company_name}.json"
        
        try:
            save_tool.invoke({
                "filename": output_filename,
                "text": company_profile_obj.model_dump_json(indent=2)
            })
            print(f"Company profile research saved to {output_filename}")
        except Exception as e_save:
            print(f"Error saving company profile research to file: {e_save}")

        return company_profile_obj

    except json.JSONDecodeError as json_err:
        print(f"JSONDecodeError: Failed to parse JSON string for CompanyProfile. Error: {json_err}")
        print(f"String that failed parsing (up to 1000 chars): {json_to_parse[:1000] if 'json_to_parse' in locals() else 'N/A'}")
        if raw_agent_output_container: print("Raw agent output at time of error:", raw_agent_output_container)
        return None
    except Exception as e:
        print(f"An error occurred during company research agent execution or response parsing: {e}")
        if raw_agent_output_container: print("Raw agent output at time of error:", raw_agent_output_container)
        return None

if __name__ == "__main__":
    print("Testing company_research_agent.py...")
    # company_to_research = "OpenAI"
    # company_to_research = "Stripe"
    # company_to_research = "Gusto"
    company_to_research = input("Enter a company name for research: ")

    if company_to_research:
        profile = run_company_research_cli(company_to_research)
        if profile:
            print(f"\nSuccessfully retrieved profile for {profile.company_name}")
            # print(f"Website: {profile.website}")
            # print(f"Total Funding: {profile.total_funding_raised}")
        else:
            print(f"Failed to retrieve profile for {company_to_research}")
    else:
        print("No company name provided for test.") 