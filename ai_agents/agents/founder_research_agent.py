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
            "You have access to 'search_tool' for general web research and 'research_prospect_tool' for detailed LinkedIn profile analysis (if a LinkedIn URL is provided).\n\n"
            "Research Strategy:\n"
            "1. If a LinkedIn URL is provided in the input, STRONGLY PREFER using 'research_prospect_tool' with that URL. Its output often contains structured data, including a crucial 'Investment Criteria' assessment (6 yes/no questions). Extract this assessment carefully.\n"
            "2. ALWAYS use 'search_tool' to find supplementary information, such as: founder's background, detailed history of previous companies (especially if they were a founder), educational credentials, notable achievements, public speaking, articles, or other thought leadership content.\n"
            "3. Synthesize information from all tool outputs to populate the FounderProfile.\n\n"
            "Key Information to Extract for FounderProfile:\n"
            "- Full name, role in their current primary company.\n"
            "- LinkedIn profile URL (verify or find if not directly provided).\n"
            "- Comprehensive background summary.\n"
            "- Detailed list of previous companies: name, role, duration, description, and whether they were a founder.\n"
            "- Detailed list of education: institution, degree, field, graduation year, achievements.\n"
            "- Key skills and expertise.\n"
            "- Links to or descriptions of public content (articles, talks).\n\n"
            "CRITICAL: Flight Story Investment Criteria Assessment (populate 'investment_criteria_assessment' field):\n"
            "   If 'research_prospect_tool' was used and provided an 'Investment Criteria' section, meticulously extract the answers to these 6 questions and map them to the boolean fields in the 'FounderCriteriaAssessment' sub-model. These are:\n"
            "   - Focus Industry Fit (Media, Brand, Tech, Creator Economy for their venture)\n"
            "   - Mission Alignment (venture aligned to healthier, happier humanity; impact-driven)\n"
            "   - Exciting Solution to a Problem (idea is a good solution to a real problem, would excite S. Stephen Bartlett)\n"
            "   - Founded Something Relevant Before (founded something impressive and relevant previously)\n"
            "   - Impressive, Relevant Past Experience (worked somewhere impressive/relevant, making them a better founder)\n"
            "   - Exceptionally Smart or Strategic (evidence of being super smart/strategic from education, job, content)\n"
            "   If this structured assessment is not available from 'research_prospect_tool', try to infer these based on general web search results, but be conservative and use null/false if unsure. Clearly state in 'assessment_summary' how the criteria were assessed (e.g., 'Based on structured tool output' or 'Inferred from web research').\n\n"
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


def get_llm():
    """Initializes and returns the appropriate LLM based on available API keys."""
    if os.getenv("ANTHROPIC_API_KEY"):
        print("Using Anthropic Claude model for founder research.")
        return ChatAnthropic(
            model="claude-3-haiku-20240307", 
            temperature=0.2, 
            max_tokens=4000  # Increased max tokens
        )
    elif os.getenv("OPENAI_API_KEY"):
        print("Using OpenAI GPT model for founder research.")
        return ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.1)
    else:
        print("FATAL: Neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set. Cannot initialize LLM.")
        return None

def run_founder_research_cli(founder_name: str, linkedin_url: Optional[str] = None) -> Optional[FounderProfile]:
    """
    Researches a founder comprehensively and returns a FounderProfile object.
    Saves the result to a JSON file.
    """
    print(f"Starting comprehensive research for founder: \"{founder_name}\"")
    if linkedin_url:
        print(f"Provided LinkedIn URL: {linkedin_url}")

    # --- API Key and Tool Availability Checks ---
    if not tavily_search_tool_instance:
        print("Warning: Tavily search_tool is not available (check TAVILY_API_KEY). Web search capabilities will be limited.")
    
    can_use_prospect_tool = bool(linkedin_url and relevance_ai_tool_configured)
    if linkedin_url and not relevance_ai_tool_configured:
        print("Warning: LinkedIn URL provided, but Relevance AI 'Research Prospect' tool is not configured. Will rely on general web search.")
    
    llm = get_llm()
    if not llm:
        return None

    available_tools = [search_tool]
    if relevance_ai_tool_configured: # Only add if configured, agent will decide to use it based on prompt
        available_tools.append(research_prospect_tool)
    
    if not available_tools:
        print("FATAL: No research tools are available. Cannot proceed.")
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
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10 # Allow for a few steps of tool use and synthesis
    )

    agent_input = {
        "founder_name": founder_name,
        "linkedin_url_input": linkedin_url if linkedin_url else "Not provided"
    }
    
    raw_agent_output_container = None
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
                    print(f"Extracted text content from agent output list's first block: {llm_response_content_str[:200]}...")
                else:
                    # Fallback if the list structure is not as expected
                    print(f"Warning: Unexpected structure in first block of agent output list: {first_block}")
                    llm_response_content_str = str(output_payload) 
            else:
                # Fallback for other unexpected structures within output['output']
                print(f"Warning: Unexpected structure in agent output['output'] payload: {output_payload}")
                llm_response_content_str = str(output_payload)
        elif isinstance(raw_agent_output_container, str): # If the agent directly returns a string
             llm_response_content_str = raw_agent_output_container
        else:
            # Fallback for other unexpected overall output structures
            print(f"Warning: Unexpected overall output structure from agent: {raw_agent_output_container}")
            llm_response_content_str = str(raw_agent_output_container)

        if not llm_response_content_str:
            print("Agent did not produce a parsable output string for FounderProfile.")
            return None

        print(f"\nRaw LLM content string (for JSON extraction): {llm_response_content_str[:500]}...")

        # Attempt to extract the JSON block from the content string
        json_match = re.search(r"\{[\s\S]*\}", llm_response_content_str) # Use [\s\S]* for robust multiline match
        if json_match:
            json_to_parse = json_match.group(0).strip()
            print(f"Extracted JSON block for FounderProfile parsing (stripped): {json_to_parse[:300]}...")
        else:
            # Fallback if regex doesn't find a clear JSON object.
            temp_str = llm_response_content_str.strip()
            if temp_str.startswith("```json"):
                temp_str = temp_str[len("```json"):].strip()
            if temp_str.endswith("```"):
                temp_str = temp_str[:-len("```")].strip()
            
            if temp_str.startswith("{") and temp_str.endswith("}"):
                json_to_parse = temp_str
                print(f"Warning: No clear JSON block found by primary regex, attempting to parse potentially cleaned string: {json_to_parse[:300]}...")
            else:
                print(f"Error: Could not extract a valid JSON object from LLM response: {llm_response_content_str[:500]}...")
                if raw_agent_output_container: print("Raw agent output at time of error:", raw_agent_output_container)
                return None
        
        # Parse the JSON string into FounderProfile object
        parsed_dict = json.loads(json_to_parse, strict=False) # strict=False for leniency with newlines in strings
        founder_profile_obj = FounderProfile.model_validate(parsed_dict)

        print("\n--- Founder Profile Research Output ---")
        print(founder_profile_obj.model_dump_json(indent=2))

        safe_founder_name = "".join(c if c.isalnum() else "_" for c in founder_name)[:50].rstrip("_")
        output_filename = f"founder_research_{safe_founder_name}.json"
        
        try:
            save_tool.invoke({
                "filename": output_filename,
                "text": founder_profile_obj.model_dump_json(indent=2)
            })
            print(f"Founder profile research saved to {output_filename}")
        except Exception as e_save:
            print(f"Error saving founder profile research to file: {e_save}")

        return founder_profile_obj

    except json.JSONDecodeError as json_err:
        print(f"JSONDecodeError: Failed to parse JSON string for FounderProfile. Error: {json_err}")
        print(f"String that failed parsing (up to 1000 chars): {json_to_parse[:1000] if 'json_to_parse' in locals() else 'N/A'}")
        if raw_agent_output_container: print("Raw agent output at time of error:", raw_agent_output_container)
        return None
    except Exception as e:
        print(f"An error occurred during founder research agent execution or response parsing: {e}")
        if raw_agent_output_container: print("Raw agent output at time of error:", raw_agent_output_container)
        return None

if __name__ == "__main__":
    print("Testing founder_research_agent.py...")
    
    test_founder_name = input("Enter founder's name for research (e.g., 'Elon Musk'): ")
    test_linkedin_url = input("Enter founder's LinkedIn URL (optional, press Enter to skip): ")
    
    if not test_linkedin_url.strip():
        test_linkedin_url = None

    if test_founder_name:
        profile = run_founder_research_cli(test_founder_name, linkedin_url=test_linkedin_url)
        if profile:
            print(f"\nSuccessfully retrieved profile for {profile.name}")
            # if profile.investment_criteria_assessment:
            #     print(f"Mission Alignment: {profile.investment_criteria_assessment.mission_alignment}")
        else:
            print(f"Failed to retrieve profile for {test_founder_name}")
    else:
        print("No founder name provided for test.") 