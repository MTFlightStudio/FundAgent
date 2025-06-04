import os
import json
import re
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import argparse
import sys

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor

# Import models and tools from your project structure
from ai_agents.models.investment_research import MarketAnalysis, CompetitorInfo
from ai_agents.tools import search_tool, wiki_tool, save_tool
from ai_agents.tools import tavily_search_tool_instance, wiki_tool_instance

# Load .env file from the project root
load_dotenv()

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


def get_llm():
    """Initializes and returns the appropriate LLM based on available API keys."""
    if os.getenv("ANTHROPIC_API_KEY"):
        print("Using Anthropic Claude model for market intelligence.", file=sys.stderr)
        return ChatAnthropic(
            model="claude-3-haiku-20240307", 
            temperature=0.3,
            max_tokens=4000
        )
    elif os.getenv("OPENAI_API_KEY"):
        print("Using OpenAI GPT model for market intelligence.", file=sys.stderr)
        return ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.2)
    else:
        print("FATAL: Neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set. Cannot initialize LLM.", file=sys.stderr)
        return None


def run_market_intelligence_cli(market_or_industry: str) -> Optional[MarketAnalysis]:
    """
    Researches a market or industry comprehensively and returns a MarketAnalysis object.
    Saves the result to a JSON file.
    
    Args:
        market_or_industry: The market or industry to research (e.g., "AI-powered fintech", "Creator economy tools")
    
    Returns:
        MarketAnalysis object if successful, None otherwise
    """
    print(f"Starting comprehensive market intelligence for: \"{market_or_industry}\"", file=sys.stderr)

    # --- API Key and Tool Availability Checks ---
    if not tavily_search_tool_instance:
        print("FATAL: Tavily search_tool is not available (check TAVILY_API_KEY in .env). Cannot proceed with market research.", file=sys.stderr)
        return None

    if not wiki_tool_instance:
        print("Warning: Wikipedia tool is not available. Market research will rely solely on web search.", file=sys.stderr)

    llm = get_llm()
    if not llm:
        return None

    # Build list of available tools
    available_tools = [search_tool]
    if wiki_tool_instance:
        available_tools.append(wiki_tool)

    # --- Agent and Executor Setup ---
    market_intelligence_agent = create_tool_calling_agent(
        llm=llm,
        prompt=MARKET_INTELLIGENCE_PROMPT_TEMPLATE,
        tools=available_tools
    )
    
    agent_executor = AgentExecutor(
        agent=market_intelligence_agent,
        tools=available_tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=15  # Allow more iterations for comprehensive market research
    )

    raw_agent_output_container = None
    json_to_parse = ""
    try:
        raw_agent_output_container = agent_executor.invoke({"input": market_or_industry})
        
        # Robustly extract the string content from the agent's output
        llm_response_content_str = None
        if isinstance(raw_agent_output_container, dict) and "output" in raw_agent_output_container:
            output_payload = raw_agent_output_container["output"]
            if isinstance(output_payload, str):
                llm_response_content_str = output_payload
            elif isinstance(output_payload, list) and output_payload:
                first_block = output_payload[0]
                if isinstance(first_block, dict) and "text" in first_block and isinstance(first_block["text"], str):
                    llm_response_content_str = first_block["text"]
                    print(f"Extracted text content from agent output list's first block: {llm_response_content_str[:200]}...", file=sys.stderr)
                else:
                    print(f"Warning: Unexpected structure in first block of agent output list: {first_block}", file=sys.stderr)
                    llm_response_content_str = str(output_payload) 
            else:
                print(f"Warning: Unexpected structure in agent output['output'] payload: {output_payload}", file=sys.stderr)
                llm_response_content_str = str(output_payload)
        elif isinstance(raw_agent_output_container, str):
             llm_response_content_str = raw_agent_output_container
        else:
            print(f"Warning: Unexpected overall output structure from agent: {raw_agent_output_container}", file=sys.stderr)
            llm_response_content_str = str(raw_agent_output_container)

        if not llm_response_content_str:
            print("Agent did not produce a parsable output string for MarketAnalysis.", file=sys.stderr)
            return None

        print(f"\nRaw LLM content string (for JSON extraction): {llm_response_content_str[:500]}...", file=sys.stderr)

        json_match = re.search(r"\{[\s\S]*\}", llm_response_content_str)
        if json_match:
            json_to_parse = json_match.group(0).strip()
            print(f"Extracted JSON block for MarketAnalysis parsing (stripped): {json_to_parse[:300]}...", file=sys.stderr)
        else:
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
        
        parsed_dict = json.loads(json_to_parse, strict=False)
        market_analysis_obj = MarketAnalysis.model_validate(parsed_dict)

        print("\n--- Market Intelligence Output (to be saved to file) ---", file=sys.stderr)
        print(market_analysis_obj.model_dump_json(indent=2), file=sys.stderr)

        safe_market_name = "".join(c if c.isalnum() else "_" for c in market_or_industry)[:50].rstrip("_")
        output_filename = f"market_analysis_{safe_market_name}.json"
        
        try:
            save_tool.invoke({
                "filename": output_filename,
                "text": market_analysis_obj.model_dump_json(indent=2)
            })
            print(f"\nMarket analysis saved to {output_filename}", file=sys.stderr)
        except Exception as e_save:
            print(f"Error saving market analysis to file: {e_save}", file=sys.stderr)

        # Print summary of key findings to stderr
        print("\n--- Market Intelligence Summary (for logs) ---", file=sys.stderr)
        if market_analysis_obj.industry_overview:
            print(f"Industry Overview: {market_analysis_obj.industry_overview[:200]}...", file=sys.stderr)
        if market_analysis_obj.market_size_tam:
            print(f"Total Addressable Market (TAM): {market_analysis_obj.market_size_tam}", file=sys.stderr)
        if market_analysis_obj.market_growth_rate_cagr:
            print(f"Market Growth Rate (CAGR): {market_analysis_obj.market_growth_rate_cagr}", file=sys.stderr)
        if market_analysis_obj.market_timing_assessment:
            print(f"Market Timing: {market_analysis_obj.market_timing_assessment}", file=sys.stderr)
        if market_analysis_obj.competitors:
            print(f"Key Competitors Identified: {len(market_analysis_obj.competitors)}", file=sys.stderr)
            for comp in market_analysis_obj.competitors[:3]:
                print(f"  - {comp.name}: {comp.funding_raised or 'Funding unknown'}", file=sys.stderr)

        return market_analysis_obj

    except json.JSONDecodeError as json_err:
        print(f"JSONDecodeError: Failed to parse JSON string for MarketAnalysis. Error: {json_err}", file=sys.stderr)
        print(f"String that failed parsing (up to 1000 chars): {json_to_parse[:1000]}", file=sys.stderr)
        if raw_agent_output_container:
            print("Raw agent output at time of error:", raw_agent_output_container, file=sys.stderr)
        return None
    except Exception as e:
        print(f"An error occurred during market intelligence agent execution or response parsing: {e}", file=sys.stderr)
        if raw_agent_output_container:
            print("Raw agent output at time of error:", raw_agent_output_container, file=sys.stderr)
        return None


if __name__ == "__main__":
    print("Market intelligence agent CLI starting...", file=sys.stderr)
    
    parser = argparse.ArgumentParser(description="Market Intelligence Agent CLI - Researches a market/industry sector.")
    parser.add_argument("--sector", type=str, required=True, help="The market or industry sector to research.")
    
    args = parser.parse_args()

    try:
        analysis = run_market_intelligence_cli(args.sector)
        
        if analysis:
            print(analysis.model_dump_json(indent=None))
            print(f"Market intelligence agent: Successfully generated JSON output for sector: {args.sector}", file=sys.stderr)
        else:
            error_output = {
                "status": "error",
                "sector": args.sector,
                "error_message": f"Failed to retrieve or generate market analysis for sector: {args.sector}."
            }
            print(json.dumps(error_output, indent=None))
            print(f"Market intelligence agent: Failed to generate analysis for sector: {args.sector}.", file=sys.stderr)
            sys.exit(1)
            
    except Exception as e_main:
        error_output = {
            "status": "error",
            "sector": args.sector,
            "error_message": f"Critical error in market_intelligence_agent CLI for sector {args.sector}: {str(e_main)}"
        }
        print(json.dumps(error_output, indent=None))
        print(f"Critical error in market_intelligence_agent CLI for sector {args.sector}: {e_main}", file=sys.stderr)
        sys.exit(1)