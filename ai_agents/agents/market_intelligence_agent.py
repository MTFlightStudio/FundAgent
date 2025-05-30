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
            "You are a specialized market intelligence AI analyst. Your goal is to gather comprehensive information about a given market or industry and structure it into a JSON object adhering to the MarketAnalysis schema provided below.\n"
            "You MUST use the 'search_tool' and 'wiki_tool' to find the required information. Make multiple targeted search queries to cover all aspects:\n\n"
            "Required Research Areas:\n"
            "1. Market Size and Growth:\n"
            "   - Search for 'TAM SAM SOM [market name]', '[market name] market size billion'\n"
            "   - Find Total Addressable Market (TAM), Serviceable Addressable Market (SAM), and Serviceable Obtainable Market (SOM)\n"
            "   - Look for CAGR (Compound Annual Growth Rate) and growth projections\n"
            "   - Use terms like 'market forecast', 'industry analysis', 'market research report'\n"
            "2. Key Market Trends:\n"
            "   - Search for '[market name] trends 2024-2025', 'future of [market]'\n"
            "   - Look for technological shifts, consumer behavior changes, emerging segments\n"
            "3. Competitive Landscape:\n"
            "   - Search for 'top companies [market name]', '[market name] competitive landscape'\n"
            "   - Find 3-5 key competitors with their funding amounts, strengths, and positioning\n"
            "   - Look for market share data, competitive advantages, and differentiation\n"
            "4. Market Timing and Maturity:\n"
            "   - Assess if the market is in early/growth/mature/declining phase\n"
            "   - Look for indicators like number of new entrants, M&A activity, funding trends\n"
            "5. Barriers to Entry:\n"
            "   - Search for 'barriers to entry [market name]', 'challenges [market name] startups'\n"
            "   - Consider regulatory, capital, technical, and network effect barriers\n"
            "6. Regulatory Environment:\n"
            "   - Search for '[market name] regulations', 'compliance requirements [market]'\n"
            "   - Look for government policies, licensing requirements, data protection laws\n\n"
            "Search Strategy Tips:\n"
            "- Start with Wikipedia for industry overview and established facts\n"
            "- Use specific search queries for recent market data (include years like 2024, 2025)\n"
            "- Look for credible sources: research firms (Gartner, McKinsey, CB Insights), industry reports, news outlets\n"
            "- Cross-reference multiple sources for market size estimates\n"
            "- Be specific with competitor searches to get funding and positioning data\n\n"
            "After gathering comprehensive information, your FINAL output MUST be a single, valid JSON object that strictly conforms to the MarketAnalysis schema below. Do NOT include any other text, explanations, or markdown outside of this JSON object.\n"
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
        print("Using Anthropic Claude model for market intelligence.")
        return ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.2)
    elif os.getenv("OPENAI_API_KEY"):
        print("Using OpenAI GPT model for market intelligence.")
        return ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.2)
    else:
        print("FATAL: Neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set. Cannot initialize LLM.")
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
    print(f"Starting comprehensive market intelligence for: \"{market_or_industry}\"")

    # --- API Key and Tool Availability Checks ---
    if not tavily_search_tool_instance:
        print("FATAL: Tavily search_tool is not available (check TAVILY_API_KEY in .env). Cannot proceed with market research.")
        return None

    if not wiki_tool_instance:
        print("Warning: Wikipedia tool is not available. Market research will rely solely on web search.")

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
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=15  # Allow more iterations for comprehensive market research
    )

    raw_agent_output_container = None
    try:
        raw_agent_output_container = agent_executor.invoke({"input": market_or_industry})
        
        llm_response_str = None
        if isinstance(raw_agent_output_container, dict) and "output" in raw_agent_output_container:
            output_content = raw_agent_output_container["output"]
            if isinstance(output_content, str):
                llm_response_str = output_content
            else:
                print(f"Warning: Unexpected structure in agent output['output']: {output_content}")
                llm_response_str = str(output_content)
        else:
            print(f"Warning: Unexpected output structure from agent: {raw_agent_output_container}")
            llm_response_str = str(raw_agent_output_container)

        if not llm_response_str:
            print("Agent did not produce a parsable output string for MarketAnalysis.")
            return None

        print(f"\nRaw LLM response string (before JSON extraction for MarketAnalysis): {llm_response_str[:500]}...")

        # Extract JSON block using regex
        json_match = re.search(r"\{.*\}", llm_response_str, re.DOTALL)
        if json_match:
            json_to_parse = json_match.group(0).strip()
            print(f"Extracted JSON block for MarketAnalysis parsing (stripped): {json_to_parse[:300]}...")
        else:
            json_to_parse = llm_response_str.strip()
            print(f"Warning: No clear JSON block found, attempting to parse whole string for MarketAnalysis (stripped): {json_to_parse[:300]}...")

        # Parse the JSON string into MarketAnalysis object
        parsed_dict = json.loads(json_to_parse, strict=False)
        market_analysis_obj = MarketAnalysis.model_validate(parsed_dict)

        print("\n--- Market Intelligence Output ---")
        print(market_analysis_obj.model_dump_json(indent=2))

        # Save the structured data
        safe_market_name = "".join(c if c.isalnum() else "_" for c in market_or_industry)[:50].rstrip("_")
        output_filename = f"market_analysis_{safe_market_name}.json"
        
        try:
            save_tool.invoke({
                "filename": output_filename,
                "text": market_analysis_obj.model_dump_json(indent=2)
            })
            print(f"\nMarket analysis saved to {output_filename}")
        except Exception as e_save:
            print(f"Error saving market analysis to file: {e_save}")

        # Print summary of key findings
        print("\n--- Market Intelligence Summary ---")
        if market_analysis_obj.industry_overview:
            print(f"Industry Overview: {market_analysis_obj.industry_overview[:200]}...")
        if market_analysis_obj.market_size_tam:
            print(f"Total Addressable Market (TAM): {market_analysis_obj.market_size_tam}")
        if market_analysis_obj.market_growth_rate_cagr:
            print(f"Market Growth Rate (CAGR): {market_analysis_obj.market_growth_rate_cagr}")
        if market_analysis_obj.market_timing_assessment:
            print(f"Market Timing: {market_analysis_obj.market_timing_assessment}")
        if market_analysis_obj.competitors:
            print(f"Key Competitors Identified: {len(market_analysis_obj.competitors)}")
            for comp in market_analysis_obj.competitors[:3]:  # Show first 3
                print(f"  - {comp.name}: {comp.funding_raised or 'Funding unknown'}")

        return market_analysis_obj

    except json.JSONDecodeError as json_err:
        print(f"JSONDecodeError: Failed to parse JSON string for MarketAnalysis. Error: {json_err}")
        print(f"String that failed parsing (up to 1000 chars): {json_to_parse[:1000] if 'json_to_parse' in locals() else 'N/A'}")
        if raw_agent_output_container:
            print("Raw agent output at time of error:", raw_agent_output_container)
        return None
    except Exception as e:
        print(f"An error occurred during market intelligence agent execution or response parsing: {e}")
        if raw_agent_output_container:
            print("Raw agent output at time of error:", raw_agent_output_container)
        return None


if __name__ == "__main__":
    print("Testing market_intelligence_agent.py...")
    
    # Example test markets
    test_markets = [
        "AI-powered fintech",
        "Creator economy tools",
        "Sustainable food tech",
        "Mental health apps",
        "Electric vehicle charging infrastructure"
    ]
    
    print("\nExample markets to research:")
    for i, market in enumerate(test_markets, 1):
        print(f"{i}. {market}")
    
    market_to_research = input("\nEnter a market/industry name for research (or press Enter for default): ").strip()
    
    if not market_to_research:
        market_to_research = "AI-powered fintech"  # Default
        print(f"Using default market: {market_to_research}")

    if market_to_research:
        analysis = run_market_intelligence_cli(market_to_research)
        if analysis:
            print(f"\nSuccessfully completed market intelligence for: {market_to_research}")
            if analysis.market_timing_assessment:
                print(f"Market Timing Assessment: {analysis.market_timing_assessment}")
        else:
            print(f"Failed to complete market intelligence for: {market_to_research}")
    else:
        print("No market name provided for test.")