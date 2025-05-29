import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from typing import List, Optional
import re # Import the 're' module for regular expressions

# Import tools from the refactored tools package
from ai_agents.tools import (
    search_tool, wiki_tool, save_tool, research_prospect_tool,
    tavily_search_tool_instance, relevance_ai_tool_configured # For config checks
)

# Load .env file from the project root
load_dotenv()

# --- Pydantic Model for Structured Output ---
class ResearchResponse(BaseModel):
    summary: str = Field(description="A comprehensive summary of the research findings, directly answering the user's query.")
    sources: List[str] = Field(description="A list of URLs for the primary sources used to construct the summary.")
    search_quality_reflection: Optional[str] = Field(None, description="Your brief reflection on the quality and relevance of the search results obtained to answer the query.")
    search_quality_score: Optional[int] = Field(None, description="A score from 1 (poor) to 5 (excellent) indicating how well the search results helped answer the query.")
    potential_biases: Optional[str] = Field(None, description="Identify any potential biases observed in the information or its sources relevant to the query.")
    confidence_score: Optional[float] = Field(None, description="A score from 0.0 (low) to 1.0 (high) indicating your confidence in the accuracy and completeness of the research findings presented in the summary.")
    further_research_suggestions: Optional[List[str]] = Field(None, description="Suggest 1-2 specific, actionable further research questions or areas to explore based on the findings and any gaps identified.")

# --- Output Parser ---
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# --- Prompt Template ---
# Updated system prompt for stricter JSON output
RESEARCH_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a meticulous research assistant. Your goal is to answer the user's query based on the provided context (tool results).\n"
            "After reviewing the information, you MUST generate a single, valid JSON object as your final answer.\n"
            "This JSON object must strictly adhere to the following schema and contain nothing else:\n"
            "{format_instructions}\n"
            "Do NOT include any XML-like tags, markdown, or any other text outside of this single JSON object."
            "Ensure all string fields in the JSON are properly escaped."
        ),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
).partial(format_instructions=parser.get_format_instructions())

def run_research_cli(query: str):
    """
    Runs the research agent with the given query and prints the structured output.
    """
    # --- Check for Essential API Keys ---
    OPENAI_API_KEY_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))
    # Tavily key check is implicitly handled by tavily_search_tool_instance
    # Relevance AI check is implicitly handled by relevance_ai_tool_configured

    if not OPENAI_API_KEY_AVAILABLE and not os.getenv("ANTHROPIC_API_KEY"):
        print("FATAL: Neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set in .env. Cannot initialize LLM for research.")
        return

    if not tavily_search_tool_instance: # Check if Tavily tool was successfully initialized
        print("Warning: Tavily search_tool is not available (check TAVILY_API_KEY in .env). Web search capabilities will be limited.")

    if not relevance_ai_tool_configured:
        print("Warning: Relevance AI 'Research Prospect' tool is not configured (check RELEVANCE_AI_* keys in .env). Prospect research will not be available.")

    llm = None
    if os.getenv("ANTHROPIC_API_KEY"):
        llm = ChatAnthropic(model="claude-3-haiku-20240307")
        print("Using Anthropic Claude model for research.")
    elif OPENAI_API_KEY_AVAILABLE:
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        print("Using OpenAI GPT model for research.")
    else:
        # This case should have been caught above, but as a safeguard:
        print("LLM could not be initialized for research.")
        return

    print("Research Agent Initialized.")

    # Include all available tools that make sense for general research
    # Ensure tools are correctly imported and initialized
    available_tools = []
    if tavily_search_tool_instance:
        available_tools.append(search_tool)
    if wiki_tool: # Assuming wiki_tool_instance is the check for Wikipedia
        available_tools.append(wiki_tool)
    # Add research_prospect_tool if the query looks like a LinkedIn URL, otherwise it might be noisy
    if "linkedin.com/in/" in query and relevance_ai_tool_configured:
        available_tools.append(research_prospect_tool)
    # save_tool is more for explicit agent actions, maybe not for default research synthesis

    if not available_tools:
        print("No research tools (like Tavily or Wikipedia) are configured. Research agent may not function well.")
        # Optionally, you could prevent the agent from running or use a fallback.

    research_agent = create_tool_calling_agent(llm=llm, prompt=RESEARCH_PROMPT_TEMPLATE, tools=available_tools)
    agent_executor = AgentExecutor(agent=research_agent, tools=available_tools, verbose=True)

    try:
        print(f"\nStarting research for: \"{query}\"")
        raw_agent_output_container = agent_executor.invoke({"input": query})

        llm_response_str = None
        # Refined logic to extract the actual text string from the agent's output
        if isinstance(raw_agent_output_container, dict) and "output" in raw_agent_output_container:
            output_content = raw_agent_output_container["output"]
            if isinstance(output_content, str):
                llm_response_str = output_content
            elif isinstance(output_content, list) and len(output_content) > 0 and \
                 isinstance(output_content[0], dict) and "text" in output_content[0]:
                llm_response_str = output_content[0]["text"]
                print(f"Extracted text from dict->list->dict structure: {llm_response_str[:200]}...")
            else:
                print(f"Warning: Unexpected structure within 'output' key: {output_content}")
                llm_response_str = str(output_content) # Fallback, likely to fail parsing
        elif isinstance(raw_agent_output_container, list) and len(raw_agent_output_container) > 0 and \
             isinstance(raw_agent_output_container[0], dict) and "text" in raw_agent_output_container[0]:
            llm_response_str = raw_agent_output_container[0]["text"]
            print(f"Extracted text from direct list->dict structure: {llm_response_str[:200]}...")
        else:
            print(f"Warning: Unexpected output structure from agent: {raw_agent_output_container}")
            llm_response_str = str(raw_agent_output_container) # Fallback

        if not llm_response_str:
            print("Agent did not produce a parsable output string.")
            return

        print(f"\nRaw LLM response string (before JSON extraction): {llm_response_str[:500]}...")

        # Attempt to extract the JSON block from the potentially wrapped LLM output
        # This regex looks for the outermost curly braces and everything in between.
        json_match = re.search(r"\{.*\}", llm_response_str, re.DOTALL)
        if json_match:
            json_to_parse = json_match.group(0)
            print(f"Extracted JSON block for parsing: {json_to_parse[:300]}...")
        else:
            # If no clear JSON block is found, try to parse the whole string.
            # This might fail if there's extra text, but it's a fallback.
            json_to_parse = llm_response_str
            print(f"Warning: No clear JSON block found with regex, attempting to parse whole string: {json_to_parse[:300]}...")

        structured_response = ResearchResponse.model_validate_json(json_to_parse, strict=False)

        print("\n--- Research Agent Output ---")
        print(f"Query: {query}") # Use the original query as the topic
        print(f"Summary: {structured_response.summary}")
        print("Sources:")
        for source in structured_response.sources:
            print(f"- {source}")
        
        if structured_response.search_quality_reflection:
            print(f"Search Quality Reflection: {structured_response.search_quality_reflection}")
        if structured_response.search_quality_score is not None:
            print(f"Search Quality Score: {structured_response.search_quality_score}/5")
        if structured_response.potential_biases:
            print(f"Potential Biases: {structured_response.potential_biases}")
        if structured_response.confidence_score is not None:
            print(f"Confidence Score: {structured_response.confidence_score*100:.0f}%")
        if structured_response.further_research_suggestions:
            print("Further Research Suggestions:")
            for suggestion in structured_response.further_research_suggestions:
                print(f"- {suggestion}")

        # Create a filename-safe version of the query
        safe_query_filename = "".join(c if c.isalnum() else "_" for c in query)[:50].rstrip("_")
        output_filename = f"research_on_{safe_query_filename}.json" # Save as .json
        
        # Save the structured data as JSON
        save_tool.invoke({"filename": output_filename, "text": structured_response.model_dump_json(indent=2)})
        print(f"Full structured response saved to {output_filename}")

    except Exception as e:
        print(f"An error occurred during research agent execution or response parsing: {e}")
        if 'raw_agent_output_container' in locals():
            print("Raw Response at time of error:", raw_agent_output_container)

if __name__ == "__main__":
    # Example of how to run this directly for testing
    print("Testing research_agent.py...")
    test_query = input("Enter a research query for testing: ")
    if test_query:
        run_research_cli(test_query)
    else:
        print("No query provided for test.") 