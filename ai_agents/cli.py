import argparse
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
# Updated import path for tools
from ai_agents.tools import search_tool, wiki_tool, save_tool, research_prospect_tool

# load_dotenv() will search in the current directory (ai_agents/) and then parent directories.
# If .env is in ai_agents/, this will find it.
load_dotenv()

# --- Check for Essential API Keys ---
OPENAI_API_KEY_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))
TAVILY_API_KEY_AVAILABLE = bool(os.getenv("TAVILY_API_KEY"))
RELEVANCE_AI_CONFIGURED = all([
    os.getenv("RELEVANCE_AI_API_KEY"),
    os.getenv("RELEVANCE_AI_STUDIO_ID"),
    os.getenv("RELEVANCE_AI_PROJECT_ID")
])

if not OPENAI_API_KEY_AVAILABLE:
    print("Warning: OPENAI_API_KEY not found in .env. The agent may not function correctly.")
if not TAVILY_API_KEY_AVAILABLE:
    print("Warning: TAVILY_API_KEY not found in .env. The search_tool will be limited or non-functional.")
    print("Please get a key from https://tavily.com and add it to your .env file.")
if not RELEVANCE_AI_CONFIGURED:
    print("Warning: Relevance AI 'Research Prospect' tool is not fully configured (missing API key, Studio ID, or Project ID in .env).")
    print("The research_prospect_tool will not be available.")

class ResearchResponse(BaseModel):
    topic: str = Field(description="The main topic of the research query.")
    summary: str = Field(description="A comprehensive summary of the research findings.")
    sources: list[str] = Field(description="A list of URLs or identifiers for the sources used.")


# Determine which LLM to use based on available API keys
# You can extend this logic if you prefer one over the other when both are available
llm = None
if os.getenv("ANTHROPIC_API_KEY"):
    llm = ChatAnthropic(model="claude-3-haiku-20240307") # Or your preferred Claude model
    print("Using Anthropic Claude model.")
elif OPENAI_API_KEY_AVAILABLE:
    llm = ChatOpenAI(model="gpt-3.5-turbo") # Or your preferred OpenAI model
    print("Using OpenAI GPT model.")
else:
    print("FATAL: Neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set. Cannot initialize LLM.")
    exit()


parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a powerful research assistant. Your goal is to answer the user's query "
            "comprehensively using the available tools (web search, Wikipedia, save text, and a specialized 'Research Prospect' tool for LinkedIn URLs). "
            "When using the web search_tool, it leverages Tavily Search for optimized results. "
            "If a LinkedIn profile URL is provided or relevant to the query, use the 'research_prospect_tool' to get detailed information about the prospect. "
            "Synthesize all information gathered into a coherent summary. "
            "Always cite your sources. "
            "Respond in the following Pydantic format:\n{format_instructions}"
        ),
        ("placeholder", "{chat_history}"), # Placeholder for chat history if you add memory
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"), # Placeholder for agent's intermediate steps
    ]
).partial(format_instructions=parser.get_format_instructions())


# Define the list of tools available to the agent
# Ensure all tools are correctly imported and configured
available_tools = []
if TAVILY_API_KEY_AVAILABLE:
    available_tools.append(search_tool)
else:
    print("search_tool (Tavily) is not available due to missing API key.")

available_tools.append(wiki_tool) # Wikipedia tool doesn't require an API key through langchain
available_tools.append(save_tool)

if RELEVANCE_AI_CONFIGURED:
    available_tools.append(research_prospect_tool)
else:
    print("research_prospect_tool (Relevance AI) is not available due to missing configuration.")


# Create the agent (this function takes llm, prompt, and tools)
# Ensure llm is initialized before this point
if llm:
    agent = create_tool_calling_agent(
        llm=llm,
        prompt=prompt,
        tools=available_tools
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=available_tools,
        verbose=True, # Set to True to see agent's thought process and tool calls
        handle_parsing_errors=True # Gracefully handle if LLM output doesn't match Pydantic
    )
    print("Research Agent Initialized.")
else:
    print("Agent could not be initialized because LLM is not available.")
    agent_executor = None


def run_agent():
    if not agent_executor:
        print("Agent executor not available. Exiting.")
        return

    query = input("What can I help you research? ")
    if not query:
        print("No query provided. Exiting.")
        return

    try:
        # Assuming chat_history is not yet implemented, pass an empty list or appropriate placeholder
        raw_response = agent_executor.invoke({"query": query, "chat_history": []})

        # The output structure from agent_executor.invoke might vary.
        # It's often in raw_response['output'] for tool-calling agents.
        # Inspect raw_response to confirm the path to the LLM's final text response.
        output_text = raw_response.get("output")
        if not output_text:
            print("Error: No output from agent.")
            print("Raw response:", raw_response)
            return

        # If the output is already a Pydantic model (some agent types might do this)
        # or if it's a string that needs parsing:
        if isinstance(output_text, str):
            structured_response = parser.parse(output_text)
        elif isinstance(output_text, dict) and hasattr(ResearchResponse, "model_validate"): # For Pydantic v2
             structured_response = ResearchResponse.model_validate(output_text)
        elif isinstance(output_text, ResearchResponse):
            structured_response = output_text
        else:
            print(f"Unexpected output type from agent: {type(output_text)}")
            print("Raw output:", output_text)
            # Attempt to parse if it looks like a string representation of the model
            try:
                structured_response = parser.parse(str(output_text))
            except Exception as parse_error:
                print(f"Could not parse agent output: {parse_error}")
                return


        print("\n--- Research Report ---")
        print(f"Topic: {structured_response.topic}")
        print(f"Summary: {structured_response.summary}")
        print("Sources:")
        for source in structured_response.sources:
            print(f"- {source}")

        output_filename = f"research_on_{structured_response.topic.replace(' ', '_')}.txt"
        # Use the save_tool correctly by invoking it
        save_tool.invoke({"filename": output_filename, "text": structured_response.model_dump_json(indent=2)})
        print(f"Full structured response saved to {output_filename}")

    except Exception as e:
        print(f"An error occurred during agent execution or response parsing: {e}")
        if 'raw_response' in locals():
            print("Raw Response at time of error:", raw_response)

# Import agent functions AFTER load_dotenv
# Assuming these files exist and are structured correctly:
# from ai_agents.agents.email_triage import process_inbox # Placeholder
from ai_agents.agents.research_agent import run_research_cli

# Placeholder for process_inbox if not yet implemented
def process_inbox(run_once=False):
    print(f"Email triage process_inbox called. Run once: {run_once}")
    print("This is a placeholder. Implement email triage logic in ai_agents/agents/email_triage.py")
    # Example:
    # if run_once:
    #     print("Processing inbox once...")
    # else:
    #     print("Starting continuous inbox processing (not implemented)...")


def main():
    ap = argparse.ArgumentParser(description="AI Agents CLI")
    subparsers = ap.add_subparsers(dest="command", help="Available commands")
    subparsers.required = True # Make a command mandatory

    # Triage command
    triage_parser = subparsers.add_parser("triage", help="Process email inbox for triage.")
    triage_parser.add_argument("--once", action="store_true", help="Run the triage process once and exit.")
    triage_parser.set_defaults(func=lambda args: process_inbox(run_once=args.once))

    # Research command
    research_parser = subparsers.add_parser("research", help="Run the research agent with a query.")
    research_parser.add_argument("query", nargs="+", help="The research query (can be multiple words).")
    research_parser.set_defaults(func=lambda args: run_research_cli(" ".join(args.query)))

    args = ap.parse_args()
    args.func(args) # Call the function associated with the chosen subparser

if __name__ == "__main__":
    main() 