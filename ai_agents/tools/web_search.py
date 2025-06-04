import os
import json
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
import sys

# Load environment variables from .env file at the project root
load_dotenv()

tavily_search_tool_instance = None
tavily_api_key = os.getenv("TAVILY_API_KEY")

if tavily_api_key:
    try:
        tavily_search_tool_instance = TavilySearchResults(
            api_key=tavily_api_key,
            max_results=5,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=False,
            include_images=False,
        )
    except Exception as e:
        print(f"Error initializing TavilySearchResults: {e}. Tavily search will not be available.", file=sys.stderr)
        tavily_search_tool_instance = None
else:
    print("TAVILY_API_KEY not found in .env. Tavily search_tool will not be available.", file=sys.stderr)

@tool
def search_tool(query: str) -> str:
    """
    Performs a web search using Tavily to find up-to-date information,
    answer specific questions, or get diverse perspectives.
    Provides a concise answer directly if available, along with search results.
    """
    if not tavily_search_tool_instance:
        return "Tavily search_tool is not available (API key missing or initialization failed)."
    print(f"search_tool: Searching Tavily for: {query}", file=sys.stderr)
    try:
        results = tavily_search_tool_instance.invoke(query)
        if isinstance(results, list):
            return json.dumps(results)
        return str(results)
    except Exception as e:
        return f"Error during Tavily search: {str(e)}"

if __name__ == '__main__':
    # Basic test
    if tavily_api_key and tavily_search_tool_instance:
        print("Testing web_search.py (Tavily)...", file=sys.stderr)
        test_query = "What is LangChain?"
        print(f"Query: {test_query}", file=sys.stderr)
        result = search_tool.invoke(test_query)
        print("Result:", file=sys.stderr)
        print(result, file=sys.stderr)
    else:
        print("Cannot run web_search.py test: Tavily API key not found or tool not initialized.", file=sys.stderr) 