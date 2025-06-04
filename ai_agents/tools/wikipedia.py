from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
import sys

# Load environment variables (though Wikipedia tool doesn't strictly need them from .env)
load_dotenv()

wiki_tool_instance = None
try:
    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=2000)
    wiki_tool_instance = WikipediaQueryRun(api_wrapper=api_wrapper)
except ImportError:
    print("Wikipedia package not installed. `pip install wikipedia`", file=sys.stderr)
    wiki_tool_instance = None
except Exception as e:
    print(f"Error initializing Wikipedia tool: {e}", file=sys.stderr)
    wiki_tool_instance = None

@tool
def wiki_tool(query: str) -> str:
    """
    Looks up information on Wikipedia. Use this for general knowledge,
    definitions, historical events, and established facts.
    """
    if not wiki_tool_instance:
        return "Wikipedia tool is not available."
    print(f"wiki_tool: Searching Wikipedia for: {query}", file=sys.stderr)
    try:
        return wiki_tool_instance.run(query)
    except Exception as e:
        return f"Error during Wikipedia search: {str(e)}"

if __name__ == '__main__':
    # Basic test
    if wiki_tool_instance:
        print("Testing wikipedia.py...", file=sys.stderr)
        test_query = "Python (programming language)"
        print(f"Query: {test_query}", file=sys.stderr)
        result = wiki_tool.invoke(test_query)
        print("Result:", file=sys.stderr)
        print(result, file=sys.stderr)
    else:
        print("Cannot run wikipedia.py test: tool not initialized.", file=sys.stderr) 