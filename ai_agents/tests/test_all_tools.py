import os
import sys
import json # Added for prospect tool test
from dotenv import load_dotenv

# Load .env from the project root (searches current dir and parents)
load_dotenv()

# Add the project root to sys.path to allow imports like 'from ai_agents.tools import ...'
# This is often needed if tests are run as individual scripts from their directory
# or if the package structure isn't automatically recognized by the test runner.
# The 'AI-AGENTS' directory should be the project_root.
project_root_name = "AI-AGENTS" # Adjust if your root folder has a different name
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up until project_root_name is found or filesystem root is hit
project_root_path = current_dir
while os.path.basename(project_root_path) != project_root_name and project_root_path != os.path.dirname(project_root_path):
    project_root_path = os.path.dirname(project_root_path)

if os.path.basename(project_root_path) == project_root_name and project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)
elif project_root_path == os.path.dirname(project_root_path) and current_dir not in sys.path:
    # Fallback if project_root_name not found, add current script's dir parent (ai_agents)
    # This might be needed if tests are run with CWD inside ai_agents/tests
    sys.path.insert(0, os.path.dirname(current_dir))


from ai_agents.tools import (
    search_tool,
    wiki_tool,
    save_tool,
    research_prospect_tool,
    tavily_search_tool_instance,
    wiki_tool_instance,
    relevance_ai_tool_configured
)
# Note: Pydantic, LangChain components are not directly tested here, but by the tools using them.

# --- Test Functions ---

def test_tavily_search():
    print("\n--- Testing Tavily Search (Direct Tool Call from test_all_tools.py) ---")
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key or not tavily_search_tool_instance:
        print("Tavily Search tool not configured or API key missing. Skipping test.")
        result = search_tool.invoke("test query if not configured")
        print(f"Result from search_tool (not configured): {result}")
        assert "not available" in result or "API key missing" in result, "Error message for unconfigured Tavily not as expected."
        return

    query = "Latest AI advancements"
    print(f"Test Query: {query}")
    try:
        results = search_tool.invoke(query)
        print("Search Results:")
        print(results)
        assert results is not None, "Tavily search returned None"
        assert "Error" not in results, f"Tavily search returned an error string: {results}"
    except Exception as e:
        print(f"An error occurred during Tavily search test: {e}")
        assert False, f"Exception during Tavily search test: {e}"

def test_wikipedia_search():
    print("\n--- Testing Wikipedia Search (Direct Tool Call from test_all_tools.py) ---")
    if not wiki_tool_instance:
        print("Wikipedia tool not available. Skipping test.")
        result = wiki_tool.invoke("test query if not configured")
        print(f"Result from wiki_tool (not configured): {result}")
        assert "not available" in result, "Error message for unconfigured Wikipedia not as expected."
        return

    query = "Artificial General Intelligence"
    print(f"Test Query: {query}")
    try:
        results = wiki_tool.invoke(query)
        print("Search Results:")
        print(results)
        assert results is not None, "Wikipedia search returned None"
        assert "Error" not in results, f"Wikipedia search returned an error string: {results}"
        assert "Artificial General Intelligence" in results or "AGI" in results, "Query term not found in Wikipedia results"
    except Exception as e:
        print(f"An error occurred during Wikipedia search test: {e}")
        assert False, f"Exception during Wikipedia search test: {e}"

def test_save_functionality():
    print("\n--- Testing Save Functionality (from test_all_tools.py) ---")
    # Save in the current directory of the test script (ai_agents/tests/)
    filename = "test_output_from_test_script.txt"
    full_path = os.path.join(os.path.dirname(__file__), filename)
    content = "This is a test string to be saved to a file during the test from test_all_tools.py."
    print(f"Attempting to save to {full_path}...")
    try:
        result_message = save_tool.invoke({"filename": full_path, "text": content})
        print(result_message)
        assert "successfully" in result_message, "Save tool did not report success."
        assert os.path.exists(full_path), f"File '{full_path}' was not created."
        with open(full_path, "r", encoding="utf-8") as f:
            saved_content = f.read()
        assert saved_content == content, "Content verification FAILED. Mismatch between saved and original content."
        print("Content verification successful.")
        os.remove(full_path) # Clean up the test file
        print(f"Test file '{full_path}' removed.")
    except Exception as e:
        print(f"An error occurred during save functionality test: {e}")
        assert False, f"Exception during save functionality test: {e}"
    finally:
        if os.path.exists(full_path): # Ensure cleanup even if assert fails mid-try
            os.remove(full_path)

def test_relevance_ai_prospect_tool():
    print("\n--- Testing Relevance AI 'Research Prospect' Tool (Direct Tool Call from test_all_tools.py) ---")
    if not relevance_ai_tool_configured:
        print("Relevance AI 'Research Prospect' tool is not configured (check .env). Skipping test.")
        result = research_prospect_tool.invoke({"linkedin_url": "https://www.linkedin.com/in/someprofile"})
        print(f"Result from research_prospect_tool (not configured): {result}")
        assert "not configured" in result, "Error message for unconfigured Relevance AI not as expected."
        return

    test_url_valid = "https://www.linkedin.com/in/reidhoffman/"
    print(f"Test LinkedIn URL for Relevance AI (valid): {test_url_valid}")
    try:
        results = research_prospect_tool.invoke({"linkedin_url": test_url_valid})
        print("Relevance AI 'Research Prospect' Tool Results (valid):")
        print(results)
        assert results is not None, "Relevance AI tool returned None for valid URL."
        assert "Error" not in results, f"Relevance AI tool returned an error for valid URL: {results}"
        # Basic check for JSON structure or key content
        try:
            data = json.loads(results) # Assuming tool returns JSON string
            assert "answer" in data or "name" in data or "overview" in data, "Expected keys not found in Relevance AI JSON response."
        except json.JSONDecodeError:
            # If it's not JSON, it might be a direct string output from the tool's logic
            assert len(results) > 50, "Relevance AI output string seems too short for a valid profile."


    except Exception as e:
        print(f"An error occurred during Relevance AI 'Research Prospect' tool test (valid URL): {e}")
        assert False, f"Exception during Relevance AI test (valid URL): {e}"

    test_url_invalid_format = "https://example.com/notlinkedin"
    print(f"\nTest LinkedIn URL for Relevance AI (invalid format): {test_url_invalid_format}")
    results_invalid_format = research_prospect_tool.invoke({"linkedin_url": test_url_invalid_format})
    print("Relevance AI 'Research Prospect' Tool Results (invalid format):")
    print(results_invalid_format)
    assert "Invalid LinkedIn profile URL" in results_invalid_format, "Expected error message for invalid format not found."

    test_url_non_existent = "https://www.linkedin.com/in/thisisnotaveryrealprofile123xyzabc/"
    print(f"\nTest LinkedIn URL for Relevance AI (non-existent): {test_url_non_existent}")
    results_non_existent = research_prospect_tool.invoke({"linkedin_url": test_url_non_existent})
    print("Relevance AI 'Research Prospect' Tool Results (non-existent):")
    print(results_non_existent)
    # Expecting an empty JSON {} or a message indicating not found, depending on Relevance AI tool's behavior
    assert results_non_existent == "{}" or "not found" in results_non_existent.lower() or "Error" in results_non_existent, \
           "Unexpected response for non-existent profile."


if __name__ == "__main__":
    print("Running tool tests from ai_agents/tests/test_all_tools.py...")
    test_tavily_search()
    test_wikipedia_search()
    test_save_functionality()
    test_relevance_ai_prospect_tool()
    print("\nAll tests from test_all_tools.py completed.") 