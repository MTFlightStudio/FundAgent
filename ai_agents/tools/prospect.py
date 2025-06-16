import os
import json
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
import sys

# Load environment variables from .env file at the project root
load_dotenv()

RELEVANCE_AI_API_KEY = os.getenv("RELEVANCE_AI_API_KEY")
RELEVANCE_AI_STUDIO_ID = os.getenv("RELEVANCE_AI_STUDIO_ID")
RELEVANCE_AI_PROJECT_ID = os.getenv("RELEVANCE_AI_PROJECT_ID")

RELEVANCE_AI_ENDPOINT_BASE = "https://api-d7b62b.stack.tryrelevance.com/latest/studios/"

relevance_ai_tool_configured = False
if RELEVANCE_AI_API_KEY and RELEVANCE_AI_STUDIO_ID and RELEVANCE_AI_PROJECT_ID:
    relevance_ai_tool_configured = True
else:
    missing_vars = []
    if not RELEVANCE_AI_API_KEY: missing_vars.append("RELEVANCE_AI_API_KEY")
    if not RELEVANCE_AI_STUDIO_ID: missing_vars.append("RELEVANCE_AI_STUDIO_ID")
    if not RELEVANCE_AI_PROJECT_ID: missing_vars.append("RELEVANCE_AI_PROJECT_ID")
    print(f"Warning: Relevance AI 'Research Prospect' tool is not fully configured. Missing: {', '.join(missing_vars)} in .env file. The tool will not be available.", file=sys.stderr)

@tool
def research_prospect_tool(linkedin_url: str) -> str:
    """
    Researches a prospect based on their LinkedIn URL using a specialized Relevance AI tool.
    Provides a detailed report or summary about the prospect.
    Input must be a valid LinkedIn profile URL (e.g., https://www.linkedin.com/in/username/).
    """
    if not relevance_ai_tool_configured:
        return "Relevance AI 'Research Prospect' tool is not configured. Please check API key and IDs in .env."

    if not linkedin_url or "linkedin.com/in/" not in linkedin_url:
        return "Invalid LinkedIn profile URL provided to research_prospect_tool. It should look like 'https://www.linkedin.com/in/username/'."

    endpoint = f"{RELEVANCE_AI_ENDPOINT_BASE}{RELEVANCE_AI_STUDIO_ID}/trigger_webhook?project={RELEVANCE_AI_PROJECT_ID}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": RELEVANCE_AI_API_KEY
    }
    payload = {
        "linkedin_url": linkedin_url
    }

    print(f"research_prospect_tool: Calling Relevance AI for URL: {linkedin_url}", file=sys.stderr)
    try:
        response = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=180)
        response.raise_for_status()
        response_data = response.json()

        # Try to extract meaningful output, otherwise return the whole response
        if "output" in response_data:
            return json.dumps(response_data["output"], indent=2) if isinstance(response_data["output"], dict) else str(response_data["output"])
        elif "data" in response_data and "output" in response_data["data"]:
             return json.dumps(response_data["data"]["output"], indent=2) if isinstance(response_data["data"]["output"], dict) else str(response_data["data"]["output"])
        else:
            return response.text # Or json.dumps(response_data, indent=2)

    except requests.exceptions.Timeout:
        return f"Error calling Relevance AI: Request timed out (tried 180 seconds) for URL {linkedin_url}. The tool might be long-running."
    except requests.exceptions.HTTPError as http_err:
        return f"Error calling Relevance AI: HTTP error occurred: {http_err}. Response: {response.text}"
    except requests.exceptions.RequestException as req_err:
        return f"Error calling Relevance AI: Request failed: {req_err}"
    except json.JSONDecodeError:
        return f"Error calling Relevance AI: Could not decode JSON response. Response text: {response.text}"
    except Exception as e:
        return f"An unexpected error occurred in research_prospect_tool for URL {linkedin_url}: {str(e)}"

if __name__ == '__main__':
    # Basic test
    if relevance_ai_tool_configured:
        print("Testing prospect.py (Relevance AI)...", file=sys.stderr)
        test_url = "https://www.linkedin.com/in/satyanadella/"
        print(f"Query URL: {test_url}", file=sys.stderr)
        result = research_prospect_tool.invoke({"linkedin_url": test_url})
        print("Result:", file=sys.stderr)
        print(result, file=sys.stderr)

        test_invalid_url = "https://example.com"
        print(f"\nQuery invalid URL: {test_invalid_url}", file=sys.stderr)
        result_invalid = research_prospect_tool.invoke({"linkedin_url": test_invalid_url})
        print("Result (invalid):", file=sys.stderr)
        print(result_invalid, file=sys.stderr)
    else:
        print("Cannot run prospect.py test: Relevance AI tool not configured.", file=sys.stderr) 