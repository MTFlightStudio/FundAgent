import requests
import json
from langchain.tools import tool

@tool
def research_prospect_tool(linkedin_url: str) -> str:
    """
    Researches a prospect based on their LinkedIn URL using a specialized Relevance AI tool.
    Provides a detailed report or summary about the prospect.
    Input must be a valid LinkedIn profile URL (e.g., https://www.linkedin.com/in/username/).
    """
    endpoint = "https://api.relevance.ai/v1/research/prospect"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('RELEVANCE_AI_API_KEY')}"
    }
    payload = {
        "linkedin_url": linkedin_url
    }

    try:
        response = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=90) # Increased timeout to 90s
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)

        response_data = response.json()

        # Prioritize the "answer" key if it exists, as observed in successful logs
        if "answer" in response_data and isinstance(response_data["answer"], str):
            return response_data["answer"]
        # Fallback to existing logic for "output" or "data.output"
        elif "output" in response_data:
            if isinstance(response_data["output"], str):
                return response_data["output"]
            else:
                return json.dumps(response_data["output"], indent=2)
        elif "data" in response_data and "output" in response_data["data"]:
            if isinstance(response_data["data"]["output"], str):
                return response_data["data"]["output"]
            else:
                return json.dumps(response_data["data"]["output"], indent=2)
        else:
            # If no specific "answer" or "output" key, return the whole response as JSON string
            # This was the previous behavior that worked, but returning response.text is safer if response_data is not always a dict
            return response.text # Or json.dumps(response_data, indent=2)

    except requests.exceptions.Timeout:
        # Handle timeout
        return "The request timed out. Please try again later."

    except Exception as e:
        # Handle other exceptions
        return f"An error occurred: {str(e)}" 