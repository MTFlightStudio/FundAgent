from datetime import datetime, timedelta, UTC
from typing import List, Dict

from ai_agents.services.hubspot_client import get_feedback_submissions, HUBSPOT_ACCESS_TOKEN

DEFAULT_LOOKBACK_DAYS = 30


def fetch_recent_survey_responses(
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    survey_name_filter: str | None = None,
) -> List[Dict]:
    """
    Convenience wrapper that fetches recent feedback submissions and returns
    a plain list of dicts ready to embed in other tool output.
    """
    since = datetime.now(UTC) - timedelta(days=lookback_days)
    return get_feedback_submissions(since=since, survey_name_filter=survey_name_filter)

if __name__ == "__main__":
    print("Testing HubSpot Survey Loader...")
    if not HUBSPOT_ACCESS_TOKEN:
        print("HUBSPOT_ACCESS_TOKEN not set in .env. Cannot run HubSpot survey loader test.")
    else:
        print(f"Attempting to fetch survey responses from the last {DEFAULT_LOOKBACK_DAYS} days...")
        # You can customize lookback_days or add a survey_name_filter here for testing
        # Example: responses = fetch_recent_survey_responses(lookback_days=7, survey_name_filter="NPS")
        responses = fetch_recent_survey_responses()

        if responses:
            print(f"\nFetched {len(responses)} survey responses:")
            for i, response in enumerate(responses):
                print(f"\n--- Response {i+1} ---")
                # Print some key details from the response
                # The actual properties depend on what 'get_feedback_submissions' returns
                # and the 'props' defined in 'get_feedback_submissions'
                properties = response.get("properties", {})
                print(f"  ID: {response.get('id', properties.get('hs_object_id', 'N/A'))}")
                print(f"  Created Date: {properties.get('hs_createdate', 'N/A')}")
                print(f"  Survey Name: {properties.get('hs_survey_name', 'N/A')}")
                print(f"  Feedback: {properties.get('hs_feedback_submitted', 'N/A')}")
                print(f"  Sentiment Score: {properties.get('hs_sentiment_score', 'N/A')}")
                # You can print the full response if you want more detail:
                # import json
                # print(json.dumps(response, indent=2))
        elif responses == []: # Explicitly check for an empty list vs None if the function could return None on error
            print("No survey responses found for the given criteria.")
        else:
            # This case might indicate an issue with the request if responses is None
            print("Failed to fetch survey responses or an error occurred. Check console for HubSpot client errors.") 