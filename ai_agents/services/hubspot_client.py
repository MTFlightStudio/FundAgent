import os
import requests
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

HUBSPOT_API_KEY = os.getenv("HUBSPOT_API_KEY") # Or HUBSPOT_TOKEN based on your .env
HUBSPOT_BASE_URL = "https://api.hubapi.com"

# Define common HubSpot API errors that might be retriable
RETRIABLE_STATUS_CODES = [429, 500, 502, 503, 504]
MAX_RETRIES = 3
INITIAL_WAIT_SECONDS = 1
MAX_WAIT_SECONDS = 10

class HubSpotAPIError(Exception):
    """Custom exception for HubSpot API errors."""
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        super().__init__(f"HubSpot API Error {status_code}: {message}")

def is_retriable_error(exception):
    """Check if the exception is a HubSpotAPIError with a retriable status code."""
    return isinstance(exception, HubSpotAPIError) and exception.status_code in RETRIABLE_STATUS_CODES

# Decorator for retrying HubSpot API calls
hubspot_retry_decorator = retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=INITIAL_WAIT_SECONDS, max=MAX_WAIT_SECONDS),
    retry=retry_if_exception_type((requests.exceptions.ConnectionError, requests.exceptions.Timeout, HubSpotAPIError)) # Retry on connection/timeout and custom retriable API errors
    # For HubSpotAPIError, you might want a more specific retry condition using retry_if_exception(is_retriable_error)
    # but tenacity needs the exception type directly for retry_if_exception_type.
    # A more complex setup might involve a custom retry condition function.
)

def _make_request(method, endpoint, params=None, json_data=None, headers=None):
    if not HUBSPOT_API_KEY:
        print("Error: HUBSPOT_API_KEY not set in .env")
        raise ValueError("HubSpot API key not configured.")

    url = f"{HUBSPOT_BASE_URL}{endpoint}"
    default_headers = {
        "Authorization": f"Bearer {HUBSPOT_API_KEY}",
        "Content-Type": "application/json"
    }
    if headers:
        default_headers.update(headers)

    try:
        response = requests.request(method, url, params=params, json=json_data, headers=default_headers, timeout=10)
        if not response.ok:
            # Raise a custom error that can be caught by tenacity for retries if status code is retriable
            if response.status_code in RETRIABLE_STATUS_CODES:
                raise HubSpotAPIError(status_code=response.status_code, message=response.text)
            # For non-retriable errors, raise a different exception or handle directly
            response.raise_for_status() # This will raise HTTPError for other bad responses (4xx, 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"HubSpot request failed for {method} {url}: {e}")
        raise # Re-raise to be caught by tenacity or handled by caller

@hubspot_retry_decorator
def get_contact_by_email(email: str) -> Optional[dict]:
    """Fetches a contact by email from HubSpot."""
    print(f"HubSpot: Fetching contact by email: {email}")
    endpoint = "/crm/v3/objects/contacts/search"
    payload = {
        "filterGroups": [
            {
                "filters": [
                    {
                        "propertyName": "email",
                        "operator": "EQ",
                        "value": email
                    }
                ]
            }
        ],
        "properties": ["email", "firstname", "lastname", "hs_object_id"], # Add properties you need
        "limit": 1
    }
    try:
        data = _make_request("POST", endpoint, json_data=payload)
        if data and data.get("results") and len(data["results"]) > 0:
            return data["results"][0].get("properties")
        return None
    except HubSpotAPIError as e:
        if e.status_code == 404: # Not found is not an error to retry indefinitely
            return None
        print(f"HubSpot API error getting contact {email}: {e}")
        # Depending on strictness, you might re-raise or return None
        return None # Or raise e
    except requests.exceptions.HTTPError as e: # Catch non-retriable HTTP errors from raise_for_status
        print(f"HTTP error getting contact {email}: {e.response.status_code} - {e.response.text}")
        return None


@hubspot_retry_decorator
def create_contact(email: str, firstname: Optional[str] = None, lastname: Optional[str] = None, **other_props) -> Optional[dict]:
    """Creates a new contact in HubSpot."""
    print(f"HubSpot: Creating contact: {email}")
    endpoint = "/crm/v3/objects/contacts"
    properties = {"email": email}
    if firstname:
        properties["firstname"] = firstname
    if lastname:
        properties["lastname"] = lastname
    properties.update(other_props)
    
    payload = {"properties": properties}
    try:
        data = _make_request("POST", endpoint, json_data=payload)
        return data.get("properties") # Or data if you need the full object
    except HubSpotAPIError as e:
        # Example: if a contact already exists, HubSpot might return a 409 or similar
        if e.status_code == 409: # Conflict - contact might already exist
            print(f"HubSpot: Contact {email} might already exist (409 Conflict). Attempting to fetch.")
            return get_contact_by_email(email) # Try fetching instead
        print(f"HubSpot API error creating contact {email}: {e}")
        return None # Or raise e
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error creating contact {email}: {e.response.status_code} - {e.response.text}")
        return None

# Add other HubSpot functions like create_deal, send_email_transactional etc.
# decorated with @hubspot_retry_decorator as needed.

# Example:
# @hubspot_retry_decorator
# def send_transactional_email(email_id: str, to_email: str, custom_properties: Optional[dict] = None):
#     print(f"HubSpot: Sending transactional email ID {email_id} to {to_email}")
#     endpoint = f"/marketing/v3/transactional/single-email/send"
#     payload = {
#         "emailId": email_id,
#         "message": {
#             "to": to_email
#         }
#     }
#     if custom_properties:
#         payload["customProperties"] = custom_properties # HubSpot format might vary
    
#     try:
#         # This endpoint might return 202 Accepted or similar, not JSON
#         response_raw = requests.post(f"{HUBSPOT_BASE_URL}{endpoint}",
#                                   headers={"Authorization": f"Bearer {HUBSPOT_API_KEY}", "Content-Type": "application/json"},
#                                   json=payload, timeout=10)
#         if not response_raw.ok:
#             if response_raw.status_code in RETRIABLE_STATUS_CODES:
#                 raise HubSpotAPIError(status_code=response_raw.status_code, message=response_raw.text)
#             response_raw.raise_for_status()
#         print(f"HubSpot: Transactional email {email_id} to {to_email} send request successful (Status: {response_raw.status_code}).")
#         return {"status": response_raw.status_code, "message": "Email send initiated."}
#     except HubSpotAPIError as e:
#         print(f"HubSpot API error sending email {email_id} to {to_email}: {e}")
#         return {"error": str(e)}
#     except requests.exceptions.HTTPError as e:
#         print(f"HTTP error sending email {email_id} to {to_email}: {e.response.status_code} - {e.response.text}")
#         return {"error": str(e.response.text)}


if __name__ == "__main__":
    print("Testing HubSpot client...")
    if not HUBSPOT_API_KEY:
        print("HUBSPOT_API_KEY not set. Cannot run HubSpot tests.")
    else:
        # Test get_contact_by_email
        test_email_exists = "test@hubspot.com" # Use an email you know exists in your portal for testing
        test_email_not_exists = "definitelynotexistingcontact12345@example.com"
        
        print(f"\nAttempting to get contact: {test_email_exists}")
        contact = get_contact_by_email(test_email_exists)
        if contact:
            print(f"Found contact: {contact.get('hs_object_id')}, {contact.get('email')}")
        else:
            print(f"Contact {test_email_exists} not found or error occurred.")

        print(f"\nAttempting to get contact: {test_email_not_exists}")
        contact_new = get_contact_by_email(test_email_not_exists)
        if contact_new:
            print(f"Found contact: {contact_new.get('hs_object_id')}, {contact_new.get('email')}")
        else:
            print(f"Contact {test_email_not_exists} not found, as expected.")

        # Test create_contact (use with caution, creates real contacts)
        # print(f"\nAttempting to create contact: {test_email_not_exists}")
        # created_contact = create_contact(test_email_not_exists, firstname="Test", lastname="UserFromScript")
        # if created_contact:
        #     print(f"Created contact: {created_contact.get('hs_object_id')}, {created_contact.get('email')}")
        # else:
        #     print(f"Failed to create contact {test_email_not_exists} or it already existed.") 