import os
import requests
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import Optional
from datetime import datetime
import dateutil.parser
import json
import sys

load_dotenv()

HUBSPOT_ACCESS_TOKEN = os.getenv("HUBSPOT_ACCESS_TOKEN")
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

# Simpler retry for file downloads, focusing on connection/timeout
file_download_retry_decorator = retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=INITIAL_WAIT_SECONDS, max=MAX_WAIT_SECONDS),
    retry=retry_if_exception_type((requests.exceptions.ConnectionError, requests.exceptions.Timeout))
)

def _make_request(method, endpoint, params=None, json_data=None, headers=None):
    if not HUBSPOT_ACCESS_TOKEN:
        print("Error: HUBSPOT_ACCESS_TOKEN not set in .env", file=sys.stderr)
        raise ValueError("HubSpot Access Token not configured.")

    url = f"{HUBSPOT_BASE_URL}{endpoint}"
    default_headers = {
        "Authorization": f"Bearer {HUBSPOT_ACCESS_TOKEN}",
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
        print(f"HubSpot request failed for {method} {url}: {e}", file=sys.stderr)
        raise # Re-raise to be caught by tenacity or handled by caller

@hubspot_retry_decorator
def get_contact_by_email(email: str) -> Optional[dict]:
    """Fetches a contact by email from HubSpot."""
    print(f"HubSpot: Fetching contact by email: {email}", file=sys.stderr)
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
        print(f"HubSpot API error getting contact {email}: {e}", file=sys.stderr)
        # Depending on strictness, you might re-raise or return None
        return None # Or raise e
    except requests.exceptions.HTTPError as e: # Catch non-retriable HTTP errors from raise_for_status
        print(f"HTTP error getting contact {email}: {e.response.status_code} - {e.response.text}", file=sys.stderr)
        return None


@hubspot_retry_decorator
def create_contact(email: str, firstname: Optional[str] = None, lastname: Optional[str] = None, **other_props) -> Optional[dict]:
    """Creates a new contact in HubSpot."""
    print(f"HubSpot: Creating contact: {email}", file=sys.stderr)
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
            print(f"HubSpot: Contact {email} might already exist (409 Conflict). Attempting to fetch.", file=sys.stderr)
            return get_contact_by_email(email) # Try fetching instead
        print(f"HubSpot API error creating contact {email}: {e}", file=sys.stderr)
        return None
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error creating contact {email}: {e.response.status_code} - {e.response.text}", file=sys.stderr)
        return None

@hubspot_retry_decorator
def list_objects(
    object_type: str,
    properties: Optional[list[str]] = None,
    limit: int = 100,
    after: Optional[str] = None,
) -> dict:
    """
    Generic pager for HubSpot CRM objects (v3), e.g. feedback_submissions.
    """
    endpoint = f"/crm/v3/objects/{object_type}"
    params = {"limit": limit, "archived": "false"}
    if after:
        params["after"] = after
    if properties:
        params["properties"] = ",".join(properties)
    return _make_request("GET", endpoint, params=params)


def get_feedback_submissions(
    since: Optional[datetime] = None,
    survey_name_filter: Optional[str] = None,
) -> list[dict]:
    """
    Returns survey / feedback submissions after `since` (UTC) and optionally
    whose survey name *contains* `survey_name_filter` (case-insensitive).
    """
    props = [
        "hs_object_id",
        "hs_createdate",
        "hs_survey_name",
        "hs_feedback_submitted",
        "hs_sentiment_score",
    ]
    submissions, after = [], None

    while True:
        page = list_objects("feedback_submissions", properties=props, after=after)
        for item in page.get("results", []):
            created = dateutil.parser.isoparse(item["properties"]["hs_createdate"])
            if since and created < since:
                continue
            if survey_name_filter and survey_name_filter.lower() not in (
                item["properties"].get("hs_survey_name", "").lower()
            ):
                continue
            submissions.append(item)
        after = page.get("paging", {}).get("next", {}).get("after")
        if not after:
            break

    return submissions

@hubspot_retry_decorator
def get_deal_by_id(deal_id: str, properties: Optional[list[str]] = None) -> Optional[dict]:
    """Fetches a single deal by its ID from HubSpot."""
    print(f"HubSpot: Fetching deal by ID: {deal_id}", file=sys.stderr)
    endpoint = f"/crm/v3/objects/deals/{deal_id}"
    
    params = {}
    if properties:
        params["properties"] = ",".join(properties)
    else:
        # Default properties if none are specified from your provided file
        params["properties"] = "dealname,amount,dealstage,hs_object_id"

    try:
        api_response_data = _make_request("GET", endpoint, params=params)
        return api_response_data
        
    except HubSpotAPIError as e:
        print(f"HubSpot API error fetching deal {deal_id}: {e.status_code} - {e.message}", file=sys.stderr)
        if e.status_code == 404:
            print(f"HubSpot: Deal with ID {deal_id} specifically not found (404).", file=sys.stderr)
        return None
    except requests.exceptions.HTTPError as e: 
        print(f"HTTP error fetching deal {deal_id}: {e.response.status_code} - {e.response.text}", file=sys.stderr)
        return None
    except Exception as e: 
        print(f"An unexpected error occurred while fetching deal {deal_id}: {type(e).__name__} - {str(e)}", file=sys.stderr)
        return None

@hubspot_retry_decorator
def get_deal_associations(deal_id: str, to_object_type: str = "contacts") -> list[dict]:
    """
    Gets associations for a deal. 
    
    Args:
        deal_id: The ID of the deal
        to_object_type: The type of associated object to retrieve ("contacts", "companies", etc.)
    
    Returns:
        List of associated object IDs
    """
    print(f"HubSpot: Fetching {to_object_type} associations for deal: {deal_id}", file=sys.stderr)
    endpoint = f"/crm/v3/objects/deals/{deal_id}/associations/{to_object_type}"
    
    try:
        data = _make_request("GET", endpoint)
        return data.get("results", [])
    except HubSpotAPIError as e:
        if e.status_code == 404: # Not found
            print(f"HubSpot: Deal {deal_id} not found or no associations for {to_object_type}.", file=sys.stderr)
            return []
        print(f"HubSpot API error getting associations for deal {deal_id} ({to_object_type}): {e}", file=sys.stderr)
        return []
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error getting associations for deal {deal_id} ({to_object_type}): {e.response.status_code} - {e.response.text}", file=sys.stderr)
        return []

@hubspot_retry_decorator
def get_contact_by_id(contact_id: str, properties: Optional[list[str]] = None) -> Optional[dict]:
    """Fetches a contact by ID from HubSpot."""
    print(f"HubSpot: Fetching contact by ID: {contact_id}", file=sys.stderr)
    endpoint = f"/crm/v3/objects/contacts/{contact_id}"
    
    params = {}
    if properties:
        params["properties"] = ",".join(properties)
    else:
        # Default properties
        params["properties"] = "email,firstname,lastname,hs_object_id,company"
    
    try:
        data = _make_request("GET", endpoint, params=params)
        return data # Returns the full contact object including 'properties'
    except HubSpotAPIError as e:
        if e.status_code == 404: # Not found
            print(f"HubSpot: Contact {contact_id} not found.", file=sys.stderr)
            return None
        print(f"HubSpot API error getting contact {contact_id}: {e}", file=sys.stderr)
        return None
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error getting contact {contact_id}: {e.response.status_code} - {e.response.text}", file=sys.stderr)
        return None

@hubspot_retry_decorator
def get_company_by_id(company_id: str, properties: Optional[list[str]] = None) -> Optional[dict]:
    """Fetches a company by ID from HubSpot."""
    print(f"HubSpot: Fetching company by ID: {company_id}", file=sys.stderr)
    endpoint = f"/crm/v3/objects/companies/{company_id}"
    
    params = {}
    if properties:
        params["properties"] = ",".join(properties)
    else:
        # Default properties
        params["properties"] = "name,domain,hs_object_id,linkedin_company_page,description"
    
    try:
        data = _make_request("GET", endpoint, params=params)
        return data
    except HubSpotAPIError as e:
        if e.status_code == 404:
            print(f"HubSpot: Company {company_id} not found.", file=sys.stderr)
            return None
        print(f"HubSpot API error getting company {company_id}: {e}", file=sys.stderr)
        return None
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error fetching company by ID {company_id}: {e.response.status_code} - {e.response.text}", file=sys.stderr)
        return None

@hubspot_retry_decorator
def get_form_submissions_by_email(email: str, form_guid: Optional[str] = None) -> list[dict]:
    """
    Gets form submissions for a specific email address.
    Uses the Forms API v3 to get submissions.
    
    Args:
        email: The email address to search for
        form_guid: Optional specific form GUID to filter by
    
    Returns:
        List of form submissions
    """
    print(f"HubSpot: Fetching form submissions for email: {email}", file=sys.stderr)
    
    # First, get all forms to find the relevant form GUIDs
    forms_endpoint = "/marketing/v3/forms"
    try:
        forms_data = _make_request("GET", forms_endpoint)
        forms = forms_data.get("results", [])
        
        all_submissions = []
        
        for form in forms:
            if form_guid and form["id"] != form_guid:
                continue
                
            # Get submissions for this form
            # Note: The Forms API submissions endpoint does not directly filter by email.
            # We fetch all submissions for the form and then filter client-side.
            submissions_endpoint = f"/marketing/v3/forms/{form['id']}/submissions"
            
            # Parameters for fetching submissions for a specific form
            # The 'filters' param used in the original snippet is not standard for this endpoint.
            # We will fetch all and filter.
            page_params = {"limit": 100} # Adjust limit as needed
            
            try:
                # Paginate through submissions for this form
                while True:
                    sub_data = _make_request("GET", submissions_endpoint, params=page_params)
                    submissions_page = sub_data.get("results", [])
                    
                    for sub in submissions_page:
                        # Check if email matches (case-insensitive)
                        sub_email_found = None
                        for field in sub.get("values", []):
                            if field.get("name") == "email": # Assuming 'email' is the field name for email
                                sub_email_found = field.get("value", "").lower()
                                break
                        
                        if sub_email_found == email.lower():
                            # Add form name to submission
                            sub["form_name"] = form.get("name", "Unknown Form")
                            sub["form_id"] = form.get("id")
                            all_submissions.append(sub)
                    
                    # Check for next page
                    next_page_after = sub_data.get("paging", {}).get("next", {}).get("after")
                    if next_page_after:
                        page_params["after"] = next_page_after
                    else:
                        break # No more pages for this form's submissions
                        
            except Exception as e:
                print(f"Error getting submissions for form {form['id']} ({form.get('name', 'N/A')}): {e}", file=sys.stderr)
                continue # Move to the next form
        
        return all_submissions
        
    except Exception as e:
        print(f"Error getting list of forms (required for fetching submissions by email {email}): {e}", file=sys.stderr)
        return []

@hubspot_retry_decorator
def get_all_form_submissions(form_guid: Optional[str] = None, after: Optional[str] = None, limit: int = 50) -> dict:
    """
    Gets form submissions, optionally filtered by form GUID.
    
    Args:
        form_guid: Optional specific form GUID to get submissions for
        after: Pagination cursor
        limit: Number of results per page
    
    Returns:
        Dict with submissions and pagination info
    """
    if form_guid:
        endpoint = f"/marketing/v3/forms/{form_guid}/submissions"
        print(f"HubSpot: Fetching submissions for form GUID: {form_guid}", file=sys.stderr)
    else:
        # This endpoint might not exist for all forms - adjust as needed
        # Or this might be intended to list all forms, not submissions.
        # The Forms API doesn't have a single endpoint to list *all* submissions across *all* forms.
        # Typically, you list forms, then get submissions per form.
        # Using a placeholder or raising an error if form_guid is not provided might be safer.
        print("HubSpot: get_all_form_submissions called without form_guid. This usage might be ambiguous.", file=sys.stderr)
        endpoint = "/form-integrations/v1/submissions/forms" # This is likely an older or different API.
    
    params = {"limit": limit}
    if after:
        params["after"] = after
    
    try:
        return _make_request("GET", endpoint, params=params)
    except Exception as e:
        print(f"Error getting form submissions (endpoint: {endpoint}): {e}", file=sys.stderr)
        return {"results": [], "paging": {}}

def parse_form_submission_values(submission: dict) -> dict:
    """
    Parses form submission values into a more readable format.
    
    Args:
        submission: Raw form submission data from Marketing Forms API
    
    Returns:
        Dictionary with parsed field values
    """
    parsed = {
        "submitted_at": submission.get("submittedAt"), # HubSpot Forms API uses submittedAt
        "form_name": submission.get("form_name"), # Custom field added by get_form_submissions_by_email
        "form_id": submission.get("form_id"),     # Custom field added by get_form_submissions_by_email
        "values": {}
    }
    
    # 'values' is a list of {'name': ..., 'value': ...} dicts in Forms API submissions
    for field in submission.get("values", []):
        field_name = field.get("name", "unknown_field_name")
        field_value = field.get("value", "")
        parsed["values"][field_name] = field_value
    
    return parsed

@hubspot_retry_decorator
def get_pipeline_stages(pipeline_id: str, object_type: str = "deals") -> list[dict]:
    """
    Fetches all stages for a given pipeline.

    Args:
        pipeline_id: The ID of the pipeline.
        object_type: The type of object the pipeline is for (e.g., "deals", "tickets").

    Returns:
        A list of stage dictionaries, where each dictionary includes 'id' and 'label'.
        Returns an empty list if an error occurs or stages cannot be fetched.
    """
    print(f"HubSpot: Fetching stages for {object_type} pipeline ID: {pipeline_id}", file=sys.stderr)
    endpoint = f"/crm/v3/pipelines/{object_type}/{pipeline_id}/stages"
    try:
        data = _make_request("GET", endpoint)
        return data.get("results", [])
    except HubSpotAPIError as e:
        print(f"HubSpot API error fetching stages for pipeline {pipeline_id}: {e}", file=sys.stderr)
        return []
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error fetching stages for pipeline {pipeline_id}: {e.response.status_code} - {e.response.text}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"An unexpected error occurred while fetching stages for pipeline {pipeline_id}: {e}", file=sys.stderr)
        return []

# Renaming and modifying this function to focus on deal, contact, and company properties
def get_deal_with_associated_data(deal_id: str) -> dict:
    """
    Gets deal data with associated contacts and companies, focusing on their properties.
    
    Args:
        deal_id: The ID of the deal
    
    Returns:
        Dictionary containing deal info, contacts, and companies with their properties.
    """
    result = {
        "deal": {}, # Initialize deal as an empty dict
        "associated_contacts": [],
        "associated_companies": []
    }
    
    # Define properties to fetch for each object type
    deal_properties_to_fetch = [
        "dealname", "amount", "dealstage", "pipeline", "closedate", 
        "hubspot_owner_id", "hs_object_id", "createdate", "hs_lastmodifieddate",
        "request" 
    ]
    
    # !!! Use the ACTUAL INTERNAL HUBSPOT NAMES found from the script output !!!
    contact_custom_properties = [
        "hs_linkedin_url",                                      # For "Your LinkedIn or other social profile URL"
        "attach_link_to_all_founders_linkedin_profiles",      # For "Attach link to other founders LinkedIn profiles"
        "do_you_and_the_founders_have_the_right_to_work_in_the_uk_", # For "Do the founders have the right to work in the UK?"
        "list_name_of_all_founders",                          # For "List name of all founders" (often a contact property)
    ]
    contact_properties_to_fetch = [
        "email", "firstname", "lastname", "hs_object_id", "company", "phone", 
        "lifecyclestage", "hs_createdate", "lastmodifieddate", "jobtitle", "website", # Added website here as it's standard
    ] + contact_custom_properties
    
    company_properties_to_fetch = [
        "name", "domain", "hs_object_id", "industry", "city", "country", 
        "description", 
        "numberofemployees", # Standard HubSpot property
        "number_of_employees", # Alternative standard HubSpot property (with underscore)
        "lifecyclestage", "hs_createdate", 
        "hs_lastmodifieddate", "phone", "website",
        # Adding key webform fields for company properties based on your list
        "describe_the_business_product_in_one_sentence",
        "what_is_your_usp__what_makes_you_different_from_your_competitors_",
        "where_is_your_business_based_",
        "what_best_describes_your_stage_of_business_",
        "does_your_product_contribute_to_a_healthier__happier_whole_human_experience_",
        "how_does_your_product_contribute_to_a_healthier__happier_whole_human_experience_",
        "how_does_your_company_use_innovation__through_technology_or_to_differentiate_the_business_model__", # Use of innovation
        "what_sector_is_your_business_product_",
        "which__if_any__of_the_un_sdg_17_goals_does_your_business_address_",
        "what_best_describes_your_customer_base_",
        "how_many_employees_do_you_have__full_time_equivalents_", # Custom field for FTE employees
        "what_is_your_ltm__last_12_months__revenue_", # Corrected internal name for LTM Revenue
        "what_is_your_current_monthly_revenue_",    # Corrected internal name for Current Monthly Revenue
        "how_much_are_you_raising_at_this_stage_",
        "what_valuation_are_you_raising_at_",
        "how_much_have_you_raised_prior_to_this_round_",
        "how_much_of_the_equity_do_you_your_team_have_",
        "what_is_it_that_you_re_looking_for_with_a_partnership_from_flight_",
        "please_expand",
        "please_attach_your_pitch_deck"
    ]

    # Fetch the deal
    deal_full_object = get_deal_by_id(deal_id, properties=deal_properties_to_fetch)
    
    if deal_full_object and isinstance(deal_full_object, dict) and deal_full_object.get("properties"):
        result["deal"] = deal_full_object.get("properties")
        
        # Attempt to get deal stage label (using the existing get_pipeline_stages function)
        deal_stage_id = result["deal"].get("dealstage")
        pipeline_id = result["deal"].get("pipeline")
        deal_stage_label = "N/A"

        if deal_stage_id and pipeline_id:
            stages = get_pipeline_stages(pipeline_id, "deals")
            for stage in stages:
                if stage.get("id") == deal_stage_id:
                    deal_stage_label = stage.get("label", "N/A")
                    break
        result["deal"]["deal_stage_label"] = deal_stage_label
    else:
        print(f"HubSpot (get_deal_with_associated_data): Could not retrieve valid deal properties for deal ID {deal_id}. deal_full_object was: {json.dumps(deal_full_object, indent=2)}", file=sys.stderr)
        # Ensure deal_stage_label is present even if deal properties are missing
        if not isinstance(result["deal"], dict): result["deal"] = {} # Ensure result["deal"] is a dict
        result["deal"]["deal_stage_label"] = "N/A"

    # Get associated contacts
    contact_associations = get_deal_associations(deal_id, "contacts")
    print(f"Found {len(contact_associations)} associated contact IDs for deal {deal_id}", file=sys.stderr)
    
    for assoc in contact_associations:
        contact_id = assoc.get("id")
        if not contact_id:
            continue
        
        contact_full_object = get_contact_by_id(contact_id, properties=contact_properties_to_fetch)
        if contact_full_object:
            result["associated_contacts"].append(contact_full_object)
        else:
            print(f"Could not fetch details for contact ID: {contact_id}", file=sys.stderr)
    
    # Get associated companies
    company_associations = get_deal_associations(deal_id, "companies")
    print(f"Found {len(company_associations)} associated company IDs for deal {deal_id}", file=sys.stderr)
    
    for assoc in company_associations:
        company_id = assoc.get("id")
        if not company_id:
            continue
        
        company_full_object = get_company_by_id(company_id, properties=company_properties_to_fetch)
        if company_full_object:
            result["associated_companies"].append(company_full_object)
        else:
            print(f"Could not fetch details for company ID: {company_id}", file=sys.stderr)
            
    return result

@hubspot_retry_decorator
def get_all_contact_properties() -> list[tuple[str, str]]:
    """Lists all contact properties to find internal names and their display labels."""
    print("\nHubSpot: Fetching all contact properties...", file=sys.stderr)
    endpoint = "/crm/v3/properties/contacts"
    try:
        data = _make_request("GET", endpoint)
        # Ensure 'name' (internal name) and 'label' (display name) exist
        return [(prop['name'], prop['label']) for prop in data.get('results', []) if 'name' in prop and 'label' in prop]
    except Exception as e:
        print(f"Error fetching contact properties: {e}", file=sys.stderr)
        return []

@hubspot_retry_decorator  
def get_all_company_properties() -> list[tuple[str, str]]:
    """Lists all company properties to find internal names and their display labels."""
    print("\nHubSpot: Fetching all company properties...", file=sys.stderr)
    endpoint = "/crm/v3/properties/companies"
    try:
        data = _make_request("GET", endpoint)
        # Ensure 'name' (internal name) and 'label' (display name) exist
        return [(prop['name'], prop['label']) for prop in data.get('results', []) if 'name' in prop and 'label' in prop]
    except Exception as e:
        print(f"Error fetching company properties: {e}", file=sys.stderr)
        return []

# Alternative approach using CRM search for form submissions
@hubspot_retry_decorator
def search_form_submissions_by_contact(contact_email: str) -> list[dict]:
    """
    Search for form submissions using CRM search API.
    This is an alternative approach if the forms API doesn't work.
    NOTE: 'form_submissions' is not a standard searchable CRM object type. This may not work.
    """
    print(f"HubSpot: Searching form submissions for contact via CRM Search: {contact_email}", file=sys.stderr)
    
    # This endpoint is unlikely to work for "form_submissions" as it's not a standard CRM object.
    # Standard objects are contacts, companies, deals, tickets, etc.
    # Form submissions are typically handled via the Forms API.
    endpoint = "/crm/v3/objects/form_submissions/search" # Placeholder, likely to fail
    
    payload = {
        "filterGroups": [
            {
                "filters": [
                    {
                        # Property name 'email' might not exist on a generic 'form_submissions' CRM object.
                        "propertyName": "email", 
                        "operator": "EQ", 
                        "value": contact_email
                    }
                ]
            }
        ],
        "properties": ["hs_object_id", "hs_form_id", "hs_form_guid", "hs_portal_id", "hs_submission_timestamp"],
        "limit": 100
    }
    
    try:
        data = _make_request("POST", endpoint, json_data=payload)
        return data.get("results", [])
    except Exception as e:
        print(f"Error searching 'form_submissions' CRM object (this object type may not be searchable or exist): {e}", file=sys.stderr)
        return []

@file_download_retry_decorator
def download_file_from_url(file_url: str) -> Optional[bytes]:
    """
    Downloads a file from a given URL using the HubSpot access token for authentication.
    This is intended for URLs that might require HubSpot authentication to access,
    like signed URLs for form-uploaded files.
    """
    if not HUBSPOT_ACCESS_TOKEN:
        print("HubSpot access token not configured. Cannot download file.", file=sys.stderr)
        return None

    print(f"HubSpot Client: Attempting to download file from URL with authentication: {file_url}", file=sys.stderr)
    headers = {
        "Authorization": f"Bearer {HUBSPOT_ACCESS_TOKEN}",
        # User-Agent might be helpful
        'User-Agent': 'Mozilla/5.0 (Python HubSpot Client) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        # Important: allow_redirects=True is crucial here
        response = requests.get(file_url, headers=headers, timeout=30, allow_redirects=True)
        
        # Log the final URL after redirects
        final_url = response.url
        print(f"HubSpot Client: Final URL after redirects: {final_url}", file=sys.stderr)
        
        response.raise_for_status()  # Check for HTTP errors on the final response

        content_type = response.headers.get('Content-Type', '').lower()
        print(f"HubSpot Client: Content-Type of downloaded file: '{content_type}'", file=sys.stderr)

        if 'application/pdf' in content_type:
            print("HubSpot Client: Successfully downloaded PDF content.", file=sys.stderr)
            return response.content
        elif 'text/html' in content_type:
            print("HubSpot Client: Error - Received HTML content instead of a file. This might indicate a login page or an error page.", file=sys.stderr)
            # print(f"HubSpot Client: HTML content (first 500 chars): {response.text[:500]}")
            return None # Explicitly return None for HTML
        else:
            # It might be a different file type, or an error page without text/html
            print(f"HubSpot Client: Warning - Content-Type is '{content_type}', not 'application/pdf'. Returning content anyway.", file=sys.stderr)
            # For now, let's return it and let the PDF parser try.
            # If this becomes an issue, we might want to be stricter.
            return response.content

    except requests.exceptions.HTTPError as e:
        print(f"HubSpot Client: HTTP error downloading file {file_url}: {e.response.status_code} - {e.response.reason}", file=sys.stderr)
        # print(f"HubSpot Client: Response content (first 500 chars): {e.response.text[:500]}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"HubSpot Client: Request exception downloading file {file_url}: {e}", file=sys.stderr)
        return None
    except Exception as e_general:
        print(f"HubSpot Client: An unexpected error occurred downloading file {file_url}: {e_general}", file=sys.stderr)
        return None

# Keep existing functions like get_feedback_submissions if they are used elsewhere or for other purposes.
# The old get_deal_with_survey_data is effectively replaced by get_deal_with_form_data for this workflow.

if __name__ == "__main__":
    print("Testing HubSpot client...", file=sys.stderr)
    if not HUBSPOT_ACCESS_TOKEN:
        print("HUBSPOT_ACCESS_TOKEN not set. Cannot run HubSpot tests.", file=sys.stderr)
    else:
        # --- List Contact and Company Properties to find internal names ---
        # print(f"\n{'='*60}")
        # print("Listing relevant Contact Properties (to find internal names):")
        # print(f"{'='*60}")
        # contact_props_list = get_all_contact_properties()
        # if contact_props_list:
        #     found_relevant_contact_prop = False
        #     for internal_name, display_name in contact_props_list:
        #         if 'linkedin' in display_name.lower() or \
        #            'work' in display_name.lower() or \
        #            'founder' in display_name.lower() or \
        #            'eligibility' in display_name.lower() or \
        #            'url' in display_name.lower() or \
        #            'profile' in display_name.lower(): # Added more keywords
        #             print(f"  Contact Property: Internal Name = '{internal_name}', Display Label = '{display_name}'")
        #             found_relevant_contact_prop = True
        #     if not found_relevant_contact_prop:
        #         print("  No contact properties found matching keywords like 'linkedin', 'work', 'founder', 'eligibility', 'url', 'profile'.")
        #         print("  Consider printing all properties or adjusting keywords if expected properties are missing.")
        # else:
        #     print("  Could not retrieve contact properties.")

        # print(f"\n{'='*60}")
        # print("Listing some Company Properties (to find internal names):")
        # print(f"{'='*60}")
        # company_props_list = get_all_company_properties()
        # if company_props_list:
        #     print("  (Showing all company properties for review)")
        #     for internal_name, display_name in company_props_list:
        #         print(f"  Company Property: Internal Name = '{internal_name}', Display Label = '{display_name}'")
        # else:
        #     print("  Could not retrieve company properties.")
        # --- End Listing Properties ---

        TEST_DEAL_ID = "227710582988" # Your test deal ID
        
        print(f"\nFetching comprehensive data for deal: {TEST_DEAL_ID}", file=sys.stderr)
        # print(f"{'='*60}") # Optional: remove for less verbose output
        
        comprehensive_deal_data = get_deal_with_associated_data(TEST_DEAL_ID)
        
        output_filename = "deal_data_export.json"
        try:
            with open(output_filename, 'w') as f:
                json.dump(comprehensive_deal_data, f, indent=4)
            print(f"Successfully exported comprehensive deal data for deal {TEST_DEAL_ID} to: {output_filename}", file=sys.stderr)
        except Exception as e:
            print(f"Error writing data to JSON file {output_filename} for deal {TEST_DEAL_ID}: {e}", file=sys.stderr)

        # Comment out the detailed summary printout to reduce console noise
        # if comprehensive_deal_data.get("deal"):
        #     print(f"\nSummary of Fetched Data (also saved to {output_filename}):")
        #     deal_info = comprehensive_deal_data["deal"] 
            
        #     print(f"\nDeal Information:")
        #     print(f"  ID: {deal_info.get('hs_object_id', 'N/A')}")
        #     print(f"  Name: {deal_info.get('dealname', 'N/A')}")
        #     print(f"  Stage ID: {deal_info.get('dealstage', 'N/A')}") 
        #     print(f"  Stage Label: {deal_info.get('deal_stage_label', 'N/A')}")
        #     print(f"  Pipeline ID: {deal_info.get('pipeline', 'N/A')}")
        #     print(f"  Amount: {deal_info.get('amount', 'N/A')}")
        #     print(f"  % Request (Equity): {deal_info.get('request', 'N/A')}")
            
        #     if comprehensive_deal_data.get('associated_contacts'):
        #         print(f"\nAssociated Contacts Found: {len(comprehensive_deal_data['associated_contacts'])}")
        #         for i, contact_data in enumerate(comprehensive_deal_data['associated_contacts'], 1):
        #             contact_props = contact_data.get("properties", {})
        #             print(f"  Contact {i} ID: {contact_data.get('id')}")
        #             # Print fewer contact details or none
        #             print(f"    Email: {contact_props.get('email', 'N/A')}")


        #     if comprehensive_deal_data.get('associated_companies'):
        #         print(f"\nAssociated Companies Found: {len(comprehensive_deal_data['associated_companies'])}")
        #         for i, company_data in enumerate(comprehensive_deal_data['associated_companies'], 1):
        #             company_props = company_data.get("properties", {})
        #             print(f"  Company {i} ID: {company_data.get('id')}")
        #             # Print fewer company details or none
        #             print(f"    Company Name: {company_props.get('name', 'N/A')}")
        # else:
        #     print(f"Could not retrieve sufficient data for deal {TEST_DEAL_ID} to print summary or export.", file=sys.stderr) 