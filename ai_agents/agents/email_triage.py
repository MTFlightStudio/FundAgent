import os
import re
import time
from dotenv import load_dotenv

# Assuming email_loader exists and has fetch_messages
from ai_agents.loaders import email_loader # Or specific function: from ai_agents.loaders.email_loader import fetch_messages
from ai_agents.tools import classify_email_tool, save_tool # Using absolute imports
# Assuming hubspot_client exists
from ai_agents.services import hubspot_client # Or specific functions

load_dotenv()

# --- Configuration & Hard Filters ---
# Example: Define your hard filters here
# These run before any LLM calls to quickly discard obvious spam or irrelevant emails.
HARD_FILTER_KEYWORDS = [
    "unsubscribe", "lottery", "winner", "free money", "viagra", "cialis",
    "invoice overdue", "shipping notification", "order confirmation" # Examples of common non-actionable emails
]
HARD_FILTER_SENDER_DOMAINS = [
    "example-spam.com", "junkmail.org"
]

def hard_filters(email_data: dict) -> bool:
    """
    Apply hard filters to an email.
    Returns True if the email should be DISCARDED, False otherwise.
    email_data is expected to have 'subject', 'body', 'from_address' keys.
    """
    subject = email_data.get("subject", "").lower()
    body = email_data.get("body", "").lower()
    from_address = email_data.get("from_address", "").lower()

    for keyword in HARD_FILTER_KEYWORDS:
        if keyword in subject or keyword in body:
            print(f"Hard filter: Discarding email due to keyword '{keyword}' in subject/body.")
            return True

    for domain in HARD_FILTER_SENDER_DOMAINS:
        if domain in from_address:
            print(f"Hard filter: Discarding email from sender domain '{domain}'.")
            return True

    # Add raise-amount regex
    # This regex looks for amounts like £1,000,000m, 1000m, £500,000, $2m etc.
    # It captures the numeric part before 'm' or just a large number.
    # This is a simple example; real-world currency parsing can be complex.
    raise_amount_pattern = re.compile(r"[£€\$]?([\d,]+(?:\.\d+)?)\s*m(?:illion)?|[£€\$]([\d,]{4,})")
    if raise_amount_pattern.search(subject) or raise_amount_pattern.search(body):
        # This is an example of identifying a potential investment pitch characteristic.
        # The action here might be to *not* discard, or to flag for specific processing.
        # For this example, let's say finding a raise amount means it passes this specific filter.
        print(f"Hard filter: Detected potential raise amount. Email passes this specific filter.")
        # Depending on logic, you might return False here or continue other checks.
        # For now, let's assume this is just an observation and doesn't auto-discard.
        pass # Continue to other filters or return False if this is the only relevant check

    # Add more filters as needed (e.g., sender reputation, known blacklists)

    return False # If no hard filters match, do not discard

# --- Main Processing Logic ---
def process_single_email(email_data: dict):
    """
    Processes a single email: applies hard filters, classifies, and takes action.
    email_data: dict with 'id', 'subject', 'body', 'from_address', etc.
    """
    print(f"\nProcessing email ID: {email_data.get('id', 'N/A')}, Subject: {email_data.get('subject', '')}")

    if hard_filters(email_data):
        print(f"Email ID {email_data.get('id', 'N/A')} discarded by hard filters.")
        # Optionally, move to a 'Junk' folder via email_loader if such functionality exists
        return

    # Classify email using the LLM tool
    classification_result = classify_email_tool.invoke({
        "subject": email_data.get("subject", ""),
        "body": email_data.get("body", "")
    })

    if classification_result.get("error"):
        print(f"Error classifying email ID {email_data.get('id', 'N/A')}: {classification_result['error']}")
        # Save raw email for manual review
        save_tool.invoke({
            "filename": f"error_email_{email_data.get('id', 'N/A')}.txt",
            "text": f"Subject: {email_data.get('subject', '')}\n\nBody:\n{email_data.get('body', '')}\n\nError: {classification_result['error']}"
        })
        return

    print(f"Classification for email ID {email_data.get('id', 'N/A')}: {classification_result}")

    # TODO: Implement actions based on classification_result
    # Example:
    category = classification_result.get("category")
    summary = classification_result.get("summary")
    from_email = email_data.get("from_address") # Assuming 'from_address' is in email_data

    if category == "investment_opportunity":
        print(f"Action: Identified investment opportunity from {from_email}. Summary: {summary}")
        # Example: Create a deal in HubSpot
        # contact_info = hubspot_client.get_or_create_contact(from_email, classification_result.get("extracted_person_name"))
        # if contact_info and contact_info.get("id"):
        #     deal_payload = {
        #         "properties": {
        #             "dealname": f"Investment Inquiry: {classification_result.get('extracted_company_name', from_email)}",
        #             "pipeline": "default", # Your sales pipeline ID
        #             "dealstage": "appointmentscheduled", # Your initial deal stage ID
        #             "amount": classification_result.get("funding_ask_amount_usd"),
        #             "hubspot_owner_id": "YOUR_OWNER_ID" # Assign to a HubSpot user
        #         },
        #         "associations": [
        #             {
        #                 "to": {"id": contact_info.get("id")},
        #                 "types": [{"associationCategory": "HUBSPOT_DEFINED", "associationTypeId": 2}] # Deal to Contact
        #             }
        #         ]
        #     }
        #     # hubspot_client.create_deal(deal_payload)
        #     print("HubSpot deal creation placeholder called.")
        pass
    elif category == "job_application":
        print(f"Action: Identified job application from {from_email} for {classification_result.get('job_title_mentioned')}. Summary: {summary}")
        # Example: Save to a specific folder, notify HR
        # save_tool.invoke({
        #     "filename": f"job_application_{email_data.get('id', 'N/A')}.txt",
        #     "text": f"Subject: {email_data.get('subject', '')}\nBody:\n{email_data.get('body', '')}\nClassification: {classification_result}"
        # })
        pass
    # Add more actions for other categories...
    else:
        print(f"Action: Email category '{category}' from {from_email}. Needs review or generic handling. Summary: {summary}")
        # save_tool.invoke({
        #     "filename": f"other_email_{email_data.get('id', 'N/A')}.txt",
        #     "text": f"Subject: {email_data.get('subject', '')}\nBody:\n{email_data.get('body', '')}\nClassification: {classification_result}"
        # })

    # After processing, mark email as read or move it, using email_loader functions
    # email_loader.mark_as_read(email_data.get('id'))
    # email_loader.move_email(email_data.get('id'), "Processed")


def process_inbox(run_once=False):
    """
    Fetches and processes emails from the inbox.
    Loops continuously if run_once is False.
    """
    print(f"Starting email triage process (run_once={run_once})...")
    while True:
        print("Fetching new messages...")
        try:
            # Ensure fetch_messages is correctly imported and called
            # It should return a list of email data dictionaries
            messages = email_loader.fetch_messages(max_emails=10) # Example: fetch up to 10 new emails
            if not messages:
                print("No new messages found.")
            else:
                print(f"Fetched {len(messages)} new messages.")
                for email_data in messages:
                    process_single_email(email_data)

        except Exception as e:
            print(f"An error occurred in the email processing loop: {e}")
            # Add more robust error handling, e.g., backoff, specific error types

        if run_once:
            print("Email triage process finished (run_once=True).")
            break
        else:
            print("Waiting for 60 seconds before next fetch...")
            time.sleep(60) # Check for new emails every 60 seconds

if __name__ == "__main__":
    # This is for direct testing of this module.
    # The main CLI entry point is ai_agents/cli.py
    print("Running email_triage.py directly for testing...")
    # To test, you might want to mock email_loader.fetch_messages()
    # or have a test email account configured.
    # For now, let's simulate one email for process_single_email
    sample_email = {
        "id": "test_email_001",
        "subject": "Urgent: Investment Proposal for AI Synergies",
        "body": "Dear team, We propose a £2m investment for our new AI project. Regards, Bob.",
        "from_address": "bob@example.com"
    }
    process_single_email(sample_email)

    # To test the loop (requires email_loader setup):
    # process_inbox(run_once=True) 