import os
import imaplib
import email
from email.header import decode_header
from dotenv import load_dotenv

load_dotenv()

# IMAP Credentials from .env
IMAP_HOST = os.getenv("IMAP_HOST", "imap.gmail.com")
IMAP_USER = os.getenv("IMAP_USER")
IMAP_PASS = os.getenv("IMAP_PASS")
IMAP_MAILBOX = os.getenv("EMAIL_MAILBOX", "INBOX") # Or a specific mailbox

def fetch_messages(max_emails=5, unseen_only=True) -> list[dict]:
    """
    Fetches emails from the configured IMAP server.
    Returns a list of dictionaries, where each dictionary contains
    email data like 'id', 'subject', 'body', 'from_address'.
    """
    if not all([IMAP_HOST, IMAP_USER, IMAP_PASS]):
        print("IMAP credentials (IMAP_HOST, IMAP_USER, IMAP_PASS) not fully configured in .env. Cannot fetch emails.")
        return []

    emails_data = []
    try:
        mail = imaplib.IMAP4_SSL(IMAP_HOST)
        mail.login(IMAP_USER, IMAP_PASS)
        mail.select(IMAP_MAILBOX)

        search_criteria = "UNSEEN" if unseen_only else "ALL"
        status, messages = mail.search(None, search_criteria)

        if status != "OK":
            print(f"Error searching emails: {status}")
            return []

        email_ids = messages[0].split()
        if not email_ids:
            print("No emails found matching criteria.")
            return []

        print(f"Found {len(email_ids)} email IDs. Fetching up to {max_emails}.")

        for i, email_id in enumerate(reversed(email_ids)): # Get newest first
            if i >= max_emails:
                break
            
            res, msg_data = mail.fetch(email_id, "(RFC822)")
            if res != "OK":
                print(f"Error fetching email ID {email_id.decode()}: {res}")
                continue

            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    
                    subject, encoding = decode_header(msg.get("Subject", "No Subject"))[0]
                    if isinstance(subject, bytes):
                        subject = subject.decode(encoding if encoding else "utf-8", errors="ignore")
                    
                    from_address = msg.get("From", "No Sender")

                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            content_type = part.get_content_type()
                            content_disposition = str(part.get("Content-Disposition"))
                            try:
                                if content_type == "text/plain" and "attachment" not in content_disposition:
                                    charset = part.get_content_charset()
                                    body_part = part.get_payload(decode=True)
                                    body += body_part.decode(charset if charset else "utf-8", errors="ignore")
                            except Exception as e:
                                print(f"Error decoding part for email ID {email_id.decode()}: {e}")
                    else:
                        try:
                            charset = msg.get_content_charset()
                            body_part = msg.get_payload(decode=True)
                            body = body_part.decode(charset if charset else "utf-8", errors="ignore")
                        except Exception as e:
                            print(f"Error decoding body for email ID {email_id.decode()}: {e}")
                    
                    emails_data.append({
                        "msg_id": email_id.decode(),
                        "subject": subject,
                        "body": body.strip(),
                        "from_address": from_address,
                        # You might want to add 'date', 'to_address', etc.
                    })
        mail.logout()
    except imaplib.IMAP4.error as e:
        print(f"IMAP Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during email fetching: {e}")
        # Ensure logout even on unexpected error if mail object exists
        if 'mail' in locals() and mail.state == 'SELECTED':
            try:
                mail.logout()
            except: # nosec
                pass # Ignore errors during cleanup logout
    
    return emails_data

if __name__ == "__main__":
    print("Testing email_loader.py...")
    if not all([IMAP_HOST, IMAP_USER, IMAP_PASS]):
        print("Please set IMAP_HOST, IMAP_USER, and IMAP_PASS in your .env file to run this test.")
    else:
        fetched_emails = fetch_messages(max_emails=2, unseen_only=False) # Fetch 2 emails, seen or unseen
        if fetched_emails:
            print(f"\nFetched {len(fetched_emails)} emails:")
            for em_data in fetched_emails:
                print("-" * 20)
                print(f"ID: {em_data['msg_id']}")
                print(f"From: {em_data['from_address']}")
                print(f"Subject: {em_data['subject']}")
                print(f"Body Preview: {em_data['body'][:100]}...")
        else:
            print("No emails were fetched.") 