import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal, Optional, List
import sys

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Load .env from project root
load_dotenv()

# --- Pydantic Model for Classification Output ---
class EmailClassification(BaseModel):
    """Schema for classifying an email's intent and extracting key information."""
    category: Literal["investment_opportunity", "job_application", "recruitment_inquiry", "networking", "vendor_pitch", "personal", "spam", "other"] = Field(description="The primary category of the email.")
    is_urgent: bool = Field(description="Whether the email requires immediate attention.")
    summary: str = Field(description="A brief one-sentence summary of the email's content.")
    extracted_company_name: Optional[str] = Field(None, description="Name of the company mentioned, if any (e.g., applicant's current/previous, investor's fund, vendor's company).")
    extracted_person_name: Optional[str] = Field(None, description="Name of the key person mentioned (e.g., applicant, sender, contact person).")
    funding_ask_amount_usd: Optional[float] = Field(None, description="If it's an investment opportunity, the amount of funding asked for in USD. Extract only the number.")
    job_title_mentioned: Optional[str] = Field(None, description="If it's a job application or recruitment, the job title mentioned.")
    recent_survey_responses: Optional[List[dict]] = Field(None, description="Recent survey responses related to the sender or company, if available.")

# --- LLM and Parser Setup ---
classification_llm = None
if os.getenv("ANTHROPIC_API_KEY"):
    print("classify_email_tool: Using Anthropic Claude model.", file=sys.stderr)
    classification_llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.1)
elif os.getenv("OPENAI_API_KEY"):
    print("classify_email_tool: Using OpenAI GPT model.", file=sys.stderr)
    classification_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
else:
    print("classify_email_tool: Warning - No LLM API key found for classification tool.", file=sys.stderr)

classification_parser = PydanticOutputParser(pydantic_object=EmailClassification)

classification_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert email classification system. Analyze the provided email content (subject and body) "
            "and classify it according to the following categories: investment_opportunity, job_application, "
            "recruitment_inquiry, networking, vendor_pitch, personal, spam, other. "
            "Determine if it's urgent. Provide a concise summary. "
            "Extract relevant entities like company name, person name, funding ask (in USD, numbers only), and job title if applicable. "
            "Respond ONLY with the JSON format described below:\n{format_instructions}"
        ),
        ("human", "Email Subject: {subject}\n\nEmail Body:\n{body}"),
    ]
).partial(format_instructions=classification_parser.get_format_instructions())

@tool
def classify_email_tool(subject: str, body: str) -> dict:
    """
    Classifies an email based on its subject and body content.
    Returns a structured dictionary with category, urgency, summary, and extracted entities.
    """
    if not classification_llm:
        return {"error": "LLM for classification not available. Check API keys."}

    chain = classification_prompt_template | classification_llm | classification_parser
    print(f"classify_email_tool: Classifying email - Subject: '{subject[:50]}...'")
    try:
        result = chain.invoke({"subject": subject, "body": body})
        return result.model_dump()
    except Exception as e:
        print(f"Error during email classification: {e}", file=sys.stderr)
        return {"error": f"Failed to classify email: {str(e)}"}

if __name__ == "__main__":
    print("Testing classify_email_tool...", file=sys.stderr)
    if not classification_llm:
        print("Cannot run test for classify_email_tool: LLM not configured.", file=sys.stderr)
    else:
        print("Testing classify_email_tool...", file=sys.stderr)
        test_subject = "Investment Opportunity: AI Startup Synergix"
        test_body = (
            "Dear Investor,\n\nWe are Synergix, a cutting-edge AI startup seeking $500,000 in seed funding "
            "to revolutionize the logistics industry. Our CEO is Jane Doe.\n\nBest,\nJohn Smith"
        )
        classification = classify_email_tool.invoke({"subject": test_subject, "body": test_body})
        print("\nClassification Result:", file=sys.stderr)
        import json
        print(json.dumps(classification, indent=2), file=sys.stderr)

        test_subject_job = "Application for Software Engineer Position - John Applicant"
        test_body_job = "Dear Hiring Manager, I am writing to apply for the Software Engineer role advertised on your website. My resume is attached. I previously worked at Tech Solutions Inc."
        classification_job = classify_email_tool.invoke({"subject": test_subject_job, "body": test_body_job})
        print("\nJob Application Classification Result:", file=sys.stderr)
        print(json.dumps(classification_job, indent=2), file=sys.stderr) 