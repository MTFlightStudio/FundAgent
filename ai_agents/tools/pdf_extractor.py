import os
import io
import json
import re
import requests
import pdfplumber # For PDF text extraction
from typing import Dict, Any, Optional, List
from urllib.parse import urljoin # Added for constructing absolute URLs
from bs4 import BeautifulSoup # Added for HTML parsing

from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
import sys # Add sys import

from ai_agents.config.model_config import get_llm_for_agent
from ai_agents.utils.retry_handler import with_smart_retry, RetryConfig

# Load .env file from the project root
load_dotenv()

# Attempt to import the HubSpot client
try:
    from ai_agents.services import hubspot_client
except ImportError:
    hubspot_client = None
    print("Warning: ai_agents.services.hubspot_client not found. HubSpot specific PDF download will not be available.", file=sys.stderr)

# --- Pydantic Model for Structured Pitch Deck Information ---
class PitchDeckSections(BaseModel):
    executive_summary: Optional[str] = Field(None, description="A brief overview of the company and the pitch.")
    problem_statement: Optional[str] = Field(None, description="The problem the company is trying to solve.")
    solution: Optional[str] = Field(None, description="The company's solution to the problem.")
    product_service_description: Optional[str] = Field(None, description="Detailed description of the product or service.")
    market_size_opportunity: Optional[str] = Field(None, description="Information about the target market size and opportunity.")
    business_model: Optional[str] = Field(None, description="How the company makes money.")
    team: Optional[str] = Field(None, description="Information about the founding team and key personnel.")
    traction_milestones: Optional[str] = Field(None, description="Achievements, traction, and milestones reached.")
    financials_projections: Optional[str] = Field(None, description="Key financial data, past performance, and future projections.")
    funding_ask_use_of_funds: Optional[str] = Field(None, description="The amount of funding being sought and how it will be used.")
    competition: Optional[str] = Field(None, description="Analysis of competitors and competitive advantages.")
    go_to_market_strategy: Optional[str] = Field(None, description="The company's plan for reaching customers.")


# --- LLM Initialization ---

# --- PDF Processing Functions ---

def download_pdf_content(url: str) -> Optional[bytes]:
    """Downloads PDF content from a URL.
    If it's a HubSpot URL, it attempts to use the HubSpot client for authenticated download.
    Otherwise, it falls back to a generic request and HTML parsing for redirects.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    # Check if it's a HubSpot URL and if the client is available
    if hubspot_client and ("hubspot.com" in url or "hubspotusercontent.com" in url):
        print(f"Detected HubSpot URL: {url}. Attempting download via HubSpot client.", file=sys.stderr)
        # The hubspot_client.download_file_from_url should handle authentication and redirects
        pdf_bytes = hubspot_client.download_file_from_url(url)
        if pdf_bytes:
            # We trust the hubspot_client function to have checked content type or returned None
            return pdf_bytes
        else:
            print(f"HubSpot client failed to download PDF from {url}. Falling back to generic download if applicable, or failing.", file=sys.stderr)
            # Decide if you want to fall back to the generic method or just fail here.
            # For now, let's let it fail if the HubSpot client couldn't get it,
            # as the generic method also failed previously due to auth.
            return None 
            # If you wanted to try the generic method as a last resort (less likely to work for auth-protected HubSpot URLs):
            # print("Falling back to generic download method for HubSpot URL (less likely to succeed).")
            # pass # and let the code below run

    # Generic download attempt (for non-HubSpot URLs or as a fallback if explicitly designed)
    try:
        print(f"Attempting generic download from URL: {url}", file=sys.stderr)
        response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        response.raise_for_status()

        content_type = response.headers.get('Content-Type', '').lower()
        final_url_after_redirects = response.url
        print(f"Generic download: Initial response Content-Type: '{content_type}' from URL: {final_url_after_redirects}", file=sys.stderr)

        if 'application/pdf' in content_type:
            print("Generic download: Directly received PDF content.", file=sys.stderr)
            return response.content
        elif 'text/html' in content_type:
            print("Generic download: Received HTML. Attempting to find PDF link within HTML...", file=sys.stderr)
            soup = BeautifulSoup(response.content, 'html.parser')
            pdf_link_tag = None
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                if href.lower().endswith('.pdf') or '.pdf?' in href.lower():
                    pdf_link_tag = a_tag
                    break
            
            if not pdf_link_tag:
                meta_refresh = soup.find('meta', attrs={'http-equiv': 'refresh'})
                if meta_refresh and 'content' in meta_refresh.attrs:
                    match = re.search(r'url=([\'"]?)([^\'" >]+)\1', meta_refresh['content'], re.IGNORECASE)
                    if match:
                        potential_url = match.group(2)
                        if potential_url.lower().endswith('.pdf') or '.pdf?' in potential_url.lower():
                            pdf_link_tag = {'href': potential_url}
                            print(f"Generic download: Found PDF link in meta refresh: {potential_url}", file=sys.stderr)

            if pdf_link_tag and pdf_link_tag.get('href'):
                pdf_url_from_html = pdf_link_tag['href']
                if not pdf_url_from_html.startswith(('http://', 'https://')):
                    pdf_url_from_html = urljoin(final_url_after_redirects, pdf_url_from_html)
                
                print(f"Generic download: Found potential PDF link in HTML: {pdf_url_from_html}. Attempting download...", file=sys.stderr)
                pdf_response = requests.get(pdf_url_from_html, headers=headers, timeout=30, allow_redirects=True)
                pdf_response.raise_for_status()
                
                pdf_content_type = pdf_response.headers.get('Content-Type', '').lower()
                print(f"Generic download: Second download Content-Type: '{pdf_content_type}' from URL: {pdf_response.url}", file=sys.stderr)

                if 'application/pdf' in pdf_content_type:
                    print("Generic download: Successfully downloaded PDF from extracted link.", file=sys.stderr)
                    return pdf_response.content
                else:
                    print(f"Generic download: Error - Link found in HTML ({pdf_url_from_html}) did not return PDF. Final Content-Type: {pdf_content_type}", file=sys.stderr)
                    return None
            else:
                print("Generic download: Error - Received HTML, but no .pdf link or meta refresh found.", file=sys.stderr)
                return None
        else:
            print(f"Generic download: Warning - Content-Type is '{content_type}'. Expected 'application/pdf' or 'text/html'. Returning content.", file=sys.stderr)
            return response.content

    except requests.exceptions.RequestException as e:
        print(f"Generic download: Error during HTTP request for {url} (or linked URL): {e}", file=sys.stderr)
        return None
    except Exception as e_general:
        print(f"Generic download: An unexpected error occurred for {url}: {e_general}", file=sys.stderr)
        return None

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> Optional[str]:
    """Extracts text from PDF content bytes using pdfplumber."""
    if not pdf_bytes:
        return None
    text = ""
    try:
        with io.BytesIO(pdf_bytes) as pdf_file:
            with pdfplumber.open(pdf_file) as pdf:
                if not pdf.pages:
                    print("Warning: PDF has no pages.", file=sys.stderr)
                    return None
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n" # Add separator for readability
                    else:
                        print(f"Warning: No text extracted from page {i+1}. It might be image-based.", file=sys.stderr)
        
        if not text.strip():
            print("Warning: No text could be extracted from the PDF. It might be entirely image-based or corrupted. OCR would be needed for image-based PDFs.", file=sys.stderr)
            return None
        return text.strip()
    except Exception as e: # Catches pdfplumber specific errors and others
        print(f"Error extracting text from PDF: {e}", file=sys.stderr)
        return None

@with_smart_retry(
    "pdf_extraction",
    retry_config_override=RetryConfig(switch_model_on_rate_limit=True),
    model_selector_func=lambda *args, **kwargs: (
        print(f"--- DEBUG LAMBDA: Received args: {args}, kwargs: {kwargs} ---", file=sys.stderr),
        get_llm_for_agent(args[0], prefer_fast=kwargs.get('prefer_fast', False))
    )[1]
)
def structure_text_with_llm(raw_text: str, **kwargs) -> Optional[PitchDeckSections]:
    """Uses an LLM to structure the extracted text into PitchDeckSections.
    This function is decorated with @with_smart_retry, which handles LLM instantiation,
    retries, and potential model switching.
    The 'llm' and 'model_config' are injected by the decorator via **kwargs.
    """
    llm: Optional[BaseChatModel] = kwargs.get("llm")
    # model_config: Optional[ModelConfig] = kwargs.get("model_config") # Available if needed

    if not llm:
        # This case should ideally be handled by get_llm_for_agent raising an error
        # if no model can be provided (e.g., API keys missing), which @with_smart_retry would catch.
        print("LLM not available for structuring text (not provided by decorator).", file=sys.stderr)
        return None

    parser = PydanticOutputParser(pydantic_object=PitchDeckSections)

    prompt_text = """
    You are an expert assistant tasked with extracting structured information from pitch deck text.
    Based on the following raw text from a pitch deck, please extract the relevant information and structure it strictly according to the provided JSON schema.

    JSON Schema:
    {schema}

    Raw Pitch Deck Text:
    ---
    {raw_text}
    ---

    Instructions:
    1. Analyze the raw text carefully.
    2. Populate the fields in the JSON schema with information extracted from the text.
    3. If information for a specific field is not found, use `null` for that field.
    4. IMPORTANT: Your response MUST be ONLY the JSON object that conforms to the schema. Do NOT include any other text, explanations, or conversational preamble before or after the JSON object.
    
    Output ONLY the JSON object:
    """

    prompt_template = ChatPromptTemplate.from_template(
        template=prompt_text,
        partial_variables={"schema": parser.get_format_instructions()}
    )

    chain = prompt_template | llm | parser
    
    try:
        # The retry logic is now handled by the @with_smart_retry decorator.
        # The decorator will call this function multiple times if retries are needed.
        structured_data = chain.invoke({"raw_text": raw_text})
        return structured_data
    except Exception as e:
        # This error will be caught by @with_smart_retry if it's a retryable error.
        # If it's not retryable, or retries are exhausted, the decorator will re-raise it or handle it.
        print(f"Error structuring text with LLM during chain invocation (will be handled by retry decorator): {e}", file=sys.stderr)
        raise # Re-raise for the decorator to handle


# --- Main Orchestration Function ---
def process_pdf_source(source: str, attempt_llm_structuring: bool = True) -> Dict[str, Any]:
    """
    Processes a PDF from a URL or a local file path.

    Args:
        source: The URL or local file path of the PDF.
        attempt_llm_structuring: Whether to try structuring the text with an LLM.

    Returns:
        A dictionary containing:
        - "status": "success" or "error"
        - "raw_text": The extracted text from the PDF (if successful).
        - "structured_data": A dictionary of the structured pitch deck sections (if LLM structuring is successful).
        - "error_message": A message describing the error (if status is "error").
        - "warnings": A list of warnings encountered during processing.
    """
    result: Dict[str, Any] = {
        "status": "error",
        "raw_text": None,
        "structured_data": None,
        "error_message": None,
        "warnings": []
    }

    if not source or not (source.startswith("http://") or source.startswith("https://")):
        result["error_message"] = "Invalid URL provided."
        return result

    print(f"Processing PDF from URL: {source}", file=sys.stderr)

    pdf_content = download_pdf_content(source)
    if not pdf_content:
        result["error_message"] = f"Failed to download PDF from {source}."
        return result
    
    print("PDF downloaded successfully. Extracting text...", file=sys.stderr)
    raw_text = extract_text_from_pdf_bytes(pdf_content)

    if raw_text is None or not raw_text.strip():
        result["warnings"].append("Could not extract significant text from the PDF. It might be image-based, empty, or corrupted.")
        if not raw_text:
             result["error_message"] = "No text could be extracted from the PDF."
             return result


    result["raw_text"] = raw_text
    result["status"] = "success" 
    print(f"Text extracted. Length: {len(raw_text)} characters.", file=sys.stderr)

    if attempt_llm_structuring and raw_text:
        print("Attempting to structure text with LLM...", file=sys.stderr)
        try:
            # Call structure_text_with_llm without llm arg, decorator handles it
            structured_info = structure_text_with_llm(raw_text=raw_text) # Pass raw_text as keyword argument for clarity
            if structured_info:
                result["structured_data"] = structured_info.model_dump()
                print("Text successfully structured by LLM.", file=sys.stderr)
            else:
                # This else block might be hit if structure_text_with_llm returns None
                # after exhausting retries for non-exception-raising "soft" failures,
                # or if the initial llm is None and not handled by an exception.
                # The current structure_text_with_llm re-raises exceptions, so those
                # will be caught by the outer try/except here.
                msg = "Failed to structure text using LLM (returned None or an issue occurred that wasn't an exception from the LLM call itself), but raw text is available."
                print(msg, file=sys.stderr)
                result["warnings"].append(msg)
        except Exception as e:
            # This will catch errors from structure_text_with_llm if they are not handled
            # by the retry decorator (e.g., non-retryable errors or after all retries failed).
            # It will also catch errors if get_llm_for_agent fails to return an LLM (e.g. no API keys).
            error_type = type(e).__name__
            msg = f"LLM structuring failed due to an error: {error_type} - {str(e)}. Raw text is available."
            print(msg, file=sys.stderr)
            result["warnings"].append(msg)
            
    elif not raw_text:
        msg = "Skipping LLM structuring as no raw text was extracted."
        print(msg, file=sys.stderr)
        result["warnings"].append(msg)


    if not result["warnings"]: 
        del result["warnings"]

    return result


if __name__ == "__main__":
    print("--- DIAGNOSTIC: Testing get_llm_for_agent('pdf_extraction') directly ---", file=sys.stderr)
    try:
        llm_instance, model_config_instance = get_llm_for_agent("pdf_extraction")
        if llm_instance and model_config_instance:
            print(f"--- DIAGNOSTIC: Successfully got LLM: {model_config_instance.model_name} (Provider: {model_config_instance.provider.value}) ---", file=sys.stderr)
        else:
            print(f"--- DIAGNOSTIC: get_llm_for_agent returned None for llm or config but no exception. LLM: {llm_instance}, Config: {model_config_instance} ---", file=sys.stderr)
    except Exception as e_diag:
        print(f"--- DIAGNOSTIC: Error directly calling get_llm_for_agent('pdf_extraction'): {type(e_diag).__name__} - {str(e_diag)} ---", file=sys.stderr)
    print("--- END DIAGNOSTIC ---", file=sys.stderr)

    print("Testing PDF Extractor...", file=sys.stderr)
    
    # Example HubSpot URL from your provided JSON
    test_url_hubspot = "https://api-eu1.hubspot.com/form-integrations/v1/uploaded-files/signed-url-redirect/241972526310?portalId=145458831&sign=8xy8NABPw0AeW8Z1BM2T9PB9sPQ%3D&conversionId=0470020b-17a8-48e5-ae87-0d56ea730f9c&filename=0470020b-17a8-48e5-ae87-0d56ea730f9c-0-2/please_attach_your_pitch_deck-LessDistress-PitchDeck.pdf"
    
    # A publicly accessible PDF for testing
    test_url_public_pdf = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"

    results_to_save = {}

    # --- Test with HubSpot URL (LLM structuring enabled) ---
    print(f"\n--- Testing HubSpot URL: {test_url_hubspot} ---", file=sys.stderr)
    # Ensure API keys are set in .env for this test to fully work with LLM structuring
    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")):
        print("WARNING: ANTHROPIC_API_KEY or OPENAI_API_KEY not set. LLM structuring will be skipped or fail.", file=sys.stderr)
        print("Please set them in your .env file if you want to test LLM structuring.", file=sys.stderr)
        
    hubspot_result = process_pdf_source(test_url_hubspot, attempt_llm_structuring=True)
    results_to_save["hubspot_test"] = hubspot_result
    
    print("\n--- HubSpot URL Processing Result (Console) ---", file=sys.stderr)
    if hubspot_result["status"] == "success":
        print("Status: Success", file=sys.stderr)
        if hubspot_result["raw_text"]:
            print(f"Raw Text (first 500 chars): {hubspot_result['raw_text'][:500]}...", file=sys.stderr)
        if hubspot_result["structured_data"]:
            print("Structured Data:", file=sys.stderr)
            print(json.dumps(hubspot_result["structured_data"], indent=2), file=sys.stderr)
        if hubspot_result.get("warnings"):
            print("Warnings:", hubspot_result["warnings"], file=sys.stderr)
    else:
        print(f"Status: Error - {hubspot_result.get('error_message', 'Unknown error')}", file=sys.stderr)
        if hubspot_result.get("warnings"): # Also print warnings on error
            print("Warnings:", hubspot_result["warnings"], file=sys.stderr)


    # --- OPTIONAL: Test with a simple public PDF (LLM structuring enabled to test LLM part) ---
    print(f"\n--- Testing Public PDF URL (LLM Structuring Enabled): {test_url_public_pdf} ---", file=sys.stderr)
    public_pdf_result = process_pdf_source(test_url_public_pdf, attempt_llm_structuring=True)
    results_to_save["public_pdf_test_structured"] = public_pdf_result
    print("\n--- Public PDF URL Processing Result (Console) ---", file=sys.stderr)
    if public_pdf_result["status"] == "success":
        print("Status: Success", file=sys.stderr)
        if public_pdf_result["raw_text"]:
            print(f"Raw Text (first 500 chars): {public_pdf_result['raw_text'][:500]}...", file=sys.stderr)
        if public_pdf_result["structured_data"]:
            print("Structured Data:", file=sys.stderr)
            print(json.dumps(public_pdf_result["structured_data"], indent=2), file=sys.stderr)
        if public_pdf_result.get("warnings"):
            print("Warnings:", public_pdf_result["warnings"], file=sys.stderr)
    else:
        print(f"Status: Error - {public_pdf_result.get('error_message', 'Unknown error')}", file=sys.stderr)
        if public_pdf_result.get("warnings"):
            print("Warnings:", public_pdf_result["warnings"], file=sys.stderr)


    # --- Save all results to a JSON file ---
    output_filename = "pdf_extractor_test_results.json"
    try:
        with open(output_filename, 'w') as f:
            json.dump(results_to_save, f, indent=4, ensure_ascii=False)
        print(f"\nSuccessfully exported PDF extraction test results to: {output_filename}", file=sys.stderr)
    except Exception as e:
        print(f"Error writing PDF extraction test results to JSON file {output_filename}: {e}", file=sys.stderr)

    # --- OPTIONAL: Test with an invalid URL ---
    # print("\n--- Testing Invalid URL ---")
    # invalid_url_result = process_pdf_source("htp://invalid-url", attempt_llm_structuring=False)
    # print("\n--- Invalid URL Processing Result ---")
    # print(json.dumps(invalid_url_result, indent=2))

    # --- OPTIONAL: Test with a non-PDF URL ---
    # print("\n--- Testing Non-PDF URL ---")
    # non_pdf_url_result = process_pdf_source("https://www.google.com", attempt_llm_structuring=False)
    # print("\n--- Non-PDF URL Processing Result ---")
    # print(json.dumps(non_pdf_result, indent=2)) 