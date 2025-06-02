import os
import json
import re
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Import models from your project structure
from ai_agents.models.investment_research import (
    InvestmentAssessment,
    CompanyProfile,
    FounderProfile,
    MarketAnalysis,
    InvestmentResearch
)

# Load .env file from the project root
load_dotenv()

# --- Output Parser ---
investment_assessment_parser = PydanticOutputParser(pydantic_object=InvestmentAssessment)

# --- Prompt Template for Decision Support ---
DECISION_SUPPORT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert investment analyst for Flight Story, a venture capital fund with very specific investment criteria.\n"
            "Your task is to synthesize research data about a company, its founders, and market to make an investment recommendation.\n\n"
            "FLIGHT STORY'S 6 MUST-HAVE INVESTMENT CRITERIA:\n"
            "Industry/Mission/Idea:\n"
            "1. Focus Industry Fit: Must be in Media, Brand, Tech, or Creator Economy\n"
            "2. Mission Alignment: Must align with healthier, happier humanity; impact-driven\n"
            "3. Exciting Solution: Must solve a real, meaningful problem that would excite Stephen Bartlett\n\n"
            "Founder Potential:\n"
            "4. Founded Something Before: Founders should have founded something impressive and relevant before\n"
            "5. Past Experience: Should have worked at impressive/relevant places (FAANG, major brands, VC-backed companies)\n"
            "6. Exceptionally Smart/Strategic: Evidence of being super smart/strategic from education, job, or thought leadership\n\n"
            "SCORING GUIDELINES:\n"
            "- Each criterion should be scored as true (YES), false (NO), or null (UNCLEAR/INSUFFICIENT DATA)\n"
            "- Be conservative: only mark 'true' if there's clear evidence\n"
            "- Consider all founders collectively for criteria 4-6\n\n"
            "RECOMMENDATION THRESHOLDS:\n"
            "- PASS (Proceed to partner meeting): 5-6 out of 6 criteria met\n"
            "- EXPLORE (Gather more information): 3-4 out of 6 criteria met\n"
            "- DECLINE (Pass on investment): Less than 3 criteria met\n\n"
            "ANALYSIS APPROACH:\n"
            "1. Carefully review the company profile, founder profiles, and market analysis\n"
            "2. Evaluate each of the 6 criteria based on the evidence\n"
            "3. Identify key risks and opportunities\n"
            "4. Formulate an investment thesis if applicable\n"
            "5. Recommend clear next steps\n\n"
            "Your output MUST be a valid JSON object conforming to the InvestmentAssessment schema.\n"
            "Base your assessment ONLY on the provided data. If information is missing, note it in your analysis.\n"
            "Schema:\n{format_instructions}"
        ),
        (
            "human", 
            "Please analyze this investment opportunity:\n\n"
            "COMPANY DATA:\n{company_data}\n\n"
            "FOUNDER DATA:\n{founder_data}\n\n"
            "MARKET DATA:\n{market_data}\n\n"
            "ADDITIONAL CONTEXT:\n{additional_context}"
        ),
    ]
).partial(format_instructions=investment_assessment_parser.get_format_instructions())


def get_llm():
    """Initializes and returns the appropriate LLM based on available API keys."""
    if os.getenv("ANTHROPIC_API_KEY"):
        print("Using Anthropic Claude model for investment decisions.")
        return ChatAnthropic(
            model="claude-3-haiku-20240307",
            temperature=0.1,  # Low temperature for consistent decisions
            max_tokens=4000
        )
    elif os.getenv("OPENAI_API_KEY"):
        print("Using OpenAI GPT model for investment decisions.")
        return ChatOpenAI(
            model="gpt-3.5-turbo-0125",
            temperature=0.1
        )
    else:
        print("FATAL: Neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set. Cannot initialize LLM.")
        return None


def load_research_data(
    company_file: Optional[str] = None,
    founder_files: Optional[List[str]] = None,
    market_file: Optional[str] = None
) -> Tuple[Optional[CompanyProfile], List[FounderProfile], Optional[MarketAnalysis]]:
    """
    Load research data from JSON files.
    Returns tuple of (company_profile, founder_profiles, market_analysis)
    """
    company_profile = None
    founder_profiles = []
    market_analysis = None
    
    # Load company data
    if company_file and os.path.exists(company_file):
        try:
            with open(company_file, 'r') as f:
                company_data = json.load(f)
            company_profile = CompanyProfile.model_validate(company_data)
            print(f"Loaded company profile: {company_profile.company_name}")
        except Exception as e:
            print(f"Error loading company file {company_file}: {e}")
    
    # Load founder data
    if founder_files:
        for founder_file in founder_files:
            if os.path.exists(founder_file):
                try:
                    with open(founder_file, 'r') as f:
                        founder_data = json.load(f)
                    founder_profile = FounderProfile.model_validate(founder_data)
                    founder_profiles.append(founder_profile)
                    print(f"Loaded founder profile: {founder_profile.name}")
                except Exception as e:
                    print(f"Error loading founder file {founder_file}: {e}")
    
    # Load market data
    if market_file and os.path.exists(market_file):
        try:
            with open(market_file, 'r') as f:
                market_data = json.load(f)
            market_analysis = MarketAnalysis.model_validate(market_data)
            print(f"Loaded market analysis for: {market_analysis.industry_overview[:50]}...")
        except Exception as e:
            print(f"Error loading market file {market_file}: {e}")
    
    return company_profile, founder_profiles, market_analysis


def calculate_recommendation(assessment: InvestmentAssessment) -> str:
    """
    Calculate the investment recommendation based on criteria scores.
    Returns: 'PASS', 'EXPLORE', or 'DECLINE'
    """
    criteria_scores = [
        assessment.fs_focus_industry_fit,
        assessment.fs_mission_alignment,
        assessment.fs_exciting_solution_to_problem,
        assessment.fs_founded_something_relevant_before,
        assessment.fs_impressive_relevant_past_experience,
        assessment.fs_exceptionally_smart_or_strategic
    ]
    
    # Count only True values (not None/null)
    true_count = sum(1 for score in criteria_scores if score is True)
    
    if true_count >= 5:
        return "PASS"
    elif true_count >= 3:
        return "EXPLORE"
    else:
        return "DECLINE"


def run_decision_support_cli(
    company_file: Optional[str] = None,
    founder_files: Optional[List[str]] = None,
    market_file: Optional[str] = None,
    additional_context: Optional[Dict[str, Any]] = None
) -> Optional[InvestmentAssessment]:
    """
    Run the decision support agent to analyze an investment opportunity.
    
    Args:
        company_file: Path to company research JSON
        founder_files: List of paths to founder research JSONs
        market_file: Path to market analysis JSON
        additional_context: Dict with additional data (e.g., from email, HubSpot survey)
    
    Returns:
        InvestmentAssessment object if successful, None otherwise
    """
    print("Starting investment decision analysis...")
    
    # Load research data
    company_profile, founder_profiles, market_analysis = load_research_data(
        company_file, founder_files, market_file
    )
    
    if not company_profile:
        print("ERROR: Company profile is required for investment decision.")
        return None
    
    # Initialize LLM
    llm = get_llm()
    if not llm:
        return None
    
    # Prepare data for prompt
    company_data_str = company_profile.model_dump_json(indent=2) if company_profile else "No company data available"
    founder_data_str = json.dumps([f.model_dump(mode='json') for f in founder_profiles], indent=2) if founder_profiles else "No founder data available"
    market_data_str = market_analysis.model_dump_json(indent=2) if market_analysis else "No market data available"
    additional_context_str = json.dumps(additional_context, indent=2) if additional_context else "No additional context provided"
    
    # Create the chain
    # The parser will be applied *after* we extract the clean JSON string
    chain = DECISION_SUPPORT_PROMPT_TEMPLATE | llm 
    
    try:
        print("\nAnalyzing investment opportunity against Flight Story criteria...")
        # Get the raw string output from the LLM
        raw_llm_output_str = chain.invoke({
            "company_data": company_data_str,
            "founder_data": founder_data_str,
            "market_data": market_data_str,
            "additional_context": additional_context_str
        }).content # .content for AIMessage, or handle if it's a string directly

        print(f"Raw LLM output for decision support: {raw_llm_output_str[:500]}...")

        # Extract the JSON block
        json_to_parse = None
        json_match = re.search(r"\{[\s\S]*\}", raw_llm_output_str)
        if json_match:
            json_to_parse = json_match.group(0).strip()
            print(f"Extracted JSON block for InvestmentAssessment: {json_to_parse[:300]}...")
        else:
            # Fallback if regex doesn't find a clear JSON object.
            temp_str = raw_llm_output_str.strip()
            if temp_str.startswith("```json"):
                temp_str = temp_str[len("```json"):].strip()
            if temp_str.endswith("```"):
                temp_str = temp_str[:-len("```")].strip()
            
            if temp_str.startswith("{") and temp_str.endswith("}"):
                json_to_parse = temp_str
                print(f"Warning: No clear JSON block found by primary regex, attempting to parse potentially cleaned string: {json_to_parse[:300]}...")
            else:
                print(f"Error: Could not extract a valid JSON object from LLM response: {raw_llm_output_str[:500]}...")
                raise ValueError("Failed to extract JSON from LLM response for InvestmentAssessment.")

        # Now parse the extracted JSON string using the PydanticOutputParser
        assessment = investment_assessment_parser.parse(json_to_parse)
        
        # Calculate recommendation
        recommendation = calculate_recommendation(assessment)
        
        # Create full investment research document
        investment_research = InvestmentResearch(
            query_or_target_entity=company_profile.company_name,
            primary_analyst="AI Investment Analyst",
            company_profile=company_profile,
            founder_profiles=founder_profiles,
            market_analysis=market_analysis,
            investment_assessment=assessment,
            overall_summary_and_recommendation=recommendation,
            confidence_score_overall=0.8 if recommendation == "PASS" else 0.6 if recommendation == "EXPLORE" else 0.4,
            status="Complete",
            sources_consulted=[
                {"type": "file", "value": company_file} if company_file else None,
                {"type": "file", "value": str(founder_files)} if founder_files else None,
                {"type": "file", "value": market_file} if market_file else None,
            ]
        )
        
        # Save the complete analysis
        safe_company_name = "".join(c if c.isalnum() else "_" for c in company_profile.company_name)[:30]
        output_filename = f"investment_decision_{safe_company_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_filename, 'w') as f:
            f.write(investment_research.model_dump_json(indent=2))
        
        print(f"\n{'='*60}")
        print(f"INVESTMENT DECISION: {recommendation}")
        print(f"{'='*60}")
        print(f"Company: {company_profile.company_name}")
        print(f"Industry: {company_profile.industry}")
        print(f"Funding Stage: {company_profile.funding_stage}")
        print(f"Total Raised: {company_profile.total_funding_raised}")
        
        print(f"\nFlight Story Criteria Assessment:")
        print(f"1. Focus Industry Fit: {'✓' if assessment.fs_focus_industry_fit else '✗' if assessment.fs_focus_industry_fit is False else '?'}")
        print(f"2. Mission Alignment: {'✓' if assessment.fs_mission_alignment else '✗' if assessment.fs_mission_alignment is False else '?'}")
        print(f"3. Exciting Solution: {'✓' if assessment.fs_exciting_solution_to_problem else '✗' if assessment.fs_exciting_solution_to_problem is False else '?'}")
        print(f"4. Founded Before: {'✓' if assessment.fs_founded_something_relevant_before else '✗' if assessment.fs_founded_something_relevant_before is False else '?'}")
        print(f"5. Past Experience: {'✓' if assessment.fs_impressive_relevant_past_experience else '✗' if assessment.fs_impressive_relevant_past_experience is False else '?'}")
        print(f"6. Smart/Strategic: {'✓' if assessment.fs_exceptionally_smart_or_strategic else '✗' if assessment.fs_exceptionally_smart_or_strategic is False else '?'}")
        
        if assessment.investment_thesis_summary:
            print(f"\nInvestment Thesis: {assessment.investment_thesis_summary}")
        
        if assessment.key_risk_factors:
            print(f"\nKey Risks:")
            for risk in assessment.key_risk_factors[:3]:
                print(f"  - {risk}")
        
        if assessment.recommended_next_steps:
            print(f"\nRecommended Next Steps:")
            for step in assessment.recommended_next_steps:
                print(f"  - {step}")
        
        print(f"\nFull analysis saved to: {output_filename}")
        
        return assessment
        
    except Exception as e:
        print(f"Error during investment decision analysis: {e}")
        return None


if __name__ == "__main__":
    print("Testing decision_support_agent.py...")
    
    # Example usage with your test files
    company_file = input("Enter path to company research JSON (or press Enter to skip): ").strip()
    founder_files_input = input("Enter path(s) to founder research JSON(s), comma-separated (or press Enter to skip): ").strip()
    market_file = input("Enter path to market analysis JSON (or press Enter to skip): ").strip()
    
    # Parse founder files
    founder_files = [f.strip() for f in founder_files_input.split(',')] if founder_files_input else []
    
    # Example additional context (would come from email/HubSpot in real usage)
    additional_context = {
        "source": "direct_test",
        "notes": "Manual test run",
        # In production, this would include:
        # "email_summary": "...",
        # "hubspot_survey_data": {...},
        # "meeting_notes": "...",
        # "pitch_deck_url": "..."
    }
    
    if company_file:
        assessment = run_decision_support_cli(
            company_file=company_file if company_file else None,
            founder_files=founder_files if founder_files else None,
            market_file=market_file if market_file else None,
            additional_context=additional_context
        )
        
        if assessment:
            print("\nDecision support analysis completed successfully!")
        else:
            print("\nDecision support analysis failed.")
    else:
        print("No company file provided. Cannot proceed without company data.")