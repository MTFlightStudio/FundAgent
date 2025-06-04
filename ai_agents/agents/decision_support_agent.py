import os
import json
import re
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from dotenv import load_dotenv
import argparse
import sys

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
    company_data_dict: Dict[str, Any],
    founder_data_list: List[Dict[str, Any]],
    market_data_dict: Optional[Dict[str, Any]],
    additional_context: Dict[str, Any]
) -> Optional[InvestmentResearch]:
    """
    Runs the decision support analysis using provided data dictionaries.
    """
    llm = get_llm()
    if not llm:
        print("LLM could not be initialized. Aborting decision support.", file=sys.stderr)
        return None

    try:
        # Validate and parse input data using Pydantic models
        try:
            company_profile = CompanyProfile(**company_data_dict)
        except Exception as e:
            print(f"Error parsing company data: {e}. Data: {company_data_dict}", file=sys.stderr)
            return None

        parsed_founder_profiles = []
        for i, fd in enumerate(founder_data_list):
            try:
                parsed_founder_profiles.append(FounderProfile(**fd))
            except Exception as e:
                print(f"Error parsing founder data for founder {i+1}: {e}. Data: {fd}", file=sys.stderr)
                # Continue if some founder profiles are invalid, or decide to fail
        
        if not parsed_founder_profiles and founder_data_list: # if input was given but all failed
             print("No valid founder profiles could be parsed.", file=sys.stderr)
             # return None # Or proceed without founder data if desired

        parsed_market_analysis = None
        if market_data_dict:
            try:
                parsed_market_analysis = MarketAnalysis(**market_data_dict)
            except Exception as e:
                print(f"Error parsing market data: {e}. Data: {market_data_dict}", file=sys.stderr)
                # return None # Or proceed without market data

        # Prepare data for the LLM prompt
        company_data_str = company_profile.model_dump_json(indent=2)
        founder_data_str = json.dumps([fp.model_dump(mode='json') for fp in parsed_founder_profiles], indent=2)
        market_data_str = parsed_market_analysis.model_dump_json(indent=2) if parsed_market_analysis else "N/A"
        additional_context_str = json.dumps(additional_context, indent=2)

        chain = DECISION_SUPPORT_PROMPT_TEMPLATE | llm
        raw_llm_output_str = chain.invoke({
            "company_data": company_data_str,
            "founder_data": founder_data_str,
            "market_data": market_data_str,
            "additional_context": additional_context_str,
        }).content
        
        # print(f"Raw LLM Output for Decision Support:\n{raw_llm_output_str[:1000]}...", file=sys.stderr) # For debugging

        # Extract JSON from LLM response
        json_match = re.search(r"```json\s*([\s\S]*?)\s*```", raw_llm_output_str, re.DOTALL)
        json_to_parse = ""
        if json_match:
            json_to_parse = json_match.group(1).strip()
            print("Successfully extracted JSON using markdown code block regex.", file=sys.stderr)
        else:
            # Fallback: try to find the first '{' and last '}'
            print("Markdown JSON block not found. Attempting fallback extraction.", file=sys.stderr)
            try:
                first_brace = raw_llm_output_str.index('{')
                last_brace = raw_llm_output_str.rindex('}')
                if last_brace > first_brace:
                    json_to_parse = raw_llm_output_str[first_brace : last_brace+1].strip()
                    print(f"Extracted potential JSON using brace finding: {json_to_parse[:100]}...", file=sys.stderr)
                else:
                    # This case should be rare if there's any valid JSON
                    print(f"Error: Last brace found before or at first brace. Cannot extract JSON.", file=sys.stderr)
                    raise ValueError("Failed to extract JSON: Invalid brace positions.")
            except ValueError: # Handles .index or .rindex not finding the char
                print(f"Error: Could not find opening or closing braces for JSON extraction in LLM response: {raw_llm_output_str[:500]}...", file=sys.stderr)
                raise ValueError("Failed to extract JSON from LLM response for InvestmentAssessment (braces not found).")

        if not json_to_parse:
            print(f"Error: JSON string to parse is empty after extraction attempts. Raw output: {raw_llm_output_str[:500]}...", file=sys.stderr)
            raise ValueError("Failed to extract JSON: Resulting string is empty.")

        assessment = investment_assessment_parser.parse(json_to_parse)
        recommendation = calculate_recommendation(assessment)

        investment_research = InvestmentResearch(
            query_or_target_entity=company_profile.company_name,
            primary_analyst="AI Investment Analyst (Decision Support Agent)",
            company_profile=company_profile,
            founder_profiles=parsed_founder_profiles,
            market_analysis=parsed_market_analysis,
            investment_assessment=assessment,
            overall_summary_and_recommendation=recommendation,
            confidence_score_overall=0.8 if recommendation == "PASS" else 0.6 if recommendation == "EXPLORE" else 0.4, # Simplified
            status="Complete",
            sources_consulted=[ # Simplified for agent context
                {"type": "structured_input", "source": "company_profile_data"},
                {"type": "structured_input", "source": "founder_profiles_data"},
                {"type": "structured_input", "source": "market_analysis_data"},
                {"type": "structured_input", "source": "additional_context"}
            ],
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Save the complete analysis to a file (optional, can be done by orchestrator too)
        safe_company_name = "".join(c if c.isalnum() else "_" for c in company_profile.company_name)[:30]
        output_filename = f"investment_decision_{safe_company_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(output_filename, 'w') as f:
                f.write(investment_research.model_dump_json(indent=2))
            print(f"Decision support analysis saved to: {output_filename}", file=sys.stderr)
        except Exception as e:
            print(f"Error saving decision support analysis to file: {e}", file=sys.stderr)
        
        return investment_research # Return the full research object

    except Exception as e:
        print(f"Error during investment decision analysis: {e}", file=sys.stderr)
        return None


def generate_investment_recommendation(
    founder_profile_data: Dict[str, Any],
    company_profile_data: Dict[str, Any],
    market_analysis_data: Dict[str, Any]
) -> Dict[str, Any]: # Or your Pydantic model for the decision
    """
    Analyzes the provided data and generates an investment recommendation.
    Replace this with your actual LLM call or decision logic.
    """
    print("Decision support agent: Analyzing provided data...", file=sys.stderr)
    
    recommendation = "Undecided"
    confidence = 0.0
    reasoning = "Initial analysis pending full implementation."

    # Example: Basic checks (you'll have much more sophisticated logic)
    if founder_profile_data.get("investment_criteria_assessment", {}).get("mission_alignment") is True:
        reasoning += " Founder mission alignment is positive."
        confidence += 0.2
    
    if company_profile_data.get("funding_stage") == "Seed":
        reasoning += " Company is at Seed stage."
        confidence += 0.1

    if market_analysis_data.get("market_growth_rate_cagr", "").endswith("%"):
        try:
            cagr = float(market_analysis_data["market_growth_rate_cagr"][:-1])
            if cagr > 15:
                reasoning += f" Market CAGR ({cagr}%) is strong."
                confidence += 0.3
        except ValueError:
            pass

    if confidence > 0.5:
        recommendation = "Recommend Investment"
    elif confidence > 0.2:
        recommendation = "Further Review Required"
    else:
        recommendation = "Do Not Recommend Investment"

    # This should be structured according to your InvestmentDecision model if you have one
    decision_output = {
        "recommendation": recommendation,
        "confidence_score": round(confidence, 2),
        "reasoning": reasoning.strip(),
        "supporting_data_summary": {
            "founder_name": founder_profile_data.get("name"),
            "company_name": company_profile_data.get("company_name"),
            "market_sector": market_analysis_data.get("industry_overview", market_analysis_data.get("jurisdiction", "N/A"))
        }
    }
    print(f"Decision support agent: Recommendation generated: {recommendation}", file=sys.stderr)
    return decision_output


def main():
    print("Decision support agent CLI starting...", file=sys.stderr)
    parser = argparse.ArgumentParser(description="Investment Decision Support Agent")
    parser.add_argument("--founder_profile_path", type=str, required=True, help="Path to the founder profile JSON file.")
    parser.add_argument("--company_profile_path", type=str, required=True, help="Path to the company profile JSON file.")
    parser.add_argument("--market_analysis_path", type=str, required=True, help="Path to the market analysis JSON file.")
    # Alternatively, you could pass JSON strings directly:
    # parser.add_argument("--founder_profile_json", type=str, required=True, help="JSON string of the founder profile.")
    # parser.add_argument("--company_profile_json", type=str, required=True, help="JSON string of the company profile.")
    # parser.add_argument("--market_analysis_json", type=str, required=True, help="JSON string of the market analysis.")

    args = parser.parse_args()

    try:
        with open(args.founder_profile_path, 'r') as f:
            founder_data = json.load(f)
        with open(args.company_profile_path, 'r') as f:
            company_data = json.load(f)
        with open(args.market_analysis_path, 'r') as f:
            market_data = json.load(f)
        
        # If passing JSON strings directly:
        # founder_data = json.loads(args.founder_profile_json)
        # company_data = json.loads(args.company_profile_json)
        # market_data = json.loads(args.market_analysis_json)

        decision = generate_investment_recommendation(founder_data, company_data, market_data)
        
        # Print the final decision JSON to stdout
        print(json.dumps(decision, indent=None)) # indent=None for orchestrator
        print("Decision support agent: Successfully generated JSON output.", file=sys.stderr)

    except FileNotFoundError as fnf_err:
        error_output = {"status": "error", "error_message": f"File not found: {fnf_err}"}
        print(json.dumps(error_output, indent=None), file=sys.stdout) # Error to stdout
        print(f"Decision support agent: Error - {error_output['error_message']}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as jd_err:
        error_output = {"status": "error", "error_message": f"JSON decode error: {jd_err}"}
        print(json.dumps(error_output, indent=None), file=sys.stdout) # Error to stdout
        print(f"Decision support agent: Error - {error_output['error_message']}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        error_output = {"status": "error", "error_message": f"An unexpected error occurred: {str(e)}"}
        print(json.dumps(error_output, indent=None), file=sys.stdout) # Error to stdout
        print(f"Decision support agent: Critical error - {error_output['error_message']}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()