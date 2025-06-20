from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

class FundingRound(BaseModel):
    """Details of a specific funding round."""
    round_name: Optional[str] = Field(None, description="Name of the funding round (e.g., Seed, Series A, Pre-seed).")
    amount_raised: Optional[str] = Field(None, description="Amount raised in this round (e.g., '$1M', '€500K').")
    date_announced: Optional[datetime] = Field(None, description="Date the funding round was announced.")
    key_investors: Optional[List[str]] = Field(None, description="List of key investors in this round.")

class CompanyProfile(BaseModel):
    """Detailed information about a company."""
    company_name: str = Field(description="Official name of the company.")
    website: Optional[HttpUrl] = Field(None, description="Official website of the company.")
    linkedin_url: Optional[HttpUrl] = Field(None, description="Company's LinkedIn profile URL.")
    description: Optional[str] = Field(None, description="A brief description of what the company does.")
    industry: Optional[str] = Field(None, description="Primary industry the company operates in.")
    sub_industry: Optional[str] = Field(None, description="More specific sub-industry or niche.")
    founded_year: Optional[int] = Field(None, description="Year the company was founded.")
    
    funding_stage: Optional[str] = Field(None, description="Current funding stage (e.g., Seed, Series A, Bootstrapped).")
    total_funding_raised: Optional[str] = Field(None, description="Total funding raised to date (e.g., '$5M').")
    funding_rounds_details: Optional[List[FundingRound]] = Field(None, description="Details of specific funding rounds.")
    
    key_products_services: Optional[List[str]] = Field(None, description="List of key products or services offered.")
    business_model: Optional[str] = Field(None, description="Description of how the company makes money.")
    target_customer: Optional[str] = Field(None, description="Description of the company's ideal customer profile.")
    
    key_metrics: Optional[Dict[str, Any]] = Field(None, description="Key performance indicators (e.g., {'ARR': '1M USD', 'User Growth': '20% MoM'}).")
    team_size: Optional[str] = Field(None, description="Approximate number of employees (e.g., '1-10', '11-50', '50+').")
    location_hq: Optional[str] = Field(None, description="Location of the company's headquarters.")
    other_locations: Optional[List[str]] = Field(None, description="List of other significant office locations.")
    mission_statement: Optional[str] = Field(None, description="The company's stated mission.")

class FounderCriteriaAssessment(BaseModel):
    """Assessment of a founder against specific investment criteria."""
    focus_industry_fit: Optional[bool] = Field(None, description="Is the founder's venture in our focus industry (Media, Brand, Tech, Creator Economy)?")
    mission_alignment: Optional[bool] = Field(None, description="Does the founder's venture avoid harm to humanity and brand reputational risk? (NOT about requiring positive impact, but avoiding harmful industries)")
    exciting_solution_to_problem: Optional[bool] = Field(None, description="Is the founder's idea a good solution to a real and meaningful problem that would excite S. Stephen Bartlett?")
    founded_something_relevant_before: Optional[bool] = Field(None, description="Has the founder founded something impressive and relevant to their current venture before?")
    impressive_relevant_past_experience: Optional[bool] = Field(None, description="Has the founder worked somewhere impressive and relevant in a position that'll make them more likely to be a good Founder?")
    exceptionally_smart_or_strategic: Optional[bool] = Field(None, description="Does the founder have evidence of being super smart or strategic from education, job, or thought leadership?")
    assessment_summary: Optional[str] = Field(None, description="Brief summary of the founder's alignment with these criteria.")

class PreviousCompany(BaseModel):
    """Information about a founder's previous company or role."""
    company_name: str = Field(description="Name of the previous company.")
    role: Optional[str] = Field(None, description="Role held at the company.")
    duration: Optional[str] = Field(None, description="Duration of employment (e.g., '2 years', '2018-2020').")
    description: Optional[str] = Field(None, description="Brief description of responsibilities or achievements.")
    was_founder: Optional[bool] = Field(None, description="Was the individual a founder of this company?")

class EducationDetail(BaseModel):
    """Details of a founder's educational background."""
    institution: Optional[str] = Field(None, description="Name of the educational institution.")
    degree: Optional[str] = Field(None, description="Degree obtained (e.g., 'BSc Computer Science', 'MBA').")
    field_of_study: Optional[str] = Field(None, description="Field of study.")
    graduation_year: Optional[int] = Field(None, description="Year of graduation.")
    notable_achievements: Optional[str] = Field(None, description="Any notable achievements or honors.")

class FounderProfile(BaseModel):
    """Detailed information about a company founder."""
    name: str = Field(description="Full name of the founder.")
    linkedin_url: Optional[HttpUrl] = Field(None, description="URL of the founder's LinkedIn profile.")
    role_in_company: Optional[str] = Field(None, description="Founder's current role in the company (e.g., CEO, CTO, Co-founder).")
    background_summary: Optional[str] = Field(None, description="A summary of the founder's professional background and expertise.")
    previous_companies: Optional[List[PreviousCompany]] = Field(None, description="List of previous companies the founder worked at or founded.")
    education: Optional[List[EducationDetail]] = Field(None, description="List of educational qualifications.")
    key_skills_and_expertise: Optional[List[str]] = Field(None, description="List of key skills and areas of expertise.")
    public_speaking_or_content: Optional[List[Dict[str, Any]]] = Field(None, description="Links to or descriptions of public speaking, articles, or content created by the founder (e.g., {'type': 'article', 'title': 'Future of X', 'url': '...'})")
    investment_criteria_assessment: Optional[FounderCriteriaAssessment] = Field(None, description="Assessment against Flight Story's founder-specific investment criteria.")

class CompetitorInfo(BaseModel):
    """Information about a competitor."""
    name: str = Field(description="Name of the competitor.")
    website: Optional[HttpUrl] = Field(None, description="Competitor's website.")
    strengths: Optional[List[str]] = Field(None, description="Key strengths of the competitor.")
    weaknesses: Optional[List[str]] = Field(None, description="Key weaknesses of the competitor.")
    market_share_estimate: Optional[str] = Field(None, description="Estimated market share, if known.")
    funding_raised: Optional[str] = Field(None, description="Funding raised by the competitor, if known.")

class MarketAnalysis(BaseModel):
    """Comprehensive analysis of a specific market or industry."""
    jurisdiction: Optional[str] = Field(None, description="The specific geographic region or country this analysis pertains to (e.g., 'USA', 'Europe', 'Global'). If not specified, analysis may be global or based on best available data.")
    industry_overview: Optional[str] = Field(None, description="General overview of the industry, its definition, and scope.")
    target_market_segment: Optional[str] = Field(None, description="Specific customer segments the market targets.")
    market_size_tam: Optional[str] = Field(None, description="Total Addressable Market (TAM) estimate (e.g., '$10B').")
    market_size_sam: Optional[str] = Field(None, description="Serviceable Addressable Market (SAM) estimate (e.g., '$1B').")
    market_size_som: Optional[str] = Field(None, description="Serviceable Obtainable Market (SOM) estimate (e.g., '$100M').")
    market_growth_rate_cagr: Optional[str] = Field(None, description="Compound Annual Growth Rate (CAGR) of the market (e.g., '15%').")
    key_market_trends: Optional[List[str]] = Field(None, description="Significant trends affecting the market.")
    competitor_landscape_summary: Optional[str] = Field(None, description="Summary of the competitive landscape.")
    competitors: Optional[List[CompetitorInfo]] = Field(None, description="List of key competitors with their details.")
    company_competitive_advantages: Optional[List[str]] = Field(None, description="The company's unique competitive advantages or differentiators.")
    barriers_to_entry: Optional[List[str]] = Field(None, description="Potential barriers to entry for new players in this market.")
    market_timing_assessment: Optional[str] = Field(None, description="Assessment of the market timing (e.g., 'Favorable', 'Early', 'Mature', 'Crowded').")
    regulatory_environment: Optional[str] = Field(None, description="Overview of the regulatory landscape relevant to the industry.")

class InvestmentAssessment(BaseModel):
    """Assessment of the investment opportunity based on Flight Story criteria."""
    # Flight Story Criteria - Must-Haves: Industry, Mission, Idea
    fs_focus_industry_fit: Optional[bool] = Field(None, description="Is the company in our focus industry (Media, Brand, Tech, Creator Economy)?")
    fs_mission_alignment: Optional[bool] = Field(None, description="Does the business avoid harm to humanity and brand reputational risk? (NOT about requiring positive impact, but avoiding tobacco, harmful drugs, gambling, predatory financial products, etc.)")
    fs_exciting_solution_to_problem: Optional[bool] = Field(None, description="Do we believe, understand, and would S. Stephen Bartlett be excited that the business idea is a good solution to a real and meaningful problem?")
    
    # Flight Story Criteria - Must-Haves: Founder Potential (can be an aggregation or specific to the primary founder if multiple)
    fs_founded_something_relevant_before: Optional[bool] = Field(None, description="Have the key founders founded something impressive and relevant to their current venture before?")
    fs_impressive_relevant_past_experience: Optional[bool] = Field(None, description="Have the key founders worked somewhere impressive and relevant in a position that'll make them more likely to be a good Founder?")
    fs_exceptionally_smart_or_strategic: Optional[bool] = Field(None, description="Do the key founders have evidence of being super smart or strategic from education, job, or thought leadership?")
    
    overall_criteria_summary: Optional[str] = Field(None, description="A brief summary of how the prospect aligns with the 6 key Flight Story investment criteria.")
    
    key_risk_factors: Optional[List[str]] = Field(None, description="Identified key risks associated with this investment.")
    key_opportunities: Optional[List[str]] = Field(None, description="Identified key opportunities for this investment.")
    investment_thesis_summary: Optional[str] = Field(None, description="A concise summary of the investment thesis.")
    potential_return_profile: Optional[str] = Field(None, description="Qualitative or quantitative assessment of potential returns.")
    deal_terms_summary: Optional[str] = Field(None, description="Summary of proposed or known deal terms (if applicable).")
    recommended_next_steps: Optional[List[str]] = Field(None, description="Actionable next steps for the investment team.")

class InvestmentResearch(BaseModel):
    """Master model for storing comprehensive investment research data."""
    research_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the research document.")
    query_or_target_entity: str = Field(description="The initial query, company name, or LinkedIn URL that initiated the research.")
    research_date: datetime = Field(default_factory=lambda: datetime.utcnow().replace(microsecond=0), description="Date and time the research was compiled (UTC).")
    status: Optional[str] = Field("In Progress", description="Current status of the research (e.g., In Progress, Pending Review, Complete, Rejected).")
    primary_analyst: Optional[str] = Field(None, description="Name or ID of the primary analyst responsible for this research.")
    
    company_profile: Optional[CompanyProfile] = Field(None, description="Profile of the company being researched.")
    founder_profiles: Optional[List[FounderProfile]] = Field(None, description="Profiles of key founders.")
    market_analysis: Optional[MarketAnalysis] = Field(None, description="Analysis of the relevant market.")
    investment_assessment: Optional[InvestmentAssessment] = Field(None, description="Overall investment assessment and alignment with criteria.")
    
    overall_summary_and_recommendation: Optional[str] = Field(None, description="A high-level summary of the findings and an investment recommendation (e.g., Proceed, Hold, Decline).")
    confidence_score_overall: Optional[float] = Field(None, ge=0.0, le=1.0, description="Overall confidence in the research findings and recommendation (0.0 to 1.0).")
    
    sources_consulted: Optional[List[Dict[str, Any]]] = Field(None, description="List of sources consulted during research (e.g., {'type': 'url', 'value': '...'}, {'type': 'interview', 'person': '...'}).")
    raw_tool_outputs: Optional[Dict[str, Any]] = Field(None, description="Raw outputs from various automated tools used during research, for traceability and detailed review.")
    attachments: Optional[List[Dict[str, Any]]] = Field(None, description="Links to or metadata of attached documents like pitch decks, financial models (e.g., {'filename': 'deck.pdf', 'url': '...', 'type': 'pitch_deck'}).")
    analyst_notes: Optional[str] = Field(None, description="General notes or commentary from the analyst.")

    class Config:
        # Example of how to generate a schema
        # schema_extra = {
        #     "example": {
        #         "query_or_target_entity": "ExampleTech Inc.",
        #         # ... add more example data for other fields
        #     }
        # }
        validate_assignment = True # Ensures that fields are validated when assigned a value after model initialization.

if __name__ == "__main__":
    # Example usage:
    # Create an instance of InvestmentResearch and populate it
    # This is just to show how it can be instantiated and to help catch basic errors.
    
    example_founder_criteria = FounderCriteriaAssessment(
        focus_industry_fit=True,
        mission_alignment=True,
        exciting_solution_to_problem=True,
        founded_something_relevant_before=False,
        impressive_relevant_past_experience=True,
        exceptionally_smart_or_strategic=True,
        assessment_summary="Strong alignment on most founder criteria."
    )

    example_founder = FounderProfile(
        name="Jane Doe",
        linkedin_url="http://linkedin.com/in/janedoe",
        role_in_company="CEO & Co-founder",
        background_summary="Experienced entrepreneur with a background in SaaS.",
        previous_companies=[PreviousCompany(company_name="OldSaaS Inc.", role="Product Manager", was_founder=False)],
        education=[EducationDetail(institution="State University", degree="MBA", graduation_year=2010)],
        investment_criteria_assessment=example_founder_criteria
    )

    example_company = CompanyProfile(
        company_name="Future Solutions Ltd.",
        website="http://futuresolutions.com",
        description="AI-driven platform for sustainable energy.",
        industry="Technology",
        sub_industry="CleanTech",
        founded_year=2022,
        funding_stage="Seed",
        total_funding_raised="$1.5M",
        team_size="11-50",
        location_hq="San Francisco, CA"
    )

    example_market = MarketAnalysis(
        industry_overview="The CleanTech AI market is rapidly expanding.",
        market_size_tam="$50B",
        market_growth_rate_cagr="25%",
        market_timing_assessment="Favorable, growing adoption."
    )

    example_assessment = InvestmentAssessment(
        fs_focus_industry_fit=True,
        fs_mission_alignment=True,
        fs_exciting_solution_to_problem=True,
        fs_founded_something_relevant_before=False, # Assuming this is an aggregate or for primary founder
        fs_impressive_relevant_past_experience=True,
        fs_exceptionally_smart_or_strategic=True,
        overall_criteria_summary="Strong alignment with Flight Story criteria. Primary founder shows promise.",
        key_risk_factors=["Intense competition", "Scalability challenges"],
        key_opportunities=["First-mover advantage in niche", "Strong IP"],
        investment_thesis_summary="Investing in a disruptive AI for CleanTech with a capable (though first-time founder) team.",
        recommended_next_steps=["Schedule follow-up call with founders", "Deeper dive into financial projections"]
    )
    
    research_doc = InvestmentResearch(
        query_or_target_entity="Future Solutions Ltd.",
        primary_analyst="AI Assistant",
        company_profile=example_company,
        founder_profiles=[example_founder],
        market_analysis=example_market,
        investment_assessment=example_assessment,
        overall_summary_and_recommendation="Proceed with caution. Strong tech, but founder is first-time. Market is hot.",
        confidence_score_overall=0.75,
        sources_consulted=[{"type": "url", "value": "http://techcrunch.com/article"}, {"type": "internal_db", "id": "xyz"}],
        status="Pending Review"
    )

    print("InvestmentResearch JSON Schema:")
    print(InvestmentResearch.model_json_schema(indent=2))
    print("\nExample InvestmentResearch instance (JSON):")
    print(research_doc.model_dump_json(indent=2))

    # You can also validate data when loading
    # json_data_from_somewhere = research_doc.model_dump_json()
    # loaded_research_doc = InvestmentResearch.model_validate_json(json_data_from_somewhere)
    # assert loaded_research_doc.company_profile.company_name == "Future Solutions Ltd."
    # print("\nSuccessfully loaded and validated from JSON.") 