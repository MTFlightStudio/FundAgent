import streamlit as st
import json
import pandas as pd
from typing import Dict, Any, List

def display_metric_card(title: str, value: Any, help_text: str = None):
    """Display a metric in a card format"""
    with st.container():
        st.metric(label=title, value=value, help=help_text)

def display_company_profile(company_data: Dict[str, Any]):
    """Display company profile information"""
    if not company_data:
        st.info("No company data available")
        return
    
    # Handle both direct company profile and wrapped response
    if 'company_profile' in company_data:
        profile = company_data['company_profile']
    else:
        profile = company_data
    
    # Company Overview
    st.subheader("🏢 Company Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        display_metric_card("Company Name", profile.get('company_name', 'N/A'))
        display_metric_card("Founded", profile.get('founded_year', 'N/A'))
    
    with col2:
        display_metric_card("Industry", profile.get('industry', 'N/A'))
        display_metric_card("Sub-Industry", profile.get('sub_industry', 'N/A'))
    
    with col3:
        display_metric_card("Team Size", profile.get('team_size', 'N/A'))
        display_metric_card("HQ Location", profile.get('location_hq', 'N/A'))
    
    # Description
    if profile.get('description'):
        st.subheader("📝 Description")
        st.write(profile['description'])
    
    # Funding Information
    if profile.get('total_funding_raised') or profile.get('funding_stage'):
        st.subheader("💰 Funding Information")
        col1, col2 = st.columns(2)
        with col1:
            display_metric_card("Total Funding", profile.get('total_funding_raised', 'N/A'))
        with col2:
            display_metric_card("Current Stage", profile.get('funding_stage', 'N/A'))
        
        # Funding rounds details
        if profile.get('funding_rounds_details'):
            st.write("**Funding History:**")
            for round_data in profile['funding_rounds_details']:
                st.write(f"- {round_data.get('round_name', 'N/A')}: {round_data.get('amount_raised', 'N/A')}")
    
    # Products & Services
    if profile.get('key_products_services'):
        st.subheader("🛍️ Products & Services")
        for product in profile['key_products_services']:
            st.write(f"- {product}")
    
    # Business Model
    if profile.get('business_model'):
        st.subheader("💼 Business Model")
        st.write(profile['business_model'])
    
    # Links
    st.subheader("🔗 Links")
    col1, col2 = st.columns(2)
    with col1:
        if profile.get('website'):
            st.markdown(f"[Company Website]({profile['website']})")
    with col2:
        if profile.get('linkedin_url'):
            st.markdown(f"[LinkedIn Profile]({profile['linkedin_url']})")

def display_founder_profiles(founders_data: Any):
    """Display founder profiles"""
    # Handle different data structures
    if isinstance(founders_data, dict):
        # Check if it's a wrapped response
        if 'founders' in founders_data:
            founders_list = founders_data['founders']
        else:
            # Single founder dict
            founders_list = [founders_data]
    elif isinstance(founders_data, list):
        founders_list = founders_data
    else:
        st.info("No founder data available")
        return
    
    if not founders_list:
        st.info("No founder profiles available")
        return
    
    for i, founder in enumerate(founders_list):
        with st.expander(f"👤 {founder.get('name', f'Founder {i+1}')}"):
            # Basic Info
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Role:** {founder.get('role_in_company', 'N/A')}")
                if founder.get('linkedin_url'):
                    st.markdown(f"[LinkedIn Profile]({founder['linkedin_url']})")
            
            # Background Summary
            if founder.get('background_summary'):
                st.write("**Background:**")
                st.write(founder['background_summary'])
            
            # Previous Experience
            if founder.get('previous_companies'):
                st.write("**Previous Experience:**")
                for company in founder['previous_companies']:
                    was_founder = " (Founder)" if company.get('was_founder') else ""
                    st.write(f"- {company.get('company_name', 'N/A')} - {company.get('role', 'N/A')}{was_founder}")
            
            # Education
            if founder.get('education'):
                st.write("**Education:**")
                for edu in founder['education']:
                    st.write(f"- {edu.get('institution', 'N/A')} - {edu.get('degree', 'N/A')}")
            
            # Investment Criteria Assessment
            if founder.get('investment_criteria_assessment'):
                assessment = founder['investment_criteria_assessment']
                st.write("**Investment Criteria Assessment:**")
                
                criteria_cols = st.columns(2)
                with criteria_cols[0]:
                    st.write(f"✓ Focus Industry Fit: {'Yes' if assessment.get('focus_industry_fit') else 'No'}")
                    st.write(f"✓ Mission Alignment: {'Yes' if assessment.get('mission_alignment') else 'No'}")
                    st.write(f"✓ Exciting Solution: {'Yes' if assessment.get('exciting_solution_to_problem') else 'No'}")
                
                with criteria_cols[1]:
                    st.write(f"✓ Founded Before: {'Yes' if assessment.get('founded_something_relevant_before') else 'No'}")
                    st.write(f"✓ Impressive Experience: {'Yes' if assessment.get('impressive_relevant_past_experience') else 'No'}")
                    st.write(f"✓ Exceptionally Smart: {'Yes' if assessment.get('exceptionally_smart_or_strategic') else 'No'}")
                
                if assessment.get('assessment_summary'):
                    st.write(f"**Summary:** {assessment['assessment_summary']}")

def display_market_analysis(market_data: Dict[str, Any]):
    """Display market analysis information"""
    if not market_data:
        st.info("No market analysis available")
        return
    
    # Handle wrapped response
    if 'market_analysis' in market_data:
        analysis = market_data['market_analysis']
    else:
        analysis = market_data
    
    # Market Overview
    if analysis.get('industry_overview'):
        st.subheader("🌍 Market Overview")
        st.write(analysis['industry_overview'])
    
    # Market Size Metrics
    st.subheader("📊 Market Size")
    col1, col2, col3 = st.columns(3)
    with col1:
        display_metric_card("TAM", analysis.get('market_size_tam', 'N/A'))
    with col2:
        display_metric_card("SAM", analysis.get('market_size_sam', 'N/A'))
    with col3:
        display_metric_card("SOM", analysis.get('market_size_som', 'N/A'))
    
    # Growth and Timing
    col1, col2 = st.columns(2)
    with col1:
        display_metric_card("Growth Rate (CAGR)", analysis.get('market_growth_rate_cagr', 'N/A'))
    with col2:
        display_metric_card("Market Timing", analysis.get('market_timing_assessment', 'N/A'))
    
    # Key Trends
    if analysis.get('key_market_trends'):
        st.subheader("📈 Key Market Trends")
        for trend in analysis['key_market_trends']:
            st.write(f"- {trend}")
    
    # Competitive Landscape
    if analysis.get('competitors'):
        st.subheader("🏆 Competitive Landscape")
        for competitor in analysis['competitors']:
            with st.expander(competitor.get('name', 'Competitor')):
                if competitor.get('website'):
                    st.markdown(f"[Website]({competitor['website']})")
                if competitor.get('funding_raised'):
                    st.write(f"**Funding:** {competitor['funding_raised']}")
                if competitor.get('strengths'):
                    st.write("**Strengths:**")
                    for strength in competitor['strengths']:
                        st.write(f"- {strength}")
    
    # Barriers to Entry
    if analysis.get('barriers_to_entry'):
        st.subheader("🚧 Barriers to Entry")
        for barrier in analysis['barriers_to_entry']:
            st.write(f"- {barrier}")

def display_investment_decision(decision_data: Dict[str, Any]):
    """Display investment decision and assessment"""
    if not decision_data:
        st.info("No investment decision available")
        return
    
    # Handle wrapped response
    if 'investment_research' in decision_data:
        research = decision_data['investment_research']
    elif 'investment_assessment' in decision_data:
        # Direct assessment data
        research = decision_data
    else:
        research = decision_data
    
    # Overall Recommendation
    recommendation = research.get('overall_summary_and_recommendation', 'N/A')
    confidence = research.get('confidence_score_overall', 0)
    
    # Color code based on recommendation
    if 'PASS' in str(recommendation).upper():
        color = 'green'
        emoji = '✅'
    elif 'EXPLORE' in str(recommendation).upper():
        color = 'orange'
        emoji = '🔍'
    else:
        color = 'red'
        emoji = '❌'
    
    st.markdown(f"## {emoji} Investment Recommendation: **:{color}[{recommendation}]**")
    st.progress(confidence, text=f"Confidence Score: {confidence * 100:.0f}%")
    
    # Investment Assessment
    assessment = research.get('investment_assessment', {})
    if assessment:
        st.subheader("🎯 Flight Story Criteria Assessment")
        
        criteria_data = {
            'Criteria': [
                'Focus Industry Fit',
                'Mission Alignment',
                'Exciting Solution',
                'Founded Something Before',
                'Impressive Experience',
                'Exceptionally Smart'
            ],
            'Met': [
                '✅' if assessment.get('fs_focus_industry_fit') else '❌',
                '✅' if assessment.get('fs_mission_alignment') else '❌',
                '✅' if assessment.get('fs_exciting_solution_to_problem') else '❌',
                '✅' if assessment.get('fs_founded_something_relevant_before') else '❌',
                '✅' if assessment.get('fs_impressive_relevant_past_experience') else '❌',
                '✅' if assessment.get('fs_exceptionally_smart_or_strategic') else '❌'
            ]
        }
        
        df = pd.DataFrame(criteria_data)
        st.dataframe(df, hide_index=True, use_container_width=True)
        
        # Count criteria met
        criteria_met = sum([
            1 for key in ['fs_focus_industry_fit', 'fs_mission_alignment', 
                         'fs_exciting_solution_to_problem', 'fs_founded_something_relevant_before',
                         'fs_impressive_relevant_past_experience', 'fs_exceptionally_smart_or_strategic']
            if assessment.get(key) is True
        ])
        
        st.metric("Criteria Met", f"{criteria_met}/6")
        
        # Summary
        if assessment.get('overall_criteria_summary'):
            st.write("**Summary:**")
            st.write(assessment['overall_criteria_summary'])
        
        # Risks and Opportunities
        col1, col2 = st.columns(2)
        
        with col1:
            if assessment.get('key_risk_factors'):
                st.write("**🚨 Key Risks:**")
                for risk in assessment['key_risk_factors']:
                    st.write(f"- {risk}")
        
        with col2:
            if assessment.get('key_opportunities'):
                st.write("**💡 Key Opportunities:**")
                for opp in assessment['key_opportunities']:
                    st.write(f"- {opp}")
        
        # Investment Thesis
        if assessment.get('investment_thesis_summary'):
            st.subheader("📋 Investment Thesis")
            st.write(assessment['investment_thesis_summary'])
        
        # Next Steps
        if assessment.get('recommended_next_steps'):
            st.subheader("👉 Recommended Next Steps")
            for step in assessment['recommended_next_steps']:
                st.write(f"- {step}")

def display_results(results: Dict[str, Any]):
    """Main function to display all results"""
    if not results:
        st.info("No results to display. Run an analysis first.")
        return
    
    # Create tabs for different result types
    available_tabs = []
    tab_contents = {}
    
    if 'company_research' in results:
        available_tabs.append("🏢 Company")
        tab_contents["🏢 Company"] = lambda: display_company_profile(results['company_research'])
    
    if 'founder_research' in results:
        available_tabs.append("👤 Founders")
        tab_contents["👤 Founders"] = lambda: display_founder_profiles(results['founder_research'])
    
    if 'market_research' in results:
        available_tabs.append("🌍 Market")
        tab_contents["🌍 Market"] = lambda: display_market_analysis(results['market_research'])
    
    if 'decision_support' in results:
        available_tabs.append("📊 Decision")
        tab_contents["📊 Decision"] = lambda: display_investment_decision(results['decision_support'])
    
    if available_tabs:
        tabs = st.tabs(available_tabs)
        for i, tab_name in enumerate(available_tabs):
            with tabs[i]:
                tab_content = tab_contents.get(tab_name)
                if tab_content:
                    tab_content()
    
    # Export functionality
    with st.expander("📥 Export Results"):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Download as JSON"):
                json_str = json.dumps(results, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name="investment_research.json",
                    mime="application/json"
                )
        with col2:
            if st.button("Copy to Clipboard"):
                st.code(json.dumps(results, indent=2), language='json')

# Example usage in streamlit_app.py:
"""
from ai_agents.ui.results_display import display_results

# In your main app:
if st.session_state.results:
    display_results(st.session_state.results)
""" 