import streamlit as st
import json
import pandas as pd
from typing import Dict, Any, List
import re

def display_metric_card(title: str, value: Any, help_text: str = None):
    """Display a metric in a card format"""
    with st.container():
        st.metric(label=title, value=value, help=help_text)

def display_company_profile(company_data: Dict[str, Any]):
    """Display company profile information with enhanced HubSpot data"""
    if not company_data:
        st.info("No company data available")
        return
    
    # ---------- Helper utilities ----------
    def _format_currency(val: Any, default: str = "N/A") -> str:
        """Standardise currency values: add £ symbol if missing, abbreviate thousands+.
        Accepts strings like "35000", "$35,000", "£35k", "£10-20m", "£0 (prior funding)", etc."""
        if val is None:
            return default
        
        # Convert to string for processing
        s_val = str(val).strip()
        
        # Handle empty or zero-like values
        if not s_val or s_val.lower() in ['0', '£0', '$0', '€0']:
            return "£0"
        
        # If it's already a well-formatted currency string with ranges or descriptions, return as-is
        if any(indicator in s_val.lower() for indicator in ['-', 'range', 'raising', 'funding', 'prior', 'current']):
            # Ensure it starts with a currency symbol
            if not re.match(r'^[£$€]', s_val):
                return f"£{s_val}"
            return s_val
        
        # Handle simple numeric values
        if isinstance(val, (int, float)):
            num = float(val)
            symbol = "£"
        else:
            # Extract symbol if present
            symbol_match = re.match(r"([£$€])", s_val)
            symbol = symbol_match.group(1) if symbol_match else "£"
            
            # Try to extract just the first number for simple cases
            number_match = re.search(r"(\d+(?:\.\d+)?)", s_val)
            if number_match:
                try:
                    num = float(number_match.group(1))
                except ValueError:
                    return s_val  # Return original if we can't parse
            else:
                return s_val  # Return original if no number found
        
        # Abbreviate simple numeric values
        if num == 0:
            return f"{symbol}0"
        elif num >= 1e9:
            num_fmt = f"{num/1e9:.1f}B"
        elif num >= 1e6:
            num_fmt = f"{num/1e6:.1f}M"
        elif num >= 1e3:
            num_fmt = f"{num/1e3:.0f}K"
        else:
            num_fmt = f"{num:,.0f}"
        
        return f"{symbol}{num_fmt}"
    
    # ---------- Resolve profile dict ----------
    profile = company_data.get('company_profile', company_data)
    key_metrics = profile.get('key_metrics', {})
    
    # ------------------------------------------------
    # 🏢  COMPANY HEADER  
    # ------------------------------------------------
    st.markdown(f"# 🏢 {profile.get('company_name', 'Unknown Company')}")
    
    # Basic info in a clean single row
    industry_info = profile.get('industry', 'Unknown Industry')
    location_info = profile.get('location_hq', 'Unknown Location')
    stage_info = key_metrics.get('Business Stage', profile.get('funding_stage', 'Unknown Stage'))
    
    # Add employee count if available
    employee_count = profile.get('team_size') or profile.get('number_of_employees')
    if employee_count:
        st.markdown(f"**{industry_info}** • **{location_info}** • **{stage_info}** • **{employee_count} employees**")
    else:
        st.markdown(f"**{industry_info}** • **{location_info}** • **{stage_info}**")
    
    # ------------------------------------------------
    # 📝  DESCRIPTION & BUSINESS MODEL
    # ------------------------------------------------
    if profile.get('description'):
        st.markdown("### About")
        st.write(profile['description'])
    
    if profile.get('business_model'):
        st.markdown("### Business Model & USP")
        st.write(profile['business_model'])
    
    if profile.get('key_products_services'):
        st.markdown("### Products & Services")
        for product in profile['key_products_services']:
            st.write(f"• {product}")
    
    # ------------------------------------------------
    # 🌍  IMPACT & INNOVATION (New Section)
    # ------------------------------------------------
    impact_items = []
    
    # Check for UN SDG goals
    sdg_goals = profile.get('which__if_any__of_the_un_sdg_17_goals_does_your_business_address_')
    if sdg_goals:
        impact_items.append(("🎯 UN SDG Goals", sdg_goals))
    
    # Check for health/human impact
    health_impact = profile.get('does_your_product_contribute_to_a_healthier__happier_whole_human_experience_')
    health_detail = profile.get('how_does_your_product_contribute_to_a_healthier__happier_whole_human_experience_')
    if health_impact == "Yes" and health_detail:
        impact_items.append(("❤️ Health & Wellbeing Impact", health_detail))
    
    # Check for innovation/technology
    innovation = profile.get('how_does_your_company_use_innovation__through_technology_or_to_differentiate_the_business_model__')
    if innovation:
        impact_items.append(("🚀 Innovation & Technology", innovation))
    
    if impact_items:
        st.markdown("### 🌍 Impact & Innovation")
        for title, content in impact_items:
            with st.expander(title):
                st.write(content)
    
    # ------------------------------------------------
    # 💰  FINANCIAL METRICS - Simple Grid
    # ------------------------------------------------
    st.markdown("### 💰 Financial Metrics")
    
    # Prepare financial data
    ltm_revenue = key_metrics.get('LTM Revenue')
    monthly_revenue = key_metrics.get('Monthly Revenue')
    arr_estimate = None
    if monthly_revenue and str(monthly_revenue) != "£0":
        try:
            m_val = float(re.sub(r"[^0-9\.]+", "", str(monthly_revenue)))
            if m_val > 0:
                arr_estimate = m_val * 12
        except ValueError:
            pass
    
    # Try to extract raising amount from different sources
    raising_amount = key_metrics.get('Current Raise Amount')
    if not raising_amount:
        # Look for it in the total_funding_raised string
        funding_str = profile.get('total_funding_raised', '')
        if 'raising' in str(funding_str).lower():
            # Extract the raising amount using regex
            raising_match = re.search(r'raising\s+([£$€][\d\-,.\s]+[kmb]?)', str(funding_str), re.IGNORECASE)
            if raising_match:
                raising_amount = raising_match.group(1).strip()
    
    valuation = key_metrics.get('Valuation')
    
    # For prior funding, extract just the prior amount, not the full string
    total_raised = profile.get('total_funding_raised')
    prior_funding = None
    if total_raised and 'prior funding' in str(total_raised).lower():
        # Extract just the prior funding amount
        prior_match = re.search(r'([£$€][\d,]+)\s*\(prior funding\)', str(total_raised), re.IGNORECASE)
        if prior_match:
            prior_funding = prior_match.group(1)
        else:
            prior_funding = total_raised
    else:
        prior_funding = total_raised
    
    # Two rows of metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("LTM Revenue", _format_currency(ltm_revenue))
    with col2:
        st.metric("Monthly Revenue", _format_currency(monthly_revenue))
    with col3:
        st.metric("ARR (Est.)", _format_currency(arr_estimate))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Amount Raising", _format_currency(raising_amount))
    with col2:
        st.metric("Valuation", _format_currency(valuation))
    with col3:
        st.metric("Prior Funding", _format_currency(prior_funding))
    
    # ------------------------------------------------
    # 📈  INVESTMENT & STRATEGY (New Section)
    # ------------------------------------------------
    strategy_items = []
    
    # Investment use/expansion plans
    investment_use = profile.get('please_expand')
    if investment_use:
        strategy_items.append(("💡 Investment Use & Expansion Plans", investment_use))
    
    # Partnership seeking
    partnership_seek = profile.get('what_is_it_that_you_re_looking_for_with_a_partnership_from_flight_')
    if partnership_seek and partnership_seek != "Other":
        strategy_items.append(("🤝 Partnership Goals", partnership_seek))
    
    if strategy_items:
        st.markdown("### 📈 Investment Strategy")
        for title, content in strategy_items:
            with st.expander(title):
                st.write(content)
    
    # ------------------------------------------------
    # 👥  TEAM & OPERATIONS
    # ------------------------------------------------
    st.markdown("### 👥 Team & Operations")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Team Equity", f"{key_metrics.get('Team Equity', 'N/A')}%" if key_metrics.get('Team Equity') else 'N/A')
    with col2:
        if key_metrics.get('Founders'):
            founder_name = key_metrics['Founders'].split(' (')[0]  # Remove email
            st.metric("Founder", founder_name)
    
    # ------------------------------------------------
    # 🔗  LINKS & RESOURCES
    # ------------------------------------------------
    st.markdown("### 🔗 Links & Resources")
    
    link_col1, link_col2 = st.columns(2)
    with link_col1:
        if profile.get('website'):
            st.markdown(f"🌐 [Company Website]({profile['website']})")
    with link_col2:
        if profile.get('linkedin_url'):
            st.markdown(f"🔗 [LinkedIn Profile]({profile['linkedin_url']})")
    
    # Check for pitch deck
    pitch_deck = profile.get('please_attach_your_pitch_deck')
    if pitch_deck:
        st.markdown("### 📎 Documents")
        if pitch_deck.startswith('http'):
            st.markdown(f"📄 [Pitch Deck]({pitch_deck})")
        else:
            st.write(f"📄 Pitch Deck: {pitch_deck}")
    
    # ------------------------------------------------
    # DEBUG (collapsible only)
    # ------------------------------------------------
    with st.expander("🔍 Debug: Raw Data"):
        st.json(profile)

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
    confidence = research.get('confidence_score_overall')
    
    # Also try other confidence field names
    if confidence is None:
        confidence = research.get('confidence_score')
    if confidence is None:
        confidence = research.get('confidence')
    
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
    
    # Handle confidence score properly
    if confidence is not None and isinstance(confidence, (int, float)):
        # If confidence is already a percentage (>1), don't multiply by 100
        if confidence > 1:
            st.progress(confidence/100, text=f"Confidence Score: {confidence:.0f}%")
        else:
            st.progress(confidence, text=f"Confidence Score: {confidence * 100:.0f}%")
    else:
        st.progress(0, text="Confidence Score: N/A")
    
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
    
    # Export functionality - get a unique key based on the results content
    result_types = list(results.keys())
    export_key = "_".join(sorted(result_types))
    
    with st.expander("📥 Export Results"):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Download as JSON", key=f"download_json_{export_key}"):
                json_str = json.dumps(results, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name="investment_research.json",
                    mime="application/json",
                    key=f"download_button_{export_key}"
                )
        with col2:
            if st.button("Copy to Clipboard", key=f"copy_clipboard_{export_key}"):
                st.code(json.dumps(results, indent=2), language='json')

# Example usage in streamlit_app.py:
"""
from ai_agents.ui.results_display import display_results

# In your main app:
if st.session_state.results:
    display_results(st.session_state.results)
""" 