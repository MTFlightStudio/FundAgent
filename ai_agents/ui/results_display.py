import streamlit as st
from typing import Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

def display_company_profile(company_data: Dict):
    """Display company profile information in a card layout"""
    st.header("Company Profile")
    
    # Main company info
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(company_data.get('name', 'Unknown Company'))
        st.markdown(f"**Industry:** {company_data.get('industry', 'N/A')}")
        st.markdown(f"**Founded:** {company_data.get('founded_year', 'N/A')}")
        st.markdown(f"**Location:** {company_data.get('location', 'N/A')}")
        
        # Company description in expandable section
        with st.expander("Company Description", expanded=True):
            st.markdown(company_data.get('description', 'No description available'))
    
    with col2:
        # Key metrics
        st.subheader("Key Metrics")
        metrics = company_data.get('metrics', {})
        st.metric("Team Size", metrics.get('team_size', 'N/A'))
        st.metric("Total Funding", metrics.get('total_funding', 'N/A'))
        st.metric("Valuation", metrics.get('valuation', 'N/A'))
    
    # Funding History
    st.subheader("Funding History")
    funding_rounds = company_data.get('funding_rounds', [])
    if funding_rounds:
        df_funding = pd.DataFrame(funding_rounds)
        st.dataframe(
            df_funding,
            column_config={
                "date": "Date",
                "round": "Round",
                "amount": st.column_config.NumberColumn("Amount", format="$%.2fM"),
                "investors": "Investors"
            },
            hide_index=True
        )
    else:
        st.info("No funding history available")

def display_founder_profiles(founder_data: List[Dict]):
    """Display founder profiles with background and assessment"""
    st.header("Founder Profiles")
    
    for founder in founder_data:
        with st.expander(f"{founder.get('name', 'Unknown Founder')}", expanded=True):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Background Information
                st.subheader("Background")
                st.markdown(f"**Current Role:** {founder.get('current_role', 'N/A')}")
                st.markdown(f"**Previous Experience:** {founder.get('previous_experience', 'N/A')}")
                
                # Education
                st.subheader("Education")
                for edu in founder.get('education', []):
                    st.markdown(f"- {edu.get('degree', '')} in {edu.get('field', '')} from {edu.get('institution', '')}")
            
            with col2:
                # Investment Criteria Assessment
                st.subheader("Investment Criteria")
                criteria = founder.get('investment_criteria', {})
                
                # Display each criterion with traffic light
                for criterion, status in criteria.items():
                    color = {
                        'pass': '🟢',
                        'fail': '🔴',
                        'warning': '🟡'
                    }.get(status.lower(), '⚪')
                    st.markdown(f"{color} {criterion}")

def display_market_analysis(market_data: Dict):
    """Display market analysis with charts and competitor landscape"""
    st.header("Market Analysis")
    
    # TAM/SAM/SOM Visualization
    st.subheader("Market Size Analysis")
    market_sizes = market_data.get('market_sizes', {})
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Market Size',
        x=['TAM', 'SAM', 'SOM'],
        y=[
            market_sizes.get('tam', 0),
            market_sizes.get('sam', 0),
            market_sizes.get('som', 0)
        ],
        text=[f"${v}B" for v in [
            market_sizes.get('tam', 0),
            market_sizes.get('sam', 0),
            market_sizes.get('som', 0)
        ]],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Market Size Breakdown",
        yaxis_title="Size (Billions USD)",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True, key="market_size_chart")
    
    # Growth Rate
    st.subheader("Market Growth")
    growth_data = market_data.get('growth_rate', {})
    st.metric(
        "CAGR",
        f"{growth_data.get('cagr', 0)}%",
        f"{growth_data.get('cagr_change', 0)}%"
    )
    
    # Competitor Landscape
    st.subheader("Competitor Landscape")
    competitors = market_data.get('competitors', [])
    if competitors:
        df_competitors = pd.DataFrame(competitors)
        st.dataframe(
            df_competitors,
            column_config={
                "name": "Company",
                "funding": st.column_config.NumberColumn("Funding", format="$%.2fM"),
                "team_size": "Team Size",
                "market_share": st.column_config.NumberColumn("Market Share", format="%.1f%%")
            },
            hide_index=True,
            key="competitor_table"
        )
    else:
        st.info("No competitor data available")

def display_decision_support(decision_data: Dict):
    """Display decision support summary with criteria checklist and recommendation"""
    st.header("Investment Decision Support")
    
    # Overall Recommendation
    recommendation = decision_data.get('recommendation', {})
    st.subheader("Overall Recommendation")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**Decision:** {recommendation.get('decision', 'N/A')}")
        st.markdown(f"**Confidence Score:** {recommendation.get('confidence_score', 0)}%")
    
    with col2:
        # Confidence gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=recommendation.get('confidence_score', 0),
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [0, 100]},
                  'bar': {'color': "darkblue"},
                  'steps': [
                      {'range': [0, 50], 'color': "lightgray"},
                      {'range': [50, 75], 'color': "gray"},
                      {'range': [75, 100], 'color': "darkgray"}
                  ]}
        ))
        fig.update_layout(height=200)
        st.plotly_chart(fig, use_container_width=True, key="confidence_gauge")
    
    # Investment Criteria Checklist
    st.subheader("Investment Criteria Assessment")
    criteria = decision_data.get('criteria', {})
    
    for criterion, assessment in criteria.items():
        col1, col2, col3 = st.columns([2, 1, 3])
        with col1:
            st.markdown(f"**{criterion}**")
        with col2:
            status = assessment.get('status', 'pending')
            color = {
                'pass': '🟢',
                'fail': '🔴',
                'warning': '🟡',
                'pending': '⚪'
            }.get(status.lower(), '⚪')
            st.markdown(color)
        with col3:
            st.markdown(assessment.get('explanation', 'No explanation available'))
    
    # Risks and Opportunities
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Key Risks")
        risks = decision_data.get('risks', [])
        for risk in risks:
            st.markdown(f"- {risk}")
    
    with col2:
        st.subheader("Key Opportunities")
        opportunities = decision_data.get('opportunities', [])
        for opp in opportunities:
            st.markdown(f"- {opp}")

def display_results(results: Dict):
    """Main function to display all results"""
    if not results:
        st.warning("No results available to display")
        return
    
    # Create tabs for different sections
    tabs = []
    tab_contents = {}
    
    if 'company_research' in results:
        tabs.append("🏢 Company")
        tab_contents["🏢 Company"] = lambda: display_company_profile(results['company_research'])
    
    if 'founder_research' in results:
        tabs.append("👤 Founders")
        tab_contents["👤 Founders"] = lambda: display_founder_profiles(results['founder_research'].get('founders', []))
    
    if 'market_research' in results:
        tabs.append("🌍 Market")
        tab_contents["🌍 Market"] = lambda: display_market_analysis(results['market_research'])
    
    if 'decision_support' in results:
        tabs.append("📊 Decision")
        tab_contents["📊 Decision"] = lambda: display_decision_support(results['decision_support'])
    
    if not tabs:
        st.warning("No valid results to display")
        return
    
    # Create tabs and display content
    tab_labels = st.tabs(tabs)
    for tab_label, tab_content in zip(tab_labels, tab_contents.values()):
        with tab_label:
            tab_content()

# Example usage in streamlit_app.py:
"""
from ai_agents.ui.results_display import display_results

# In your main app:
if st.session_state.results:
    display_results(st.session_state.results)
""" 