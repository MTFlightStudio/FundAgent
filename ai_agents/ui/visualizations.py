import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import streamlit as st
from typing import Dict, Any, List, Optional
import numpy as np

def create_investment_criteria_radar_chart(decision_data: Dict[str, Any]) -> go.Figure:
    """
    Create a radar chart showing Flight Story's 6 investment criteria scores
    """
    # Extract criteria scores from decision data
    criteria_mapping = {
        'fs_focus_industry_fit': 'Industry Focus Fit',
        'fs_mission_alignment': 'Mission Alignment', 
        'fs_exciting_solution_to_problem': 'Exciting Solution',
        'fs_founded_something_relevant_before': 'Founded Before',
        'fs_impressive_relevant_past_experience': 'Impressive Experience',
        'fs_exceptionally_smart_or_strategic': 'Exceptionally Smart'
    }
    
    criteria_scores = {}
    
    # Extract from the correct path in decision_support result
    investment_assessment = None
    
    if isinstance(decision_data, dict):
        # Check if it's wrapped in investment_research
        if 'investment_research' in decision_data:
            investment_research = decision_data['investment_research']
            if isinstance(investment_research, dict):
                investment_assessment = investment_research.get('investment_assessment', {})
        # Direct access to investment_assessment
        elif 'investment_assessment' in decision_data:
            investment_assessment = decision_data['investment_assessment']
    
    # Extract scores from investment_assessment
    if investment_assessment:
        for key, label in criteria_mapping.items():
            if key in investment_assessment:
                value = investment_assessment[key]
                # Convert boolean to score
                if isinstance(value, bool):
                    criteria_scores[label] = 1.0 if value else 0.0
                elif value is None:
                    criteria_scores[label] = 0.0  # None means not met
                else:
                    criteria_scores[label] = float(value)
    
    # If no scores found, use zeros
    if not criteria_scores:
        for key, label in criteria_mapping.items():
            criteria_scores[label] = 0.0
    
    # Create radar chart
    categories = list(criteria_scores.keys())
    values = list(criteria_scores.values())
    
    # Close the radar chart by adding first value at the end
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        name='Investment Criteria',
        fillcolor='rgba(99, 110, 250, 0.2)',
        line=dict(color='rgb(99, 110, 250)', width=3),
        marker=dict(size=8, color='rgb(99, 110, 250)')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                ticktext=['Not Met', '25%', '50%', '75%', 'Met']
            ),
            angularaxis=dict(
                tickfont=dict(size=12),
                rotation=90,
                direction='clockwise'
            )
        ),
        title=dict(
            text="Flight Story Investment Criteria Analysis",
            x=0.5,
            font=dict(size=16, color='rgb(50, 50, 50)')
        ),
        showlegend=False,
        height=500,
        font=dict(family="Arial, sans-serif")
    )
    
    return fig

def create_funding_timeline_chart(company_data: Dict[str, Any]) -> Optional[go.Figure]:
    """
    Create an interactive timeline chart of funding history
    """
    # Try to extract funding data from company profile
    funding_data = []
    
    if isinstance(company_data, dict):
        # Look for funding information in various structures
        search_paths = [
            company_data.get('company_profile', {}),
            company_data.get('funding_history', []),
            company_data.get('financial_info', {}),
            company_data
        ]
        
        for data_section in search_paths:
            if isinstance(data_section, dict):
                if 'funding_rounds' in data_section:
                    funding_data = data_section['funding_rounds']
                elif 'funding_history' in data_section:
                    funding_data = data_section['funding_history']
                elif 'investments' in data_section:
                    funding_data = data_section['investments']
    
    if not funding_data or not isinstance(funding_data, list):
        # Create mock data for demonstration
        funding_data = [
            {'round': 'Seed', 'amount': 500000, 'date': '2022-01', 'investors': ['Angel Investors']},
            {'round': 'Series A', 'amount': 2000000, 'date': '2023-06', 'investors': ['VC Fund A', 'VC Fund B']},
        ]
    
    if not funding_data:
        return None
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(funding_data)
    
    # Ensure required columns exist
    if 'round' not in df.columns:
        df['round'] = [f'Round {i+1}' for i in range(len(df))]
    if 'amount' not in df.columns:
        df['amount'] = [1000000] * len(df)
    if 'date' not in df.columns:
        df['date'] = ['2023-01'] * len(df)
    
    # Create timeline chart
    fig = go.Figure()
    
    # Add funding rounds as scatter plot
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['amount'],
        mode='markers+lines+text',
        text=df['round'],
        textposition='top center',
        name='Funding Rounds',
        marker=dict(
            size=[15 + (i * 5) for i in range(len(df))],  # Increasing size for later rounds
            color='rgb(99, 110, 250)',
            line=dict(width=2, color='white')
        ),
        line=dict(color='rgba(99, 110, 250, 0.3)', width=2),
        hovertemplate='<b>%{text}</b><br>Amount: $%{y:,.0f}<br>Date: %{x}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text="Funding History Timeline",
            x=0.5,
            font=dict(size=16, color='rgb(50, 50, 50)')
        ),
        xaxis_title="Date",
        yaxis_title="Funding Amount ($)",
        yaxis=dict(tickformat='$,.0f'),
        height=400,
        hovermode='closest',
        showlegend=False
    )
    
    return fig

def create_market_size_breakdown_chart(market_data: Dict[str, Any]) -> Optional[go.Figure]:
    """
    Create a market size breakdown chart (TAM, SAM, SOM)
    """
    # Extract market size data from the correct structure
    market_analysis = None
    
    if isinstance(market_data, dict):
        if 'market_analysis' in market_data:
            market_analysis = market_data['market_analysis']
        else:
            market_analysis = market_data
    
    if not market_analysis:
        return None
    
    # Extract TAM, SAM, SOM values
    tam_str = market_analysis.get('market_size_tam', '')
    sam_str = market_analysis.get('market_size_sam', '')
    som_str = market_analysis.get('market_size_som', '')
    
    # Function to parse currency strings
    def parse_market_size(size_str):
        if not size_str or size_str == 'N/A':
            return None
        
        import re
        # Remove currency symbols and spaces
        size_str = str(size_str).replace('$', '').replace('€', '').replace('£', '').strip()
        
        # Extract numeric value and unit
        match = re.match(r'([\d,]+\.?\d*)\s*([BMK])?', size_str, re.IGNORECASE)
        if match:
            value = float(match.group(1).replace(',', ''))
            unit = match.group(2)
            if unit:
                multipliers = {'K': 1e3, 'M': 1e6, 'B': 1e9}
                value *= multipliers.get(unit.upper(), 1)
            return value
        return None
    
    tam = parse_market_size(tam_str)
    sam = parse_market_size(sam_str)
    som = parse_market_size(som_str)
    
    # Use defaults if parsing failed
    if not tam:
        return None  # Can't create chart without TAM
    
    if not sam:
        sam = tam * 0.1  # Default to 10% of TAM
    if not som:
        som = sam * 0.1  # Default to 10% of SAM
    
    # Create funnel chart
    fig = go.Figure()
    
    markets = ['TAM<br>(Total Addressable)', 'SAM<br>(Serviceable Addressable)', 'SOM<br>(Serviceable Obtainable)']
    values = [tam, sam, som]
    colors = ['rgb(99, 110, 250)', 'rgb(255, 146, 51)', 'rgb(239, 85, 59)']
    
    for i, (market, value, color) in enumerate(zip(markets, values, colors)):
        # Calculate bar width for funnel effect
        width = 1 - (i * 0.2)
        
        # Format the display value
        if value >= 1e9:
            display_value = f'${value/1e9:.1f}B'
        elif value >= 1e6:
            display_value = f'${value/1e6:.1f}M'
        else:
            display_value = f'${value/1e3:.1f}K'
        
        fig.add_trace(go.Bar(
            name=market,
            x=[market],
            y=[value],
            width=width,
            marker_color=color,
            text=display_value,
            textposition='inside',
            textfont=dict(size=14, color='white'),
            hovertemplate=f'<b>{market}</b><br>Market Size: ${value:,.0f}<extra></extra>'
        ))
    
    # Add CAGR if available
    cagr = market_analysis.get('market_growth_rate_cagr', '')
    
    fig.update_layout(
        title=dict(
            text=f"Market Size Analysis{f' (CAGR: {cagr})' if cagr else ''}",
            x=0.5,
            font=dict(size=16, color='rgb(50, 50, 50)')
        ),
        xaxis_title="Market Segments",
        yaxis_title="Market Size ($)",
        yaxis=dict(tickformat='$,.0f'),
        height=400,
        showlegend=False,
        bargap=0.3
    )
    
    return fig

def create_founder_skills_matrix(founder_data: Dict[str, Any]) -> Optional[go.Figure]:
    """
    Create a skills matrix heatmap for founders
    """
    founders = []
    
    if isinstance(founder_data, dict):
        if 'founders' in founder_data:
            founders = founder_data['founders']
        elif 'founder_profiles' in founder_data:
            founders = founder_data['founder_profiles']
    
    if not founders:
        return None
    
    # Extract skills for each founder
    founder_names = []
    skills_data = []
    all_skills = set()
    
    for founder in founders:
        if isinstance(founder, dict):
            name = founder.get('name', founder.get('founder_name', 'Unknown'))
            founder_names.append(name)
            
            # Extract skills from various fields
            skills = {}
            
            # Look for skills in different structures
            skill_fields = ['skills', 'key_skills', 'expertise', 'competencies']
            for field in skill_fields:
                if field in founder:
                    founder_skills = founder[field]
                    if isinstance(founder_skills, list):
                        for skill in founder_skills:
                            if isinstance(skill, str):
                                skills[skill] = 1.0
                            elif isinstance(skill, dict) and 'skill' in skill:
                                skills[skill['skill']] = skill.get('level', 1.0)
                    elif isinstance(founder_skills, dict):
                        skills.update(founder_skills)
            
            # Default skills if none found
            if not skills:
                skills = {
                    'Leadership': 0.8,
                    'Technical': 0.7,
                    'Business Development': 0.6,
                    'Strategy': 0.75,
                    'Marketing': 0.5
                }
            
            skills_data.append(skills)
            all_skills.update(skills.keys())
    
    if not founder_names or not all_skills:
        return None
    
    # Create matrix
    skills_list = sorted(all_skills)
    matrix = []
    
    for founder_skills in skills_data:
        row = [founder_skills.get(skill, 0.0) for skill in skills_list]
        matrix.append(row)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=skills_list,
        y=founder_names,
        colorscale='RdYlBu_r',
        zmin=0,
        zmax=1,
        text=[[f'{val:.1f}' for val in row] for row in matrix],
        texttemplate='%{text}',
        textfont={'size': 12},
        hovertemplate='<b>%{y}</b><br>%{x}: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text="Founder Skills Assessment Matrix",
            x=0.5,
            font=dict(size=16, color='rgb(50, 50, 50)')
        ),
        xaxis_title="Skills",
        yaxis_title="Founders",
        height=300 + (len(founder_names) * 50),
        font=dict(family="Arial, sans-serif")
    )
    
    return fig

def create_risk_opportunity_matrix(decision_data: Dict[str, Any]) -> go.Figure:
    """
    Create a risk vs opportunity matrix scatter plot based on actual assessment data
    """
    # Extract investment assessment data
    investment_research = None
    investment_assessment = None
    
    if isinstance(decision_data, dict):
        if 'investment_research' in decision_data:
            investment_research = decision_data['investment_research']
            if isinstance(investment_research, dict):
                investment_assessment = investment_research.get('investment_assessment', {})
        elif 'investment_assessment' in decision_data:
            investment_assessment = decision_data['investment_assessment']
    
    # Calculate risk and opportunity scores based on actual criteria
    if investment_assessment:
        # Count met criteria
        criteria_scores = {
            'industry': investment_assessment.get('fs_focus_industry_fit', False),
            'mission': investment_assessment.get('fs_mission_alignment', False),
            'solution': investment_assessment.get('fs_exciting_solution_to_problem', False),
            'founded_before': investment_assessment.get('fs_founded_something_relevant_before', False),
            'experience': investment_assessment.get('fs_impressive_relevant_past_experience', False),
            'smart': investment_assessment.get('fs_exceptionally_smart_or_strategic', False)
        }
        
        criteria_met = sum(1 for v in criteria_scores.values() if v)
        
        # Calculate opportunity score (higher when more criteria are met)
        opportunity_score = criteria_met / 6.0
        
        # Calculate risk score based on missing criteria and identified risks
        risk_factors = investment_assessment.get('key_risk_factors', [])
        opportunities = investment_assessment.get('key_opportunities', [])
        
        # Base risk on missing criteria
        risk_score = (6 - criteria_met) / 6.0
        
        # Adjust based on identified risks/opportunities
        if risk_factors:
            risk_adjustment = min(0.2, len(risk_factors) * 0.05)
            risk_score = min(1.0, risk_score + risk_adjustment)
        
        if opportunities:
            opp_adjustment = min(0.2, len(opportunities) * 0.05)
            opportunity_score = min(1.0, opportunity_score + opp_adjustment)
        
        # Get recommendation and confidence
        recommendation = 'UNKNOWN'
        confidence = 0.5
        
        if investment_research:
            recommendation = investment_research.get('overall_summary_and_recommendation', 'UNKNOWN')
            confidence = investment_research.get('confidence_score_overall', 0.5)
    else:
        # Default values if no assessment data
        risk_score = 0.5
        opportunity_score = 0.5
        recommendation = 'UNKNOWN'
        confidence = 0.5
    
    # Create quadrant background
    fig = go.Figure()
    
    # Add quadrant backgrounds
    quadrants = [
        {'x': [0, 0.5, 0.5, 0], 'y': [0, 0, 0.5, 0.5], 'color': 'rgba(239, 85, 59, 0.1)', 'name': 'Low Opp, Low Risk'},
        {'x': [0.5, 1, 1, 0.5], 'y': [0, 0, 0.5, 0.5], 'color': 'rgba(255, 193, 7, 0.1)', 'name': 'Low Opp, High Risk'},
        {'x': [0, 0.5, 0.5, 0], 'y': [0.5, 0.5, 1, 1], 'color': 'rgba(40, 167, 69, 0.1)', 'name': 'High Opp, Low Risk'},
        {'x': [0.5, 1, 1, 0.5], 'y': [0.5, 0.5, 1, 1], 'color': 'rgba(255, 146, 51, 0.1)', 'name': 'High Opp, High Risk'}
    ]
    
    for quad in quadrants:
        fig.add_trace(go.Scatter(
            x=quad['x'] + [quad['x'][0]],
            y=quad['y'] + [quad['y'][0]],
            fill='toself',
            fillcolor=quad['color'],
            line=dict(color='rgba(0,0,0,0)'),
            name=quad['name'],
            hoverinfo='name',
            showlegend=False
        ))
    
    # Color based on recommendation
    if 'PASS' in str(recommendation).upper():
        color = 'green'
    elif 'EXPLORE' in str(recommendation).upper():
        color = 'orange'
    elif 'DECLINE' in str(recommendation).upper():
        color = 'red'
    else:
        color = 'gray'
    
    # Add the investment opportunity point
    fig.add_trace(go.Scatter(
        x=[risk_score],
        y=[opportunity_score],
        mode='markers+text',
        marker=dict(
            size=20 + (confidence * 20),  # Size based on confidence
            color=color,
            line=dict(width=3, color='white'),
            symbol='star'
        ),
        text=[recommendation],
        textposition='bottom center',
        name='Investment Position',
        hovertemplate='<b>Investment Position</b><br>Risk: %{x:.1%}<br>Opportunity: %{y:.1%}<br>Recommendation: ' + str(recommendation) + '<br>Confidence: ' + f'{confidence:.1%}' + '<extra></extra>'
    ))
    
    # Add quadrant labels
    fig.add_annotation(x=0.25, y=0.75, text="🎯 Sweet Spot<br>(High Opp, Low Risk)", showarrow=False, font=dict(size=10))
    fig.add_annotation(x=0.75, y=0.75, text="⚠️ High Stakes<br>(High Opp, High Risk)", showarrow=False, font=dict(size=10))
    fig.add_annotation(x=0.25, y=0.25, text="😴 Low Impact<br>(Low Opp, Low Risk)", showarrow=False, font=dict(size=10))
    fig.add_annotation(x=0.75, y=0.25, text="🚫 Avoid<br>(Low Opp, High Risk)", showarrow=False, font=dict(size=10))
    
    fig.update_layout(
        title=dict(
            text="Risk vs Opportunity Analysis",
            x=0.5,
            font=dict(size=16, color='rgb(50, 50, 50)')
        ),
        xaxis=dict(
            title="Risk Level",
            range=[0, 1],
            tickformat='.0%',
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)'
        ),
        yaxis=dict(
            title="Opportunity Level", 
            range=[0, 1],
            tickformat='.0%',
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)'
        ),
        height=500,
        showlegend=False,
        plot_bgcolor='white'
    )
    
    return fig

def display_all_visualizations(results: Dict[str, Any]):
    """
    Display all available visualizations based on results data
    """
    st.header("📊 Investment Analysis Visualizations")
    
    # Check what data is available
    has_decision = 'decision_support' in results
    has_company = 'company_research' in results
    has_founder = 'founder_research' in results
    has_market = 'market_research' in results
    
    if not any([has_decision, has_company, has_founder, has_market]):
        st.info("No analysis data available yet. Run the research agents to see visualizations.")
        return
    
    # Create tabs for different visualization categories
    viz_tabs = st.tabs([
        "🎯 Investment Criteria", 
        "📈 Market Analysis", 
        "👥 Founder Assessment", 
        "⚖️ Risk/Opportunity"
    ])
    
    with viz_tabs[0]:
        if has_decision:
            st.subheader("Flight Story Investment Criteria")
            fig = create_investment_criteria_radar_chart(results['decision_support'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Add criteria explanation
            with st.expander("📋 Criteria Explanation"):
                st.markdown("""
                **Flight Story's 6 Investment Criteria:**
                - **Industry Focus Fit**: Alignment with our focus industries (Media, Brand, Tech, Creator Economy)
                - **Mission Alignment**: Avoids harm to humanity and brand reputational risk
                - **Exciting Solution**: Innovative approach to real and meaningful problems
                - **Founded Before**: Has founded something impressive and relevant before
                - **Impressive Experience**: Impressive and relevant past work experience
                - **Exceptionally Smart**: Evidence of being super smart or strategic
                """)
        else:
            st.info("🔄 Run Decision Support analysis to see investment criteria visualization")
    
    with viz_tabs[1]:
        if has_market:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Market Size Breakdown")
                market_fig = create_market_size_breakdown_chart(results['market_research'])
                if market_fig:
                    st.plotly_chart(market_fig, use_container_width=True)
            
            with col2:
                st.subheader("Market Metrics")
                market_data = results['market_research']
                if isinstance(market_data, dict):
                    # Extract from the correct market analysis structure
                    if 'market_analysis' in market_data:
                        analysis = market_data['market_analysis']
                    else:
                        analysis = market_data
                    
                    # Extract key metrics using correct field names
                    tam = analysis.get('market_size_tam', 'Not specified')
                    cagr = analysis.get('market_growth_rate_cagr', 'Not specified')
                    timing = analysis.get('market_timing_assessment', 'Not specified')
                    
                    st.metric("Total Addressable Market", tam)
                    st.metric("Market Growth Rate (CAGR)", cagr)
                    st.metric("Market Timing", timing)
        
        if has_company:
            st.subheader("Funding Timeline")
            funding_fig = create_funding_timeline_chart(results['company_research'])
            if funding_fig:
                st.plotly_chart(funding_fig, use_container_width=True)
        
        if not has_market and not has_company:
            st.info("🔄 Run Market Research and Company Research to see market visualizations")
    
    with viz_tabs[2]:
        if has_founder:
            st.subheader("Founder Skills Assessment")
            try:
                skills_fig = create_founder_skills_matrix(results['founder_research'])
                if skills_fig:
                    st.plotly_chart(skills_fig, use_container_width=True)
                else:
                    st.info("Founder skills data not available in sufficient detail for visualization")
            except Exception as e:
                st.warning(f"⚠️ Skills visualization temporarily unavailable: {str(e)}")
                st.info("Founder skills data not available in sufficient detail for visualization")
                
                # Show basic founder info instead
                founder_data = results['founder_research']
                if isinstance(founder_data, dict) and 'founders' in founder_data:
                    st.subheader("Founder Overview")
                    for i, founder in enumerate(founder_data['founders']):
                        if isinstance(founder, dict):
                            name = founder.get('name', f'Founder {i+1}')
                            with st.expander(f"👤 {name}"):
                                st.json(founder)
        else:
            st.info("🔄 Run Founder Research to see team assessment visualizations")
    
    with viz_tabs[3]:
        if has_decision:
            st.subheader("Investment Position Analysis")
            risk_fig = create_risk_opportunity_matrix(results['decision_support'])
            st.plotly_chart(risk_fig, use_container_width=True)
            
            # Add position interpretation
            decision_data = results['decision_support']
            recommendation = decision_data.get('recommendation', 'Unknown')
            confidence = decision_data.get('confidence_score', 0)
            
            if recommendation == 'PASS':
                st.success(f"✅ **Recommended Investment** (Confidence: {confidence:.1%})")
                st.markdown("This opportunity shows strong potential with acceptable risk levels.")
            elif recommendation == 'FAIL':
                st.error(f"❌ **Not Recommended** (Confidence: {confidence:.1%})")
                st.markdown("This opportunity does not meet our investment criteria.")
            else:
                st.warning(f"⚠️ **Under Review** (Confidence: {confidence:.1%})")
                st.markdown("This opportunity requires further analysis.")
        else:
            st.info("🔄 Run Decision Support analysis to see risk/opportunity visualization") 