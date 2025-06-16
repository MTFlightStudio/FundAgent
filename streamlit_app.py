import streamlit as st
import json
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
from pathlib import Path
import sys
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from ai_agents.ui.agent_runner import AgentRunner
from ai_agents.ui.workflow_visualizer import WorkflowVisualizer
from ai_agents.ui.results_display import display_results

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Flight Story Investment Research",
    page_icon="🚀",
    layout="wide"
)

# Initialize session state
if 'agent_runner' not in st.session_state:
    st.session_state.agent_runner = AgentRunner()
if 'workflow_visualizer' not in st.session_state:
    st.session_state.workflow_visualizer = WorkflowVisualizer()
if 'workflow_state' not in st.session_state:
    st.session_state.workflow_state = {}
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'logs' not in st.session_state:
    st.session_state.logs = []

def update_workflow_state(agent_name: str, status: str, error: str = None):
    """Update workflow state and trigger visualization update"""
    st.session_state.workflow_state[agent_name] = {
        'status': status,
        'error': error,
        'timestamp': datetime.now().isoformat()
    }
    # Update the workflow visualizer with new state
    st.session_state.workflow_visualizer.workflow_state = st.session_state.workflow_state

def log_execution(message: str, level: str = "info"):
    """Add a log message to the session state"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.logs.append({
        'timestamp': timestamp,
        'message': message,
        'level': level
    })

def run_company_research():
    """Run company research agent"""
    try:
        update_workflow_state('company_research', 'running')
        log_execution("Starting company research...")
        
        # Get input data based on selected method
        if st.session_state.input_method == "HubSpot Deal ID":
            deal_id = st.session_state.deal_id
            result = st.session_state.agent_runner.run_company_research(deal_id=deal_id)
        else:
            company_name = st.session_state.company_name
            result = st.session_state.agent_runner.run_company_research(company_name=company_name)
        
        st.session_state.results['company_research'] = result
        update_workflow_state('company_research', 'completed')
        log_execution("Company research completed successfully")
        
    except Exception as e:
        error_msg = str(e)
        update_workflow_state('company_research', 'error', error_msg)
        log_execution(f"Company research failed: {error_msg}", "error")
        st.error(f"Company research failed: {error_msg}")

def run_founder_research():
    """Run founder research agent"""
    try:
        update_workflow_state('founder_research', 'running')
        log_execution("Starting founder research...")
        
        if st.session_state.input_method == "HubSpot Deal ID":
            deal_id = st.session_state.deal_id
            result = st.session_state.agent_runner.run_founder_research(deal_id=deal_id)
        else:
            founder_names = st.session_state.founder_names.split('\n')
            linkedin_urls = st.session_state.linkedin_urls.split('\n')
            result = st.session_state.agent_runner.run_founder_research(
                founder_names=founder_names,
                linkedin_urls=linkedin_urls
            )
        
        st.session_state.results['founder_research'] = result
        update_workflow_state('founder_research', 'completed')
        log_execution("Founder research completed successfully")
        
    except Exception as e:
        error_msg = str(e)
        update_workflow_state('founder_research', 'error', error_msg)
        log_execution(f"Founder research failed: {error_msg}", "error")
        st.error(f"Founder research failed: {error_msg}")

def run_market_research():
    """Run market research agent"""
    try:
        update_workflow_state('market_research', 'running')
        log_execution("Starting market research...")
        
        if st.session_state.input_method == "HubSpot Deal ID":
            deal_id = st.session_state.deal_id
            result = st.session_state.agent_runner.run_market_research(deal_id=deal_id)
        else:
            company_name = st.session_state.company_name
            industry = st.session_state.industry
            result = st.session_state.agent_runner.run_market_research(
                company_name=company_name,
                industry=industry
            )
        
        st.session_state.results['market_research'] = result
        update_workflow_state('market_research', 'completed')
        log_execution("Market research completed successfully")
        
    except Exception as e:
        error_msg = str(e)
        update_workflow_state('market_research', 'error', error_msg)
        log_execution(f"Market research failed: {error_msg}", "error")
        st.error(f"Market research failed: {error_msg}")

def run_decision_support():
    """Run decision support agent"""
    try:
        update_workflow_state('decision_support', 'running')
        log_execution("Starting decision support analysis...")
        
        result = st.session_state.agent_runner.run_decision_support(
            company_research=st.session_state.results.get('company_research', {}),
            founder_research=st.session_state.results.get('founder_research', {}),
            market_research=st.session_state.results.get('market_research', {})
        )
        
        st.session_state.results['decision_support'] = result
        update_workflow_state('decision_support', 'completed')
        log_execution("Decision support analysis completed successfully")
        
    except Exception as e:
        error_msg = str(e)
        update_workflow_state('decision_support', 'error', error_msg)
        log_execution(f"Decision support analysis failed: {error_msg}", "error")
        st.error(f"Decision support analysis failed: {error_msg}")

def run_full_pipeline():
    """Run the complete research pipeline"""
    try:
        log_execution("Starting full research pipeline...")
        
        # Run all agents in sequence
        run_company_research()
        run_founder_research()
        run_market_research()
        run_decision_support()
        
        log_execution("Full research pipeline completed successfully")
        
    except Exception as e:
        error_msg = str(e)
        log_execution(f"Full pipeline failed: {error_msg}", "error")
        st.error(f"Full pipeline failed: {error_msg}")

# Header
st.title("🚀 Flight Story Investment Research Dashboard")
st.markdown("*Analyze investment opportunities against Flight Story's 6 key criteria*")

# Sidebar
with st.sidebar:
    st.header("Research Configuration")
    
    # Store input method in session state
    st.session_state.input_method = st.radio(
        "Input Method",
        ["HubSpot Deal ID", "Manual Entry"]
    )
    
    if st.session_state.input_method == "HubSpot Deal ID":
        st.session_state.deal_id = st.text_input("Deal ID", value="227710582988")
    else:
        st.session_state.company_name = st.text_input("Company Name")
        st.session_state.founder_names = st.text_area("Founder Names (one per line)")
        st.session_state.linkedin_urls = st.text_area("LinkedIn URLs (one per line)")
        st.session_state.industry = st.text_input("Industry/Sector")
    
    st.divider()
    
    # Execution controls
    if st.button("🚀 Run Full Analysis", type="primary", use_container_width=True):
        run_full_pipeline()
    
    st.divider()
    
    st.subheader("Run Individual Agents")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🏢 Company", use_container_width=True):
            run_company_research()
        if st.button("🌍 Market", use_container_width=True):
            run_market_research()
    with col2:
        if st.button("👤 Founders", use_container_width=True):
            run_founder_research()
        if st.button("📊 Decision", use_container_width=True):
            run_decision_support()

# Main content area
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔄 Workflow", "🏢 Company", "👤 Founders", 
    "🌍 Market", "📊 Decision", "📝 Logs"
])

with tab1:
    st.header("Research Workflow")
    diagram = st.session_state.workflow_visualizer.generate_mermaid_diagram()
    st.markdown(diagram)
    
with tab2:
    st.header("Company Research")
    if 'company_research' in st.session_state.results:
        display_results({'company_research': st.session_state.results['company_research']})
    
with tab3:
    st.header("Founder Research")
    if 'founder_research' in st.session_state.results:
        display_results({'founder_research': st.session_state.results['founder_research']})
    
with tab4:
    st.header("Market Analysis")
    if 'market_research' in st.session_state.results:
        display_results({'market_research': st.session_state.results['market_research']})
    
with tab5:
    st.header("Investment Decision")
    if 'decision_support' in st.session_state.results:
        display_results({'decision_support': st.session_state.results['decision_support']})
    
with tab6:
    st.header("Execution Logs")
    for log in st.session_state.logs:
        if log['level'] == 'error':
            st.error(f"{log['timestamp']} - {log['message']}")
        else:
            st.info(f"{log['timestamp']} - {log['message']}") 