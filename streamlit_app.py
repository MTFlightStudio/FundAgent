import streamlit as st
import json
import pandas as pd
import time
from datetime import datetime
import plotly.graph_objects as go
from pathlib import Path
import sys
import logging
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from ai_agents.ui.agent_runner import AgentRunner
from ai_agents.ui.workflow_visualizer import WorkflowVisualizer
from ai_agents.ui.results_display import (
    display_company_profile, 
    display_founder_profiles, 
    display_market_analysis, 
    display_investment_decision
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
    page_title="Flight Story Investment Research",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
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
if 'execution_times' not in st.session_state:
    st.session_state.execution_times = {}

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
        
        # Always store the result, even if it's an error
        st.session_state.results['company_research'] = result
        
        # Check if the result indicates success
        if result and (result.get('status') == 'success' or (isinstance(result, dict) and 'company_name' in result)):
            update_workflow_state('company_research', 'completed')
            log_execution("Company research completed successfully")
        else:
            error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
            update_workflow_state('company_research', 'error', error_msg)
            log_execution(f"Company research failed: {error_msg}", "error")
            st.error(f"Company research failed: {error_msg}")
        
    except Exception as e:
        error_msg = str(e)
        # Store the error result
        st.session_state.results['company_research'] = {
            'status': 'error',
            'error': error_msg,
            'timestamp': time.time()
        }
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
        
        # Check if we have the required company research
        company_research = st.session_state.results.get('company_research')
        
        # Helper function to check if company research is valid
        def is_valid_company_research(data):
            if not data or not isinstance(data, dict):
                return False
            # Don't accept if explicitly marked as error
            if data.get('status') == 'error':
                return False
            # Accept if marked as success OR contains company data
            return data.get('status') == 'success' or 'company_name' in data
        
        if not is_valid_company_research(company_research):
            st.error("❌ Company research is required for decision support. Please run company research first.")
            update_workflow_state('decision_support', 'error', 'Company research not available')
            return
        
        # Log what data we're passing
        log_execution(f"Company research available: {company_research is not None}")
        log_execution(f"Company research status: {company_research.get('status') if company_research else 'None'}")
        
        result = st.session_state.agent_runner.run_decision_support(
            company_research=company_research,
            founder_research=st.session_state.results.get('founder_research'),
            market_research=st.session_state.results.get('market_research')
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
    """Run the complete research pipeline with enhanced progress tracking"""
    pipeline_start_time = time.time()
    
    try:
        log_execution("🚀 Starting comprehensive investment analysis pipeline...")
        
        # Create a progress container for the entire pipeline
        progress_container = st.empty()
        
        with progress_container.container():
            # Overall pipeline progress
            pipeline_progress = st.progress(0)
            pipeline_status = st.empty()
            
            # Step 1: Company Research
            pipeline_status.text("🏢 Step 1/4: Researching company profile...")
            pipeline_progress.progress(10)
            with st.spinner("🔍 Analyzing company background, funding, and business model..."):
                run_company_research()
            pipeline_progress.progress(25)
            st.success("✅ Company research completed!")
            
            # Step 2: Founder Research  
            pipeline_status.text("👤 Step 2/4: Analyzing founder backgrounds...")
            pipeline_progress.progress(30)
            with st.spinner("🔗 Researching founder experience and LinkedIn profiles..."):
                run_founder_research()
            pipeline_progress.progress(50)
            st.success("✅ Founder research completed!")
            
            # Step 3: Market Research
            pipeline_status.text("🌍 Step 3/4: Conducting market intelligence...")
            pipeline_progress.progress(55)
            with st.spinner("📈 Analyzing market size, trends, and competitive landscape..."):
                run_market_research()
            pipeline_progress.progress(75)
            st.success("✅ Market analysis completed!")
            
            # Step 4: Decision Support
            pipeline_status.text("🎯 Step 4/4: Generating investment recommendation...")
            pipeline_progress.progress(80)
            with st.spinner("⚖️ Evaluating against Flight Story's 6 investment criteria..."):
                run_decision_support()
            pipeline_progress.progress(100)
            
            # Final status
            pipeline_status.text("🎉 Complete investment analysis finished!")
        
        # Clear progress container and show celebration
        progress_container.empty()
        
        # Calculate total execution time
        total_time = time.time() - pipeline_start_time
        
        # Show success with balloons animation
        st.balloons()
        st.success(f"🎉 **Investment Analysis Complete!** Total time: {total_time:.1f} seconds")
        
        # Show next steps
        st.info("📋 **Next Steps:** Review results in the tabs below and export your investment report!")
        
        log_execution(f"Full research pipeline completed successfully in {total_time:.1f} seconds")
        
    except Exception as e:
        # Clear any progress indicators on error
        if 'progress_container' in locals():
            progress_container.empty()
            
        error_msg = str(e)
        log_execution(f"Full pipeline failed: {error_msg}", "error")
        st.error(f"❌ **Pipeline Failed:** {error_msg}")
        st.warning("💡 **Tip:** Check individual agent results above - partial results may still be available!")

# Simple sidebar with system status
st.sidebar.title("🚀 Flight Story")
st.sidebar.markdown("### AI Investment Research Platform")

# System Status
st.sidebar.markdown("---")
with st.sidebar.expander("🔍 System Status"):
    def check_api_connections():
        """Check if required environment variables are set"""
        import os
        status = {}
        status['hubspot'] = bool(os.getenv('HUBSPOT_ACCESS_TOKEN'))
        status['openai'] = bool(os.getenv('OPENAI_API_KEY'))
        status['tavily'] = bool(os.getenv('TAVILY_API_KEY'))
        status['relevance'] = bool(os.getenv('RELEVANCE_AI_TOKEN'))
        return status
    
    api_status = check_api_connections()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("🔗 HubSpot", "✅ Ready" if api_status['hubspot'] else "❌ Missing")
        st.metric("🤖 OpenAI", "✅ Ready" if api_status['openai'] else "❌ Missing")
    
    with col2:
        st.metric("🔍 Tavily", "✅ Ready" if api_status['tavily'] else "❌ Missing")
        st.metric("🧠 Relevance AI", "✅ Ready" if api_status['relevance'] else "❌ Missing")

# Performance Metrics
if st.session_state.execution_times:
    st.sidebar.markdown("---")
    with st.sidebar.expander("⏱️ Performance Metrics"):
        for agent, time_taken in st.session_state.execution_times.items():
            st.metric(f"🔄 {agent.replace('_', ' ').title()}", f"{time_taken:.1f}s")
        
        total_time = sum(st.session_state.execution_times.values())
        st.metric("🎯 Total Analysis Time", f"{total_time:.1f}s")

# Main content area
st.title("🚀 Flight Story Investment Research")
st.markdown("### AI-Powered Investment Analysis Platform")

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
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🔄 Workflow", "🏢 Company", "👤 Founders", 
    "🌍 Market", "📊 Decision", "📈 Charts", "📝 Logs"
])

with tab1:
    # Use the enhanced workflow display function
    from ai_agents.ui.workflow_visualizer import display_workflow
    display_workflow()
    
    # Comprehensive export section
    st.subheader("📦 Complete Research Package")
    
    # Show available results
    available_results = list(st.session_state.results.keys())
    if available_results:
        st.write("**Available Research Components:**")
        for result_type in available_results:
            status_icon = "✅" if result_type in st.session_state.results else "❌"
            result_label = {
                'company_research': 'Company Research',
                'founder_research': 'Founder Research', 
                'market_research': 'Market Analysis',
                'decision_support': 'Investment Decision'
            }.get(result_type, result_type)
            st.write(f"{status_icon} {result_label}")
        
        # Comprehensive export buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📋 Export Complete Report", key="export_complete_report"):
                # Create comprehensive report with metadata
                from datetime import datetime
                
                # Extract key information for the report
                company_name = "Unknown Company"
                deal_id = st.session_state.get('deal_id', 'N/A')
                
                if 'company_research' in st.session_state.results:
                    company_data = st.session_state.results['company_research']
                    if isinstance(company_data, dict) and 'company_profile' in company_data:
                        company_name = company_data['company_profile'].get('company_name', company_name)
                    elif isinstance(company_data, dict):
                        company_name = company_data.get('company_name', company_name)
                
                # Create comprehensive report
                complete_report = {
                    "report_metadata": {
                        "generated_at": datetime.now().isoformat(),
                        "company_name": company_name,
                        "deal_id": deal_id,
                        "input_method": st.session_state.get('input_method', 'N/A'),
                        "components_included": available_results,
                        "report_version": "1.0"
                    },
                    "research_results": st.session_state.results,
                    "execution_logs": st.session_state.logs[-20:] if st.session_state.logs else [],  # Last 20 logs
                    "workflow_state": st.session_state.workflow_state
                }
                
                json_str = json.dumps(complete_report, indent=2)
                filename = f"investment_report_{company_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                st.download_button(
                    label="Download Complete Investment Report",
                    data=json_str,
                    file_name=filename,
                    mime="application/json",
                    key="download_complete_report"
                )
        
        with col2:
            if st.button("📊 Export Results Only", key="export_results_only"):
                # Export just the research results without metadata
                json_str = json.dumps(st.session_state.results, indent=2)
                
                # Generate filename based on available results
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"research_results_{timestamp}.json"
                
                st.download_button(
                    label="Download Results JSON",
                    data=json_str,
                    file_name=filename,
                    mime="application/json",
                    key="download_results_only"
                )
        
        # Summary statistics
        st.subheader("📈 Research Summary")
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        
        with summary_col1:
            total_components = 4  # company, founder, market, decision
            completed_components = len(available_results)
            st.metric("Completion Rate", f"{completed_components}/{total_components}", 
                     f"{(completed_components/total_components)*100:.0f}%")
        
        with summary_col2:
            if 'decision_support' in st.session_state.results:
                decision_data = st.session_state.results['decision_support']
                if isinstance(decision_data, dict):
                    # Try to extract recommendation
                    recommendation = "N/A"
                    if 'investment_research' in decision_data:
                        recommendation = decision_data['investment_research'].get('overall_summary_and_recommendation', 'N/A')
                    elif 'overall_summary_and_recommendation' in decision_data:
                        recommendation = decision_data.get('overall_summary_and_recommendation', 'N/A')
                    
                    st.metric("Investment Recommendation", recommendation)
                else:
                    st.metric("Investment Recommendation", "N/A")
            else:
                st.metric("Investment Recommendation", "Pending", "Run decision support")
        
        with summary_col3:
            if 'decision_support' in st.session_state.results:
                decision_data = st.session_state.results['decision_support']
                if isinstance(decision_data, dict):
                    # Try to extract confidence
                    confidence = None
                    if 'investment_research' in decision_data:
                        confidence = decision_data['investment_research'].get('confidence_score_overall')
                    elif 'confidence_score_overall' in decision_data:
                        confidence = decision_data.get('confidence_score_overall')
                    elif 'confidence_score' in decision_data:
                        confidence = decision_data.get('confidence_score')
                    elif 'confidence' in decision_data:
                        confidence = decision_data.get('confidence')
                    
                    # Handle None or invalid confidence scores
                    if confidence is not None and isinstance(confidence, (int, float)):
                        # If confidence is already a percentage (>1), don't multiply by 100
                        if confidence > 1:
                            st.metric("Confidence Score", f"{confidence:.0f}%")
                        else:
                            st.metric("Confidence Score", f"{confidence*100:.0f}%")
                    else:
                        st.metric("Confidence Score", "N/A")
                else:
                    st.metric("Confidence Score", "0%")
            else:
                st.metric("Confidence Score", "0%", "Run decision support")
    
    else:
        st.info("No research results available yet. Run an analysis to see the workflow in action.")
        st.write("**Next Steps:**")
        st.write("1. Configure your research parameters in the sidebar")
        st.write("2. Click '🚀 Run Full Analysis' or run individual agents")
        st.write("3. Monitor progress in this workflow visualization")
        st.write("4. Export your complete investment research report")
    
with tab2:
    st.header("Company Research")
    if 'company_research' in st.session_state.results:
        display_company_profile(st.session_state.results['company_research'])
        # Export button specific to company research
        if st.button("Export Company Data", key="export_company"):
            json_str = json.dumps(st.session_state.results['company_research'], indent=2)
            st.download_button(
                label="Download Company JSON",
                data=json_str,
                file_name="company_research.json",
                mime="application/json",
                key="download_company_json"
            )
    
with tab3:
    st.header("Founder Research")
    if 'founder_research' in st.session_state.results:
        display_founder_profiles(st.session_state.results['founder_research'])
        # Export button specific to founder research
        if st.button("Export Founder Data", key="export_founders"):
            json_str = json.dumps(st.session_state.results['founder_research'], indent=2)
            st.download_button(
                label="Download Founder JSON",
                data=json_str,
                file_name="founder_research.json",
                mime="application/json",
                key="download_founder_json"
            )
    
with tab4:
    st.header("Market Analysis")
    if 'market_research' in st.session_state.results:
        display_market_analysis(st.session_state.results['market_research'])
        # Export button specific to market research
        if st.button("Export Market Data", key="export_market"):
            json_str = json.dumps(st.session_state.results['market_research'], indent=2)
            st.download_button(
                label="Download Market JSON",
                data=json_str,
                file_name="market_research.json",
                mime="application/json",
                key="download_market_json"
            )
    
with tab5:
    st.header("Investment Decision")
    if 'decision_support' in st.session_state.results:
        display_investment_decision(st.session_state.results['decision_support'])
        # Export button specific to decision support
        if st.button("Export Decision", key="export_decision"):
            json_str = json.dumps(st.session_state.results['decision_support'], indent=2)
            st.download_button(
                label="Download Decision JSON",
                data=json_str,
                file_name="investment_decision.json",
                mime="application/json",
                key="download_decision_json"
            )
    
with tab6:
    st.header("📈 Interactive Visualizations")
    if st.session_state.results:
        # Import and display visualizations
        from ai_agents.ui.visualizations import display_all_visualizations
        display_all_visualizations(st.session_state.results)
    else:
        st.info("No analysis data available yet. Run the research agents to see interactive visualizations.")
        st.write("**Available Charts:**")
        st.write("🎯 **Investment Criteria Radar** - Shows how the opportunity scores against Flight Story's 6 criteria")
        st.write("📊 **Market Size Analysis** - Visual breakdown of TAM, SAM, and SOM")
        st.write("⚖️ **Risk vs Opportunity Matrix** - Investment position analysis")

with tab7:
    st.header("Execution Logs")
    for log in st.session_state.logs:
        if log['level'] == 'error':
            st.error(f"{log['timestamp']} - {log['message']}")
        else:
            st.info(f"{log['timestamp']} - {log['message']}") 