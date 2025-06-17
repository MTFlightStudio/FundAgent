import streamlit as st
import json
import logging
import sys
import io
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import traceback
from contextlib import contextmanager

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ai_agents.agents import (
    company_research_agent,
    founder_research_agent,
    market_intelligence_agent,
    decision_support_agent,
    investment_orchestrator
)
from ai_agents.ui.workflow_visualizer import AgentStatus, update_workflow_status

# Import the actual agents
from ai_agents.agents.company_research_agent import run_company_research_cli
from ai_agents.agents.founder_research_agent import run_founder_research_cli_entrypoint
from ai_agents.agents.market_intelligence_agent import run_market_intelligence_cli
from ai_agents.agents.decision_support_agent import run_decision_support_analysis
from ai_agents.services import hubspot_client

logger = logging.getLogger(__name__)

class AgentRunner:
    """Manages the execution of AI agents for investment research"""
    
    def __init__(self):
        """Initialize the agent runner with logging and state management"""
        self.log_buffer = io.StringIO()
        self.log_handler = logging.StreamHandler(self.log_buffer)
        self.log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(self.log_handler)
        
        self.hubspot_available = hubspot_client is not None
        
        # Initialize session state if needed
        if 'agent_logs' not in st.session_state:
            st.session_state.agent_logs = []
        if 'last_error' not in st.session_state:
            st.session_state.last_error = None
        if 'results' not in st.session_state:
            st.session_state.results = {}
        if 'logs' not in st.session_state:
            st.session_state.logs = []
        if 'errors' not in st.session_state:
            st.session_state.errors = {}

    def _extract_company_info_from_hubspot(self, deal_id: str) -> Dict[str, Any]:
        """Extract company information from HubSpot deal data"""
        if not self.hubspot_available:
            raise ValueError("HubSpot client not available")
            
        try:
            deal_data = hubspot_client.get_deal_with_associated_data(deal_id)
            
            # Extract company name
            company_name = None
            if deal_data.get('associated_companies'):
                company_props = deal_data['associated_companies'][0].get('properties', {})
                company_name = company_props.get('name')
            
            # Extract industry/sector
            industry = None
            if deal_data.get('associated_companies'):
                company_props = deal_data['associated_companies'][0].get('properties', {})
                industry = company_props.get('what_sector_is_your_business_product_', 
                                            company_props.get('industry'))
            
            # Extract founder information
            founders = []
            if deal_data.get('associated_contacts'):
                for contact in deal_data['associated_contacts']:
                    props = contact.get('properties', {})
                    first_name = props.get('firstname', '')
                    last_name = props.get('lastname', '')
                    linkedin_url = props.get('hs_linkedin_url', '')
                    
                    if first_name and last_name:
                        founders.append({
                            'name': f"{first_name} {last_name}",
                            'linkedin_url': linkedin_url
                        })
            
            return {
                'company_name': company_name,
                'industry': industry,
                'founders': founders,
                'deal_data': deal_data
            }
            
        except Exception as e:
            logger.error(f"Error extracting HubSpot data for deal {deal_id}: {e}")
            raise

    @contextmanager
    def capture_output(self):
        """Context manager to capture stdout/stderr during agent execution"""
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        captured_output = io.StringIO()
        sys.stdout = captured_output
        sys.stderr = captured_output
        try:
            yield captured_output
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def update_progress(self, agent_name: str, status: AgentStatus, progress: float = None, results: Dict = None):
        """Update the workflow visualization and progress"""
        update_workflow_status(agent_name, status, results)
        if progress is not None:
            st.session_state.progress = progress

    def log_execution(self, message: str, level: str = "INFO"):
        """Log execution details and update session state"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        logger.info(log_entry)
        st.session_state.agent_logs.append(log_entry)

    def handle_error(self, error: Exception, agent_name: str):
        """Handle and log errors during agent execution"""
        error_msg = f"Error in {agent_name}: {str(error)}"
        self.log_execution(error_msg, "ERROR")
        st.session_state.last_error = {
            "agent": agent_name,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat()
        }
        self.update_progress(agent_name, AgentStatus.FAILED)

    def run_company_research(self, company_name: str = None, deal_id: str = None) -> Dict[str, Any]:
        """Run company research agent"""
        try:
            # If deal_id provided, extract company info from HubSpot
            if deal_id and not company_name:
                hubspot_info = self._extract_company_info_from_hubspot(deal_id)
                company_name = hubspot_info.get('company_name')
                
                if not company_name:
                    raise ValueError(f"Could not extract company name from deal {deal_id}")
            
            if not company_name:
                raise ValueError("Company name is required")
            
            # Run the company research agent
            logger.info(f"Running company research for: {company_name}")
            result = run_company_research_cli(company_name)
            
            if result:
                # Convert Pydantic model to dict
                company_data = {
                    'status': 'success',
                    'company_profile': result.model_dump(mode='json') if hasattr(result, 'model_dump') else result
                }
                st.session_state.results['company_research'] = company_data
                return company_data
            else:
                error_result = {
                    'status': 'error',
                    'error': 'Company research returned no results'
                }
                st.session_state.results['company_research'] = error_result
                return error_result
                
        except Exception as e:
            logger.error(f"Company research failed: {e}")
            error_result = {
                'status': 'error',
                'error': str(e)
            }
            st.session_state.results['company_research'] = error_result
            return error_result
    
    def run_founder_research(self, founder_names: List[str] = None, linkedin_urls: List[str] = None, 
                           deal_id: str = None) -> Dict[str, Any]:
        """Run founder research agent"""
        try:
            # If deal_id provided, extract founder info from HubSpot
            if deal_id and not founder_names:
                hubspot_info = self._extract_company_info_from_hubspot(deal_id)
                founders_data = hubspot_info.get('founders', [])
                
                founder_names = [f['name'] for f in founders_data if f.get('name')]
                linkedin_urls = [f['linkedin_url'] for f in founders_data if f.get('linkedin_url')]
            
            if not founder_names:
                raise ValueError("At least one founder name is required")
            
            # Ensure lists are same length
            if not linkedin_urls:
                linkedin_urls = [''] * len(founder_names)
            elif len(linkedin_urls) < len(founder_names):
                linkedin_urls.extend([''] * (len(founder_names) - len(linkedin_urls)))
            
            # Run research for each founder
            founder_profiles = []
            for name, linkedin_url in zip(founder_names, linkedin_urls):
                if name and name.strip():
                    logger.info(f"Running founder research for: {name}")
                    result = run_founder_research_cli_entrypoint(name, linkedin_url if linkedin_url else None)
                    
                    if result:
                        # Convert Pydantic model to dict
                        founder_dict = result.model_dump(mode='json') if hasattr(result, 'model_dump') else result
                        founder_profiles.append(founder_dict)
            
            founder_data = {
                'status': 'success',
                'founders': founder_profiles
            }
            st.session_state.results['founder_research'] = founder_data
            return founder_data
                
        except Exception as e:
            logger.error(f"Founder research failed: {e}")
            error_result = {
                'status': 'error',
                'error': str(e),
                'founders': []
            }
            st.session_state.results['founder_research'] = error_result
            return error_result
    
    def run_market_research(self, company_name: str = None, industry: str = None, 
                          deal_id: str = None) -> Dict[str, Any]:
        """Run market research agent"""
        try:
            # If deal_id provided, extract industry from HubSpot
            if deal_id and not industry:
                hubspot_info = self._extract_company_info_from_hubspot(deal_id)
                industry = hubspot_info.get('industry')
                
                if not industry:
                    # Try to infer from company name if available
                    if not company_name:
                        company_name = hubspot_info.get('company_name')
                    
                    if company_name:
                        industry = f"{company_name} industry"  # Fallback
                    else:
                        raise ValueError(f"Could not extract industry from deal {deal_id}")
            
            if not industry:
                raise ValueError("Industry/sector is required for market research")
            
            # Run the market intelligence agent
            logger.info(f"Running market research for sector: {industry}")
            result = run_market_intelligence_cli(industry)
            
            if result:
                # Convert Pydantic model to dict
                market_data = {
                    'status': 'success',
                    'market_analysis': result.model_dump(mode='json') if hasattr(result, 'model_dump') else result
                }
                st.session_state.results['market_research'] = market_data
                return market_data
            else:
                error_result = {
                    'status': 'error',
                    'error': 'Market research returned no results'
                }
                st.session_state.results['market_research'] = error_result
                return error_result
                
        except Exception as e:
            logger.error(f"Market research failed: {e}")
            error_result = {
                'status': 'error',
                'error': str(e)
            }
            st.session_state.results['market_research'] = error_result
            return error_result
    
    def run_decision_support(self, company_research: Dict[str, Any] = None,
                           founder_research: Dict[str, Any] = None,
                           market_research: Dict[str, Any] = None,
                           deal_id: str = None) -> Dict[str, Any]:
        """Run decision support agent"""
        try:
            # Extract the actual data from the research results
            company_data = None
            if company_research and company_research.get('status') == 'success':
                company_data = company_research.get('company_profile')
            
            founder_data = []
            if founder_research and founder_research.get('status') == 'success':
                founder_data = founder_research.get('founders', [])
            
            market_data = None
            if market_research and market_research.get('status') == 'success':
                market_data = market_research.get('market_analysis')
            
            # Get additional context from HubSpot if deal_id provided
            additional_context = {}
            if deal_id and self.hubspot_available:
                try:
                    deal_data = hubspot_client.get_deal_with_associated_data(deal_id)
                    additional_context['hubspot_deal_data'] = deal_data
                except Exception as e:
                    logger.warning(f"Could not fetch HubSpot data for context: {e}")
            
            # Run the decision support analysis
            logger.info("Running decision support analysis...")
            result = run_decision_support_analysis(
                company_data_dict=company_data,
                founder_data_list=founder_data,
                market_data_dict=market_data,
                additional_context=additional_context
            )
            
            if result:
                # Convert Pydantic model to dict
                decision_data = {
                    'status': 'success',
                    'investment_research': result.model_dump(mode='json') if hasattr(result, 'model_dump') else result
                }
                st.session_state.results['decision_support'] = decision_data
                return decision_data
            else:
                error_result = {
                    'status': 'error',
                    'error': 'Decision support analysis returned no results'
                }
                st.session_state.results['decision_support'] = error_result
                return error_result
                
        except Exception as e:
            logger.error(f"Decision support analysis failed: {e}")
            error_result = {
                'status': 'error',
                'error': str(e)
            }
            st.session_state.results['decision_support'] = error_result
            return error_result

    def run_full_pipeline(self, deal_id: str, company_name: str, founder_name: str = None, 
                         linkedin_url: str = None, sector: str = None) -> Dict:
        """Run the complete research pipeline"""
        try:
            self.log_execution("Starting full research pipeline...")
            
            # Run company research
            company_result = self.run_company_research(company_name, deal_id)
            if company_result.get("status") == "error":
                return company_result
            
            # Run founder research if founder info is provided
            founder_result = None
            if founder_name:
                founder_names = [founder_name] if isinstance(founder_name, str) else founder_name
                linkedin_urls = [linkedin_url] if isinstance(linkedin_url, str) else linkedin_url
                founder_result = self.run_founder_research(founder_names, linkedin_urls, deal_id)
                if founder_result.get("status") == "error":
                    return founder_result
            
            # Run market research if sector is provided
            market_result = None
            if sector:
                market_result = self.run_market_research(company_name, sector, deal_id)
                if market_result.get("status") == "error":
                    return market_result
            
            # Run decision support with all available data
            decision_result = self.run_decision_support(
                company_research=company_result,
                founder_research=founder_result,
                market_research=market_result,
                deal_id=deal_id
            )
            
            self.log_execution("Full research pipeline completed successfully")
            return decision_result
            
        except Exception as e:
            error_msg = f"Error in full pipeline execution: {str(e)}"
            self.log_execution(error_msg, "ERROR")
            return {"status": "error", "error": error_msg}

# Initialize the runner
_runner = None

def initialize_runner():
    """Initialize the global runner instance"""
    global _runner
    if _runner is None:
        _runner = AgentRunner()
    return _runner

def get_runner():
    """Get the global runner instance"""
    if _runner is None:
        return initialize_runner()
    return _runner

# Example usage in streamlit_app.py:
"""
from ai_agents.ui.agent_runner import initialize_runner, get_runner

# In your main app:
initialize_runner()

# When running an agent:
runner = get_runner()
results = runner.run_company_research(params)

# Or run the full pipeline:
results = runner.run_full_pipeline(params)
""" 