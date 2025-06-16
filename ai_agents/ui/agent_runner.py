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

logger = logging.getLogger(__name__)

class AgentRunner:
    def __init__(self):
        """Initialize the agent runner with logging and state management"""
        self.log_buffer = io.StringIO()
        self.log_handler = logging.StreamHandler(self.log_buffer)
        self.log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(self.log_handler)
        
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

    def parse_agent_output(self, output: str) -> Dict:
        """Parse and structure agent output"""
        try:
            # Try to parse as JSON first
            return json.loads(output)
        except json.JSONDecodeError:
            # If not JSON, structure as text output
            return {
                "status": "completed",
                "output_type": "text",
                "content": output.strip(),
                "timestamp": datetime.now().isoformat()
            }

    def _capture_output(self, func, *args, **kwargs):
        """Capture stdout/stderr during function execution"""
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            result = func(*args, **kwargs)
            return result
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            # Update session state with logs
            stdout_text = stdout_capture.getvalue()
            stderr_text = stderr_capture.getvalue()
            if stdout_text:
                st.session_state.logs.append(f"Output: {stdout_text}")
            if stderr_text:
                st.session_state.logs.append(f"Error: {stderr_text}")
    
    def _update_progress(self, agent_type: str, status: str, error: Optional[str] = None):
        """Update progress in session state"""
        if error:
            st.session_state.errors[agent_type] = error
        else:
            if agent_type in st.session_state.errors:
                del st.session_state.errors[agent_type]

    def run_company_research(self, deal_id: str = None, company_name: str = None) -> Dict:
        """Run company research using the actual agent"""
        try:
            result = self._capture_output(
                run_company_research_cli,
                deal_id=deal_id,
                company_name=company_name
            )
            if result:
                st.session_state.results['company_research'] = result
                self._update_progress('company_research', 'complete')
                return result
            else:
                error_msg = "Company research failed to return results"
                self._update_progress('company_research', 'error', error_msg)
                return {"error": error_msg}
        except Exception as e:
            error_msg = f"Error in company research: {str(e)}"
            self._update_progress('company_research', 'error', error_msg)
            return {"error": error_msg}

    def run_founder_research(self, deal_id: str = None, founder_name: str = None, linkedin_url: str = None) -> Dict:
        """Run founder research using the actual agent"""
        try:
            result = self._capture_output(
                run_founder_research_cli_entrypoint,
                founder_name=founder_name,
                linkedin_url=linkedin_url
            )
            if result:
                st.session_state.results['founder_research'] = result
                self._update_progress('founder_research', 'complete')
                return result
            else:
                error_msg = "Founder research failed to return results"
                self._update_progress('founder_research', 'error', error_msg)
                return {"error": error_msg}
        except Exception as e:
            error_msg = f"Error in founder research: {str(e)}"
            self._update_progress('founder_research', 'error', error_msg)
            return {"error": error_msg}

    def run_market_research(self, deal_id: str = None, sector: str = None) -> Dict:
        """Run market research using the actual agent"""
        try:
            result = self._capture_output(
                run_market_intelligence_cli,
                market_or_industry=sector
            )
            if result:
                st.session_state.results['market_research'] = result
                self._update_progress('market_research', 'complete')
                return result
            else:
                error_msg = "Market research failed to return results"
                self._update_progress('market_research', 'error', error_msg)
                return {"error": error_msg}
        except Exception as e:
            error_msg = f"Error in market research: {str(e)}"
            self._update_progress('market_research', 'error', error_msg)
            return {"error": error_msg}

    def run_decision_support(self, deal_id: str = None, company_data: Dict = None, 
                           founder_data: List[Dict] = None, market_data: Dict = None) -> Dict:
        """Run decision support analysis using the actual agent"""
        try:
            result = self._capture_output(
                run_decision_support_analysis,
                company_data_dict=company_data,
                founder_data_list=founder_data,
                market_data_dict=market_data
            )
            if result:
                st.session_state.results['decision_support'] = result
                self._update_progress('decision_support', 'complete')
                return result
            else:
                error_msg = "Decision support analysis failed to return results"
                self._update_progress('decision_support', 'error', error_msg)
                return {"error": error_msg}
        except Exception as e:
            error_msg = f"Error in decision support analysis: {str(e)}"
            self._update_progress('decision_support', 'error', error_msg)
            return {"error": error_msg}

    def run_full_pipeline(self, deal_id: str, company_name: str, founder_name: str = None, 
                         linkedin_url: str = None, sector: str = None) -> Dict:
        """Run the complete research pipeline"""
        try:
            # Run company research
            company_result = self.run_company_research(deal_id, company_name)
            if "error" in company_result:
                return company_result
            
            # Run founder research if founder info is provided
            founder_result = None
            if founder_name:
                founder_result = self.run_founder_research(deal_id, founder_name, linkedin_url)
                if "error" in founder_result:
                    return founder_result
            
            # Run market research if sector is provided
            market_result = None
            if sector:
                market_result = self.run_market_research(deal_id, sector)
                if "error" in market_result:
                    return market_result
            
            # Run decision support with all available data
            decision_result = self.run_decision_support(
                deal_id=deal_id,
                company_data=company_result,
                founder_data=[founder_result] if founder_result else None,
                market_data=market_result
            )
            
            return decision_result
            
        except Exception as e:
            error_msg = f"Error in full pipeline execution: {str(e)}"
            self._update_progress('pipeline', 'error', error_msg)
            return {"error": error_msg}

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