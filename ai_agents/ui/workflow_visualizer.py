import streamlit as st
from typing import Dict, Any
from enum import Enum

class AgentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class WorkflowVisualizer:
    """Visualizes the agent workflow state using Mermaid diagrams"""
    
    def __init__(self):
        self.workflow_state = {}
        
    def generate_mermaid_diagram(self) -> str:
        """Generate a Mermaid diagram showing the workflow state"""
        
        # Define the workflow structure
        mermaid_code = """
```mermaid
graph TD
    Start([Start Research]) --> Company[Company Research]
    Start --> Founders[Founder Research]
    Start --> Market[Market Research]
    
    Company --> Decision{Decision Support}
    Founders --> Decision
    Market --> Decision
    
    Decision --> Result([Investment Recommendation])
    
    %% Styling based on state
"""
        
        # Add state-based styling
        for agent_name, state in self.workflow_state.items():
            status = state.get('status', 'pending')
            
            if agent_name == 'company_research':
                node_name = 'Company'
            elif agent_name == 'founder_research':
                node_name = 'Founders'
            elif agent_name == 'market_research':
                node_name = 'Market'
            elif agent_name == 'decision_support':
                node_name = 'Decision'
            else:
                continue
            
            if status == 'completed':
                mermaid_code += f"    class {node_name} completed\n"
            elif status == 'running':
                mermaid_code += f"    class {node_name} running\n"
            elif status == 'error':
                mermaid_code += f"    class {node_name} error\n"
        
        # Add CSS classes
        mermaid_code += """
    classDef completed fill:#90EE90,stroke:#2E8B57,stroke-width:2px
    classDef running fill:#FFD700,stroke:#FF8C00,stroke-width:2px,color:#000
    classDef error fill:#FFB6C1,stroke:#DC143C,stroke-width:2px
    classDef default fill:#E6E6FA,stroke:#4B0082,stroke-width:1px
```
"""
        
        return mermaid_code
    
    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get a summary of the workflow state"""
        summary = {
            'total_agents': 4,
            'completed': 0,
            'running': 0,
            'error': 0,
            'pending': 0
        }
        
        for state in self.workflow_state.values():
            status = state.get('status', 'pending')
            if status in summary:
                summary[status] += 1
            else:
                summary['pending'] += 1
        
        summary['progress'] = (summary['completed'] / summary['total_agents']) * 100
        
        return summary
    
    def display_workflow_status(self):
        """Display workflow status metrics"""
        summary = self.get_workflow_summary()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Completed", summary['completed'], 
                     delta=f"{summary['completed']}/{summary['total_agents']}")
        
        with col2:
            st.metric("Running", summary['running'])
        
        with col3:
            st.metric("Errors", summary['error'])
        
        with col4:
            st.metric("Progress", f"{summary['progress']:.0f}%")
        
        # Progress bar
        st.progress(summary['progress'] / 100)

    def update_node_status(self, node_id: str, status: AgentStatus, results: Dict = None):
        """Update the status of a workflow node (for compatibility)"""
        self.workflow_state[node_id] = {
            'status': status.value if isinstance(status, AgentStatus) else status,
            'results': results,
            'timestamp': st.session_state.get('current_time', 'N/A')
        }

def initialize_workflow():
    """Initialize the workflow visualizer in the session state"""
    if 'workflow_visualizer' not in st.session_state:
        st.session_state.workflow_visualizer = WorkflowVisualizer()

def update_workflow_status(node_id: str, status: AgentStatus, results: Dict = None):
    """Update the status of a workflow node"""
    if 'workflow_visualizer' in st.session_state:
        st.session_state.workflow_visualizer.update_node_status(node_id, status, results)

def display_workflow():
    """Display the workflow visualization"""
    if 'workflow_visualizer' in st.session_state:
        # Display the Mermaid diagram
        diagram = st.session_state.workflow_visualizer.generate_mermaid_diagram()
        st.markdown(diagram)
        
        # Display workflow status metrics
        st.session_state.workflow_visualizer.display_workflow_status()
    else:
        st.error("Workflow visualizer not initialized")

# Example usage in streamlit_app.py:
"""
from ai_agents.ui.workflow_visualizer import initialize_workflow, update_workflow_status, display_workflow

# In your main app:
initialize_workflow()

# When starting an agent:
update_workflow_status("company_research", AgentStatus.RUNNING)

# When agent completes:
update_workflow_status("company_research", AgentStatus.COMPLETED, results={"data": "..."})

# To display the workflow:
display_workflow()
""" 