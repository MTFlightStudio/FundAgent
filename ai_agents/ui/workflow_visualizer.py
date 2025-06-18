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
        """Generate a comprehensive Mermaid diagram showing the investment research workflow"""
        
        # Get current workflow state for styling
        company_status = self.workflow_state.get('company_research', {}).get('status', 'pending')
        founder_status = self.workflow_state.get('founder_research', {}).get('status', 'pending')
        market_status = self.workflow_state.get('market_research', {}).get('status', 'pending')
        decision_status = self.workflow_state.get('decision_support', {}).get('status', 'pending')
        
        # Define the enhanced workflow structure
        mermaid_code = f"""```mermaid
graph TD
    Start([🚀 Investment Research Pipeline]) --> Input{{📋 Deal Input}}
    
    Input --> CompanyAgent[🏢 Company Research Agent]
    Input --> FounderAgent[👤 Founder Research Agent]  
    Input --> MarketAgent[🌍 Market Intelligence Agent]
    
    CompanyAgent --> CompanyData[📊 Company Profile<br/>• Business Model<br/>• Funding History<br/>• Key Metrics]
    FounderAgent --> FounderData[👥 Founder Profiles<br/>• Experience<br/>• Background<br/>• LinkedIn Analysis]
    MarketAgent --> MarketData[📈 Market Analysis<br/>• Market Size<br/>• Competition<br/>• Trends]
    
    CompanyData --> DecisionAgent[🎯 Decision Support Agent]
    FounderData --> DecisionAgent
    MarketData --> DecisionAgent
    
    DecisionAgent --> Analysis[⚖️ Flight Story Criteria Analysis<br/>• Focus Industry Fit<br/>• Mission Alignment<br/>• Exciting Solution<br/>• Founder Excellence<br/>• Market Timing<br/>• Scalable Business Model]
    
    Analysis --> Result{{📋 Investment Recommendation<br/>PASS / NO PASS}}
    
    %% Apply status-based styling
    class CompanyAgent {company_status}
    class FounderAgent {founder_status}
    class MarketAgent {market_status}
    class DecisionAgent {decision_status}
    
    classDef completed fill:#90EE90,stroke:#2E8B57,stroke-width:3px
    classDef running fill:#FFD700,stroke:#FF8C00,stroke-width:3px,color:#000
    classDef error fill:#FFB6C1,stroke:#DC143C,stroke-width:3px
    classDef pending fill:#E6E6FA,stroke:#4B0082,stroke-width:2px
    
    classDef data fill:#E0F2F1,stroke:#00695C,stroke-width:2px
    classDef process fill:#FFF3E0,stroke:#E65100,stroke-width:2px
    
    class CompanyData,FounderData,MarketData data
    class Analysis process
```"""
        
        return mermaid_code
    
    def generate_simple_progress_diagram(self) -> str:
        """Generate a simpler progress-focused diagram"""
        
        # Get completion status
        agents = ['company_research', 'founder_research', 'market_research', 'decision_support']
        completed = sum(1 for agent in agents if self.workflow_state.get(agent, {}).get('status') == 'completed')
        total = len(agents)
        
        mermaid_code = f"""```mermaid
graph LR
    A[📋 Input] --> B[🏢 Company]
    B --> C[👤 Founders]
    C --> D[🌍 Market]
    D --> E[🎯 Decision]
    E --> F[📊 Result]
    
    %% Progress: {completed}/{total} Complete
    classDef completed fill:#90EE90,stroke:#2E8B57,stroke-width:3px
    classDef running fill:#FFD700,stroke:#FF8C00,stroke-width:3px
    classDef pending fill:#E6E6FA,stroke:#4B0082,stroke-width:2px
```"""
        
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
        
        summary['progress'] = (summary['completed'] / summary['total_agents']) * 100 if summary['total_agents'] > 0 else 0
        
        return summary
    
    def display_workflow_status(self):
        """Display enhanced workflow status metrics"""
        summary = self.get_workflow_summary()
        
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("✅ Completed", summary['completed'], 
                     delta=f"{summary['completed']}/{summary['total_agents']}")
        
        with col2:
            st.metric("⏳ Running", summary['running'],
                     delta="Active" if summary['running'] > 0 else None)
        
        with col3:
            st.metric("❌ Errors", summary['error'],
                     delta="Issues" if summary['error'] > 0 else None)
        
        with col4:
            st.metric("📊 Progress", f"{summary['progress']:.0f}%")
        
        # Enhanced progress bar with color
        progress_value = summary['progress'] / 100
        
        if summary['error'] > 0:
            st.error(f"⚠️ {summary['error']} agent(s) encountered errors")
        elif summary['running'] > 0:
            st.info(f"🔄 {summary['running']} agent(s) currently running...")
        elif summary['completed'] == summary['total_agents']:
            st.success("🎉 All agents completed successfully!")
        
        st.progress(progress_value)
        
        # Detailed agent status
        if self.workflow_state:
            st.subheader("🔍 Agent Status Details")
            
            agent_names = {
                'company_research': '🏢 Company Research',
                'founder_research': '👤 Founder Research', 
                'market_research': '🌍 Market Research',
                'decision_support': '🎯 Decision Support'
            }
            
            for agent_id, display_name in agent_names.items():
                state = self.workflow_state.get(agent_id, {})
                status = state.get('status', 'pending')
                
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    status_emoji = {
                        'completed': '✅',
                        'running': '⏳', 
                        'error': '❌',
                        'pending': '⏸️'
                    }.get(status, '❓')
                    
                    st.write(f"{status_emoji} **{display_name}**")
                
                with col2:
                    st.write(status.title())
                
                with col3:
                    if 'timestamp' in state:
                        st.write(f"📅 {state['timestamp']}")

    def update_node_status(self, node_id: str, status: AgentStatus, results: Dict = None):
        """Update the status of a workflow node"""
        from datetime import datetime
        
        self.workflow_state[node_id] = {
            'status': status.value if isinstance(status, AgentStatus) else status,
            'results': results,
            'timestamp': datetime.now().strftime("%H:%M:%S")
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