import streamlit as st
from typing import Dict, Any
from enum import Enum
from streamlit_mermaid import st_mermaid

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
        
        # Build the basic diagram structure
        mermaid_code = """graph TD
    Start["🚀 Investment Research Pipeline"] --> Input["📋 Deal Input"]
    
    Input --> CompanyAgent["🏢 Company Research Agent"]
    Input --> FounderAgent["👤 Founder Research Agent"]  
    Input --> MarketAgent["🌍 Market Intelligence Agent"]
    
    CompanyAgent --> CompanyData["📊 Company Profile"]
    FounderAgent --> FounderData["👥 Founder Profiles"]
    MarketAgent --> MarketData["📈 Market Analysis"]
    
    CompanyData --> DecisionAgent["🎯 Decision Support Agent"]
    FounderData --> DecisionAgent
    MarketData --> DecisionAgent
    
    DecisionAgent --> Analysis["⚖️ Flight Story Criteria Analysis"]
    
    Analysis --> Result["📋 Investment Recommendation"]
    
    classDef completed fill:#90EE90,stroke:#2E8B57,stroke-width:3px
    classDef running fill:#FFD700,stroke:#FF8C00,stroke-width:3px,color:#000
    classDef error fill:#FFB6C1,stroke:#DC143C,stroke-width:3px
    classDef pending fill:#E6E6FA,stroke:#4B0082,stroke-width:2px
    
    classDef data fill:#E0F2F1,stroke:#00695C,stroke-width:2px
    classDef process fill:#FFF3E0,stroke:#E65100,stroke-width:2px
    
    class CompanyData,FounderData,MarketData data
    class Analysis process"""
    
        # Add status-based styling for agents
        if company_status != 'pending':
            mermaid_code += f"\n    class CompanyAgent {company_status}"
        if founder_status != 'pending':
            mermaid_code += f"\n    class FounderAgent {founder_status}"
        if market_status != 'pending':
            mermaid_code += f"\n    class MarketAgent {market_status}"
        if decision_status != 'pending':
            mermaid_code += f"\n    class DecisionAgent {decision_status}"
        
        return mermaid_code
    
    def generate_simple_progress_diagram(self) -> str:
        """Generate a simpler progress-focused diagram"""
        
        # Get completion status
        agents = ['company_research', 'founder_research', 'market_research', 'decision_support']
        completed = sum(1 for agent in agents if self.workflow_state.get(agent, {}).get('status') == 'completed')
        total = len(agents)
        
        mermaid_code = """graph LR
    A["📋 Input"] --> B["🏢 Company"]
    B --> C["👤 Founders"]
    C --> D["🌍 Market"]
    D --> E["🎯 Decision"]
    E --> F["📊 Result"]
    
    classDef completed fill:#90EE90,stroke:#2E8B57,stroke-width:3px
    classDef running fill:#FFD700,stroke:#FF8C00,stroke-width:3px
    classDef pending fill:#E6E6FA,stroke:#4B0082,stroke-width:2px"""
        
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

def display_workflow(view_mode="detailed"):
    """Display the workflow visualization
    
    Args:
        view_mode: "detailed" for full workflow diagram, "simple" for progress-only view
    """
    try:
        if 'workflow_visualizer' not in st.session_state:
            st.warning("⚠️ Workflow visualizer not initialized. Initializing now...")
            st.session_state.workflow_visualizer = WorkflowVisualizer()
            st.info("✅ Workflow visualizer initialized")
        
        # Add view mode selector
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("🔄 Investment Research Workflow")
        
        with col2:
            view_option = st.selectbox(
                "View Mode:",
                ["Detailed", "Simple Progress"],
                index=0 if view_mode == "detailed" else 1,
                key="workflow_view_mode"
            )
        
        # Generate appropriate diagram based on view mode
        try:
            if view_option == "Simple Progress":
                diagram = st.session_state.workflow_visualizer.generate_simple_progress_diagram()
                diagram_height = "300px"
            else:
                diagram = st.session_state.workflow_visualizer.generate_mermaid_diagram()
                diagram_height = "600px"
            
            # Check if diagram is empty or invalid
            if not diagram or len(diagram.strip()) < 10:
                st.error("❌ Generated diagram is empty or too short")
                return
            
            # Add debug option in an expander to keep interface clean
            with st.expander("🔧 Debug Options (click to expand)", expanded=False):
                st.write(f"Diagram length: {len(diagram)} characters")
                
                if st.checkbox("🧪 Test with simple diagram", key="test_simple_diagram"):
                    test_diagram = """graph TD
    A[Start] --> B[Process]
    B --> C[End]
    classDef default fill:#f9f9f9"""
                    st_mermaid(test_diagram, height="200px")
                    st.write("✅ Simple diagram test completed")
                    return
                
                if st.checkbox("🔍 Show diagram source", key="show_diagram_source"):
                    st.code(diagram[:500] + "..." if len(diagram) > 500 else diagram)
            
            # Render the Mermaid diagram
            st_mermaid(diagram, height=diagram_height)
            
        except Exception as diagram_error:
            st.error(f"❌ Error generating/rendering diagram: {diagram_error}")
            st.write("**Falling back to text display:**")
            st.code(diagram if 'diagram' in locals() else "No diagram generated")
            st.write("**Debug - Workflow State:**")
            st.json(st.session_state.workflow_visualizer.workflow_state)
        
        # Display workflow status metrics  
        st.session_state.workflow_visualizer.display_workflow_status()
        
    except Exception as e:
        st.error(f"❌ Critical error in display_workflow: {e}")
        st.write("**Debug Information:**")
        st.write(f"- Session state keys: {list(st.session_state.keys())}")
        st.write(f"- Error type: {type(e).__name__}")
        st.write(f"- Error details: {str(e)}")
        
        # Try to provide a fallback
        st.write("**Attempting fallback display...**")
        try:
            st.text("Basic workflow: Input → Company Research → Founder Research → Market Research → Decision Support → Output")
        except:
            st.text("Unable to display any workflow visualization")

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