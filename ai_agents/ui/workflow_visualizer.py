import streamlit as st
from typing import Dict, Optional, List
from enum import Enum
import json
from datetime import datetime

class AgentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class WorkflowNode:
    def __init__(self, id: str, label: str, status: AgentStatus = AgentStatus.PENDING):
        self.id = id
        self.label = label
        self.status = status
        self.results = None
        self.last_updated = None

class WorkflowVisualizer:
    def __init__(self):
        self.nodes = {
            "hubspot": WorkflowNode("hubspot", "HubSpot Data Fetch"),
            "pdf_extract": WorkflowNode("pdf_extract", "PDF Extraction"),
            "company_research": WorkflowNode("company_research", "Company Research"),
            "founder_research": WorkflowNode("founder_research", "Founder Research"),
            "market_research": WorkflowNode("market_research", "Market Research"),
            "decision_support": WorkflowNode("decision_support", "Decision Support"),
            "final_report": WorkflowNode("final_report", "Final Report")
        }
        
        # Define the workflow connections
        self.connections = [
            ("hubspot", "pdf_extract"),
            ("pdf_extract", "company_research"),
            ("pdf_extract", "founder_research"),
            ("pdf_extract", "market_research"),
            ("company_research", "decision_support"),
            ("founder_research", "decision_support"),
            ("market_research", "decision_support"),
            ("decision_support", "final_report")
        ]

    def update_node_status(self, node_id: str, status: AgentStatus, results: Optional[Dict] = None):
        """Update the status and results of a workflow node"""
        if node_id in self.nodes:
            self.nodes[node_id].status = status
            if results is not None:
                self.nodes[node_id].results = results
            self.nodes[node_id].last_updated = datetime.now()

    def get_node_color(self, status: AgentStatus) -> str:
        """Get the color for a node based on its status"""
        colors = {
            AgentStatus.PENDING: "#808080",  # Gray
            AgentStatus.RUNNING: "#FFA500",  # Orange
            AgentStatus.COMPLETED: "#00FF00",  # Green
            AgentStatus.FAILED: "#FF0000"   # Red
        }
        return colors.get(status, "#808080")

    def generate_mermaid_diagram(self) -> str:
        """Generate a Mermaid diagram representation of the workflow"""
        diagram = ["```mermaid", "graph TD"]
        
        # Add nodes with styling
        for node_id, node in self.nodes.items():
            color = self.get_node_color(node.status)
            diagram.append(f"    {node_id}[\"{node.label}\"]:::status_{node.status.value}")
        
        # Add connections
        for source, target in self.connections:
            diagram.append(f"    {source} --> {target}")
        
        # Add styling
        diagram.append("    classDef status_pending fill:#808080,stroke:#333,stroke-width:2px")
        diagram.append("    classDef status_running fill:#FFA500,stroke:#333,stroke-width:2px")
        diagram.append("    classDef status_completed fill:#00FF00,stroke:#333,stroke-width:2px")
        diagram.append("    classDef status_failed fill:#FF0000,stroke:#333,stroke-width:2px")
        diagram.append("```")
        
        return "\n".join(diagram)

    def display_workflow(self):
        """Display the workflow diagram and handle interactions"""
        st.markdown("### Research Workflow")
        
        # Generate and display the Mermaid diagram
        diagram = self.generate_mermaid_diagram()
        st.markdown(diagram)
        
        # Display node details when clicked
        st.markdown("### Node Details")
        selected_node = st.selectbox(
            "Select a node to view details",
            options=list(self.nodes.keys()),
            format_func=lambda x: self.nodes[x].label
        )
        
        if selected_node:
            node = self.nodes[selected_node]
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Status", node.status.value.title())
                if node.last_updated:
                    st.metric("Last Updated", node.last_updated.strftime("%Y-%m-%d %H:%M:%S"))
            
            with col2:
                if node.results:
                    st.json(node.results)
                else:
                    st.info("No results available for this node")

def initialize_workflow():
    """Initialize the workflow visualizer in the session state"""
    if 'workflow' not in st.session_state:
        st.session_state.workflow = WorkflowVisualizer()

def update_workflow_status(node_id: str, status: AgentStatus, results: Optional[Dict] = None):
    """Update the status of a workflow node"""
    if 'workflow' in st.session_state:
        st.session_state.workflow.update_node_status(node_id, status, results)

def display_workflow():
    """Display the workflow visualization"""
    if 'workflow' in st.session_state:
        st.session_state.workflow.display_workflow()
    else:
        st.error("Workflow not initialized")

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