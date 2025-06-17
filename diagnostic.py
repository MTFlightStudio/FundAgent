#!/usr/bin/env python3
"""
Diagnostic script to test Streamlit AI Agent integration
Run this before starting the Streamlit app to identify issues
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_status(message: str, status: str = "info"):
    """Print colored status messages"""
    if status == "success":
        print(f"{GREEN}✓ {message}{RESET}")
    elif status == "error":
        print(f"{RED}✗ {message}{RESET}")
    elif status == "warning":
        print(f"{YELLOW}⚠ {message}{RESET}")
    else:
        print(f"{BLUE}ℹ {message}{RESET}")

def test_environment():
    """Test environment variables"""
    print("\n=== Testing Environment Variables ===")
    
    required_vars = {
        "HUBSPOT_ACCESS_TOKEN": "HubSpot integration",
        "TAVILY_API_KEY": "Web search functionality",
        "RELEVANCE_AI_API_KEY": "LinkedIn research"
    }
    
    optional_vars = {
        "OPENAI_API_KEY": "OpenAI LLM",
        "ANTHROPIC_API_KEY": "Anthropic Claude LLM",
        "RELEVANCE_AI_STUDIO_ID": "Relevance AI configuration",
        "RELEVANCE_AI_PROJECT_ID": "Relevance AI configuration"
    }
    
    # Check required
    all_required_present = True
    for var, purpose in required_vars.items():
        if os.getenv(var):
            print_status(f"{var} is set ({purpose})", "success")
        else:
            print_status(f"{var} is MISSING - Required for {purpose}", "error")
            all_required_present = False
    
    # Check optional
    for var, purpose in optional_vars.items():
        if os.getenv(var):
            print_status(f"{var} is set ({purpose})", "success")
        else:
            print_status(f"{var} not set - {purpose}", "warning")
    
    # Check at least one LLM
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print_status("No LLM API key found (need OPENAI_API_KEY or ANTHROPIC_API_KEY)", "error")
        all_required_present = False
    
    return all_required_present

def test_imports():
    """Test all required imports"""
    print("\n=== Testing Imports ===")
    
    imports_to_test = [
        ("ai_agents.ui.agent_runner", "AgentRunner"),
        ("ai_agents.ui.results_display", "display_results"),
        ("ai_agents.ui.workflow_visualizer", "WorkflowVisualizer"),
        ("ai_agents.services.hubspot_client", "get_deal_with_associated_data"),
        ("ai_agents.agents.company_research_agent", "run_company_research_cli"),
        ("ai_agents.agents.founder_research_agent", "run_founder_research_cli_entrypoint"),
        ("ai_agents.agents.market_intelligence_agent", "run_market_intelligence_cli"),
        ("ai_agents.agents.decision_support_agent", "run_decision_support_analysis")
    ]
    
    all_imports_ok = True
    for module_name, attr_name in imports_to_test:
        try:
            module = __import__(module_name, fromlist=[attr_name])
            if hasattr(module, attr_name):
                print_status(f"Import {module_name}.{attr_name}", "success")
            else:
                print_status(f"Module {module_name} missing attribute {attr_name}", "error")
                all_imports_ok = False
        except Exception as e:
            print_status(f"Failed to import {module_name}: {str(e)}", "error")
            all_imports_ok = False
    
    return all_imports_ok

def test_hubspot_connection(deal_id: str = "227710582988"):
    """Test HubSpot connection and data extraction"""
    print(f"\n=== Testing HubSpot Connection (Deal: {deal_id}) ===")
    
    try:
        from ai_agents.services import hubspot_client
        
        if not hubspot_client:
            print_status("HubSpot client not available", "error")
            return False
        
        # Test basic connection
        deal_data = hubspot_client.get_deal_with_associated_data(deal_id)
        
        if not deal_data:
            print_status(f"No data returned for deal {deal_id}", "error")
            return False
        
        print_status(f"Successfully fetched deal data", "success")
        
        # Check structure
        if deal_data.get('deal'):
            print_status(f"Deal name: {deal_data['deal'].get('dealname', 'N/A')}", "info")
        
        if deal_data.get('associated_companies'):
            company = deal_data['associated_companies'][0]['properties']
            print_status(f"Company: {company.get('name', 'N/A')}", "info")
            print_status(f"Industry: {company.get('what_sector_is_your_business_product_', company.get('industry', 'N/A'))}", "info")
        else:
            print_status("No associated companies found", "warning")
        
        if deal_data.get('associated_contacts'):
            print_status(f"Found {len(deal_data['associated_contacts'])} contacts", "info")
            for contact in deal_data['associated_contacts'][:2]:  # Show first 2
                props = contact.get('properties', {})
                name = f"{props.get('firstname', '')} {props.get('lastname', '')}".strip()
                print_status(f"  Contact: {name or 'N/A'} - LinkedIn: {props.get('hs_linkedin_url', 'N/A')}", "info")
        else:
            print_status("No associated contacts found", "warning")
        
        return True
        
    except Exception as e:
        print_status(f"HubSpot test failed: {str(e)}", "error")
        return False

def test_agent_runner():
    """Test AgentRunner initialization and basic methods"""
    print("\n=== Testing AgentRunner ===")
    
    try:
        from ai_agents.ui.agent_runner import AgentRunner
        
        runner = AgentRunner()
        print_status("AgentRunner initialized successfully", "success")
        
        # Test data extraction
        if os.getenv("HUBSPOT_ACCESS_TOKEN"):
            try:
                test_deal = "227710582988"
                extracted = runner._extract_company_info_from_hubspot(test_deal)
                
                print_status(f"Company extraction: {extracted.get('company_name', 'N/A')}", "info")
                print_status(f"Industry extraction: {extracted.get('industry', 'N/A')}", "info")
                print_status(f"Founders extraction: {len(extracted.get('founders', []))} found", "info")
                
            except Exception as e:
                print_status(f"Data extraction test failed: {str(e)}", "warning")
        
        return True
        
    except Exception as e:
        print_status(f"AgentRunner test failed: {str(e)}", "error")
        return False

def test_individual_agent(agent_name: str, test_func):
    """Test an individual agent with minimal data"""
    print(f"\n=== Testing {agent_name} ===")
    
    try:
        result = test_func()
        if result:
            print_status(f"{agent_name} returned result", "success")
            print_status(f"Result type: {type(result)}", "info")
            return True
        else:
            print_status(f"{agent_name} returned None", "error")
            return False
    except Exception as e:
        print_status(f"{agent_name} test failed: {str(e)}", "error")
        return False

def test_file_structure():
    """Test that all required files exist"""
    print("\n=== Testing File Structure ===")
    
    required_files = [
        "streamlit_app.py",
        "ai_agents/__init__.py",
        "ai_agents/ui/__init__.py",
        "ai_agents/ui/agent_runner.py",
        "ai_agents/ui/results_display.py",
        "ai_agents/ui/workflow_visualizer.py",
        "ai_agents/services/hubspot_client.py",
        "ai_agents/agents/company_research_agent.py",
        "ai_agents/agents/founder_research_agent.py",
        "ai_agents/agents/market_intelligence_agent.py",
        "ai_agents/agents/decision_support_agent.py"
    ]
    
    all_files_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print_status(f"Found {file_path}", "success")
        else:
            print_status(f"Missing {file_path}", "error")
            all_files_exist = False
    
    return all_files_exist

def test_streamlit_imports():
    """Test Streamlit-specific imports"""
    print("\n=== Testing Streamlit Dependencies ===")
    
    streamlit_deps = [
        ("streamlit", "Main Streamlit library"),
        ("pandas", "Data manipulation"),
        ("plotly", "Plotting library"),
        ("pydantic", "Data validation")
    ]
    
    all_deps_ok = True
    for module_name, purpose in streamlit_deps:
        try:
            __import__(module_name)
            print_status(f"{module_name} - {purpose}", "success")
        except ImportError:
            print_status(f"Missing {module_name} - {purpose}", "error")
            all_deps_ok = False
    
    return all_deps_ok

def run_diagnostics():
    """Run all diagnostic tests"""
    print(f"{BLUE}{'='*60}")
    print("AI Agents Streamlit Integration Diagnostics")
    print(f"{'='*60}{RESET}")
    
    # Track overall status
    all_tests_passed = True
    
    # Test file structure first
    if not test_file_structure():
        all_tests_passed = False
        print_status("\nCannot continue - fix missing files first", "error")
        return False
    
    # Test Streamlit dependencies
    if not test_streamlit_imports():
        all_tests_passed = False
        print_status("\nInstall missing dependencies: pip install streamlit pandas plotly pydantic", "error")
    
    # Test environment
    if not test_environment():
        all_tests_passed = False
    
    # Test imports
    if not test_imports():
        all_tests_passed = False
        print_status("\nCannot continue - fix import errors first", "error")
        return False
    
    # Test HubSpot
    if os.getenv("HUBSPOT_ACCESS_TOKEN"):
        if not test_hubspot_connection():
            all_tests_passed = False
    else:
        print_status("\nSkipping HubSpot tests - no access token", "warning")
    
    # Test AgentRunner
    if not test_agent_runner():
        all_tests_passed = False
    
    # Test individual agents with minimal data
    print("\n=== Testing Individual Agents (Minimal) ===")
    print_status("Note: These tests use simple test data and may take a few seconds each", "info")
    
    # Company research
    def test_company():
        from ai_agents.agents.company_research_agent import run_company_research_cli
        return run_company_research_cli("OpenAI")
    
    # Skip individual agent tests if API keys are missing
    if os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"):
        test_individual_agent("Company Research", test_company)
    else:
        print_status("Skipping agent tests - no LLM API keys", "warning")
    
    # Summary
    print(f"\n{BLUE}{'='*60}{RESET}")
    if all_tests_passed:
        print_status("All tests passed! You can run: streamlit run streamlit_app.py", "success")
    else:
        print_status("Some tests failed. Please fix the issues above before running Streamlit.", "error")
        print_status("\nCommon fixes:", "info")
        print("  1. Ensure all .py files from the artifacts are saved in the correct locations")
        print("  2. Check that all environment variables are set in .env")
        print("  3. Verify HubSpot access token has correct permissions")
        print("  4. Make sure at least one LLM API key is set")
        print("  5. Install missing dependencies: pip install streamlit pandas plotly pydantic")
    
    return all_tests_passed

if __name__ == "__main__":
    run_diagnostics() 