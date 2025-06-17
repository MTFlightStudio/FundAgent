# Streamlit AI Agent Setup Guide

## Overview

This guide will help you fix the integration issues between your AI agents and the Streamlit UI.

## Step-by-Step Instructions

### 1. Update Directory Structure

Ensure your directory structure includes these UI files:

```
ai_agents/
├── ui/
│   ├── __init__.py
│   ├── agent_runner.py      # New file (provided)
│   ├── results_display.py   # Updated file (provided)
│   └── workflow_visualizer.py # New file (provided)
├── agents/
│   ├── company_research_agent.py
│   ├── founder_research_agent.py
│   ├── market_intelligence_agent.py
│   └── decision_support_agent.py
└── services/
    └── hubspot_client.py
```

### 2. Create Missing Files

Create `ai_agents/ui/__init__.py`:

```python
# This file makes 'ui' a sub-package of 'ai_agents'
```

### 3. Fix Import Issues

In your Cursor IDE, use this prompt to check and fix imports:

**Prompt for Cursor:**

```
Check all import statements in:
1. streamlit_app.py
2. ai_agents/ui/agent_runner.py
3. ai_agents/ui/results_display.py
4. ai_agents/ui/workflow_visualizer.py

Ensure:
- All imports use absolute paths from the project root
- The sys.path manipulation is correct
- All required modules are imported
```

### 4. Test Individual Components

Test each component separately before running the full app:

#### Test HubSpot Connection:

```python
# test_hubspot.py
from ai_agents.services import hubspot_client

deal_id = "227710582988"  # Your test deal
data = hubspot_client.get_deal_with_associated_data(deal_id)
print(f"Company: {data['associated_companies'][0]['properties']['name'] if data['associated_companies'] else 'None'}")
print(f"Contacts: {len(data['associated_contacts'])}")
```

#### Test Agent Runner:

```python
# test_agent_runner.py
from ai_agents.ui.agent_runner import AgentRunner

runner = AgentRunner()
# Test with a simple company name
result = runner.run_company_research(company_name="OpenAI")
print(f"Status: {result['status']}")
```

### 5. Environment Variables Check

Ensure all required environment variables are set in your `.env` file:

```bash
# LLM APIs (at least one required)
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

# HubSpot
HUBSPOT_ACCESS_TOKEN=your_token_here

# Tavily (for web search)
TAVILY_API_KEY=your_key_here

# Relevance AI (for LinkedIn research)
RELEVANCE_AI_API_KEY=your_key_here
RELEVANCE_AI_STUDIO_ID=your_id_here
RELEVANCE_AI_PROJECT_ID=your_id_here
```

### 6. Run the Streamlit App

1. **Start with minimal functionality:**

   ```bash
   streamlit run streamlit_app.py
   ```

2. **Test in this order:**

   - Manual entry with a known company (e.g., "OpenAI")
   - Company research only
   - Then add founder research
   - Then add market research
   - Finally test decision support

3. **Monitor logs:**
   - Check the Logs tab in the Streamlit UI
   - Check terminal output for errors

### 7. Common Issues and Solutions

#### Issue: "None" values in agent calls

**Solution:** The HubSpot data extraction might be failing. Check:

- The property names match your HubSpot setup
- The deal has associated companies and contacts

**Cursor Prompt:**

```
In ai_agents/ui/agent_runner.py, add debug logging to the _extract_company_info_from_hubspot method to print:
1. The raw deal_data structure
2. Each extraction attempt
3. The final extracted values
```

#### Issue: Pydantic model errors

**Solution:** Ensure all agent responses are converted to dicts using `model_dump(mode='json')`

#### Issue: Decision support parameter mismatch

**Solution:** The updated agent_runner.py handles the parameter conversion correctly

### 8. Debugging Tips

1. **Add debug prints in AgentRunner:**

   ```python
   logger.info(f"Debug: Company name extracted: {company_name}")
   logger.info(f"Debug: Founders found: {len(founders)}")
   ```

2. **Test with hardcoded data first:**

   ```python
   # In streamlit_app.py, temporarily hardcode:
   test_company = "Test Company Inc"
   test_founder = "John Doe"
   ```

3. **Use Streamlit's st.write() for debugging:**
   ```python
   st.write("Debug - Company Research Result:", st.session_state.results.get('company_research'))
   ```

### 9. Optimization Prompts for Cursor

**Prompt 1: Error Handling**

```
Add comprehensive error handling to all methods in ai_agents/ui/agent_runner.py:
1. Wrap each agent call in try-except
2. Return structured error responses
3. Log all errors with context
4. Handle None/empty responses gracefully
```

**Prompt 2: Progress Tracking**

```
Add progress tracking to AgentRunner:
1. Emit progress updates during long-running operations
2. Add estimated time remaining
3. Allow cancellation of running agents
```

**Prompt 3: Caching**

```
Add caching to AgentRunner to avoid re-running agents:
1. Cache results by input parameters
2. Add cache expiration (e.g., 1 hour)
3. Allow force refresh option
```

### 10. Next Steps

Once basic functionality is working:

1. **Add PDF Processing:**

   - Test pitch deck extraction
   - Integrate with decision support

2. **Add Batch Processing:**

   - Process multiple deals
   - Export bulk results

3. **Add Visualizations:**

   - Investment criteria spider chart
   - Founder assessment comparison
   - Market positioning graphs

4. **Add HubSpot Integration:**
   - Update deal properties with results
   - Create tasks for next steps
   - Send email summaries

## Testing Checklist

- [ ] Environment variables loaded correctly
- [ ] HubSpot connection working
- [ ] Company research agent runs successfully
- [ ] Founder research agent runs successfully
- [ ] Market research agent runs successfully
- [ ] Decision support integrates all data
- [ ] UI displays results correctly
- [ ] Export functionality works
- [ ] Error states handled gracefully
- [ ] Workflow visualization updates correctly

## Quick Test Commands

### Test Environment Setup:

```bash
# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Test imports
python -c "from ai_agents.services import hubspot_client; print('HubSpot: OK')"
python -c "from ai_agents.ui.agent_runner import AgentRunner; print('AgentRunner: OK')"
```

### Test Agent Functions:

```python
# Quick agent test
from ai_agents.agents.company_research_agent import run_company_research_cli
result = run_company_research_cli("OpenAI")
print(f"Company research result type: {type(result)}")
```

### Debug Streamlit Session State:

```python
# Add to streamlit_app.py for debugging
with st.sidebar.expander("Debug Info"):
    st.write("Session State Keys:", list(st.session_state.keys()))
    st.write("Results Keys:", list(st.session_state.results.keys()) if 'results' in st.session_state else "No results")
    st.write("Workflow State:", st.session_state.get('workflow_state', {}))
```

## Troubleshooting Commands

If you encounter issues, use these commands to diagnose:

```bash
# Check if all required packages are installed
pip list | grep -E "(streamlit|pydantic|plotly|pandas)"

# Run with debug logging
PYTHONPATH=. streamlit run streamlit_app.py --logger.level=debug

# Test individual agent outside of Streamlit
python -c "
from ai_agents.agents.company_research_agent import run_company_research_cli
result = run_company_research_cli('Test Company')
print('Success!' if result else 'Failed!')
"
```

## Support

If you continue to experience issues after following this guide:

1. Check the terminal output for detailed error messages
2. Verify all environment variables are set correctly
3. Ensure your HubSpot access token has the necessary permissions
4. Test each agent individually before running the full pipeline

Remember: Start simple (manual entry, single agent) and gradually add complexity (HubSpot integration, full pipeline) to isolate any remaining issues.
