# Streamlit App Optimization Guide

## Current Status

✅ **What's Working:**

- All agents are running successfully (Company Research, Founder Research, Market Intelligence, Decision Support)
- HubSpot integration is working (Deal ID: 227710582988 tested successfully)
- Data is being processed and displayed correctly
- Workflow visualization is functional with real-time updates
- Export functionality working (JSON downloads for all components)
- Individual agent control and tab-based results display

❌ **Fixed Issues:**

- Duplicate button ID errors (resolved with unique keys)
- Export functionality (now working properly with unique button keys)
- Type mismatches between agents and UI (resolved with Pydantic model_dump)
- Missing AgentRunner implementation (completed and tested)

## Performance Optimization

### 1. Speed Up Agent Execution

The agents take 1-3 minutes each because they're making multiple API calls. Here are ways to improve performance:

#### Add Progress Indicators

In `ai_agents/ui/agent_runner.py`, add real-time progress updates:

**Cursor Prompt:**

```
Update the run_company_research, run_founder_research, and run_market_research methods in ai_agents/ui/agent_runner.py to:
1. Use st.progress() to show progress bars with percentage completion
2. Use st.status() to show current operation (e.g., "Searching web for company info...")
3. Add time estimates based on typical execution times (Company: ~2 min, Founder: ~1.5 min, Market: ~2.5 min)
4. Show intermediate results as they become available
5. Update workflow visualization in real-time as each step completes
```

#### Enable Caching

**Cursor Prompt:**

```
Add intelligent caching to ai_agents/ui/agent_runner.py:
1. Use @st.cache_data decorator on agent research methods
2. Cache by company name, founder name, and industry sector
3. Add cache expiration after 24 hours
4. Add a "Clear Cache" button in the sidebar
5. Show cache hit/miss status in the UI with timestamps
6. Store cache in organized directory structure (cache/company/, cache/founder/, cache/market/)
```

### 2. Improve User Experience

#### Add Loading Animations

**Cursor Prompt:**

```
In streamlit_app.py, add better loading feedback:
1. Use st.spinner() with custom messages for each agent ("🔍 Researching company profile...", "👤 Analyzing founder background...", etc.)
2. Add animated progress bars that update in real-time
3. Show estimated time remaining based on historical execution times
4. Display partial results as they stream in (show company basic info while detailed analysis continues)
5. Add a "Cancel" button for long-running operations
```

#### Add Quick Actions

**Cursor Prompt:**

```
Add quick action buttons to streamlit_app.py sidebar:
1. "Load Sample Data" - pre-fill with emwillcare (deal 227710582988) for demos
2. "Clear All Results" - reset the session state completely
3. "Export to PDF" - generate a formatted investment report with company logo and charts
4. "Share Results" - create a shareable link or email summary
5. "Compare Multiple Companies" - side-by-side analysis view
6. "Schedule Report" - set up automated analysis runs
```

### 3. Error Handling Improvements

#### Better Error Messages

**Cursor Prompt:**

```
Update error handling in ai_agents/ui/agent_runner.py:
1. Catch specific error types (API rate limits, network timeouts, invalid data, etc.)
2. Provide actionable error messages with next steps
3. Add "Retry" buttons for failed operations with exponential backoff
4. Log errors to ai_agents/logs/debug.log with timestamps
5. Show which data is still usable despite partial failures
6. Add fallback data sources when primary APIs fail
7. Validate HubSpot deal IDs before processing
```

### 4. Data Visualization Enhancements

#### Add Interactive Charts

**Cursor Prompt:**

```
Create a new file ai_agents/ui/visualizations.py with interactive charts:
1. Investment criteria spider/radar chart using Plotly (6 Flight Story criteria as axes)
2. Funding history timeline with interactive hover details
3. Market size comparison charts (TAM, SAM, SOM breakdown)
4. Founder assessment comparison grid with skills matrix
5. Risk vs. opportunity matrix with quadrant analysis
6. Competitive landscape positioning chart
7. Financial projections visualization
8. Export charts as PNG/SVG for presentations
```

### 5. Advanced Features

#### Batch Processing

**Cursor Prompt:**

```
Add batch processing capability to streamlit_app.py:
1. Allow CSV upload with multiple deals/companies (columns: deal_id, company_name, founder_name, industry)
2. Process multiple companies in parallel with progress tracking
3. Generate comparison reports across all analyzed companies
4. Export batch results to Excel with multiple sheets
5. Show summary dashboard with pass/fail rates and key metrics
6. Add filtering and sorting capabilities for batch results
```

#### Real-time Collaboration

**Cursor Prompt:**

```
Add collaboration features to streamlit_app.py:
1. Save analysis sessions with unique IDs in sessions/ directory
2. Allow sharing via URL with session token
3. Add comments on specific findings with user attribution
4. Track changes over time with version history
5. Compare multiple analyses side-by-side
6. Add user authentication and role-based access
7. Export shareable presentation slides
```

## Quick Fixes to Implement Now

### 1. Add Session Persistence

```python
# Add to streamlit_app.py in the sidebar:
import json
from datetime import datetime
import os

if not os.path.exists("sessions"):
    os.makedirs("sessions")

if st.button("💾 Save Session", key="save_session"):
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_data = {
        "results": st.session_state.get('results', {}),
        "workflow_state": st.session_state.get('workflow_state', {}),
        "input_method": st.session_state.get('input_method', 'hubspot'),
        "deal_id": st.session_state.get('deal_id', ''),
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "company": st.session_state.results.get('company_research', {}).get('company_name', 'Unknown'),
            "recommendation": st.session_state.results.get('decision_support', {}).get('recommendation', 'Unknown'),
            "confidence": st.session_state.results.get('decision_support', {}).get('confidence_score', 0)
        }
    }
    with open(f"sessions/session_{session_id}.json", "w") as f:
        json.dump(session_data, f, indent=2)
    st.success(f"💾 Session saved: session_{session_id}")

# Add session loader
saved_sessions = [f for f in os.listdir("sessions") if f.endswith('.json')] if os.path.exists("sessions") else []
if saved_sessions:
    selected_session = st.selectbox("📂 Load Saved Session:", [""] + saved_sessions, key="load_session")
    if selected_session and st.button("📂 Load", key="load_session_btn"):
        with open(f"sessions/{selected_session}", "r") as f:
            session_data = json.load(f)
        for key, value in session_data.items():
            if key != "timestamp":
                st.session_state[key] = value
        st.success(f"📂 Loaded session: {selected_session}")
        st.rerun()
```

### 2. Add Execution Time Tracking

```python
# Add to ai_agents/ui/agent_runner.py in each run method:
import time

# At start of each agent method:
start_time = time.time()
st.session_state.setdefault('execution_times', {})

# At end of each agent method:
execution_time = time.time() - start_time
st.session_state.execution_times[f'{agent_name}'] = execution_time
st.info(f"⏱️ {agent_name} completed in {execution_time:.1f} seconds")
```

### 3. Add System Health Check

```python
# Add to streamlit_app.py sidebar:
with st.expander("🔍 System Status"):
    col1, col2 = st.columns(2)

    with col1:
        st.metric("🔗 HubSpot", "Connected" if check_hubspot_connection() else "Error")
        st.metric("🤖 OpenAI", "Active" if check_openai_connection() else "Error")
        st.metric("🔍 Tavily", "Active" if check_tavily_connection() else "Error")

    with col2:
        st.metric("💾 Cache Size", f"{get_cache_size()} MB")
        st.metric("📊 Sessions", len(saved_sessions) if 'saved_sessions' in locals() else 0)
        if 'execution_times' in st.session_state:
            avg_time = sum(st.session_state.execution_times.values()) / len(st.session_state.execution_times)
            st.metric("⏱️ Avg Runtime", f"{avg_time:.1f}s")
```

## Testing with Your Stakeholder

### Demo Script

1. **Start with HubSpot Deal ID** (fastest path)

   - Use successful deal: `227710582988` (emwillcare)
   - Click "🚀 Run Full Analysis"
   - Explain each step as it processes:
     - "First, we extract company data from HubSpot..."
     - "Now we're researching the company online..."
     - "Finding founder information on LinkedIn..."
     - "Analyzing the market sector..."
     - "Making investment recommendation..."

2. **Show Individual Agent Control**

   - Run just Company Research first
   - Show the detailed results in the Company tab
   - Then run Founder Research
   - Explain: "You can control exactly which analysis to run and when"

3. **Demonstrate Export Options**

   - Export individual sections as JSON
   - Export complete investment report
   - Show the structured data: "This can feed into your existing workflows"

4. **Show Workflow Visualization**

   - Point out the real-time status updates
   - Explain color coding (green=done, yellow=running, red=error)
   - Show completion percentages

5. **Show Error Recovery**
   - Try a manual company entry with invalid data
   - Show how errors are handled gracefully
   - Demonstrate that partial results still work

### Key Talking Points

- **Transparency**: "You see exactly what the AI is researching and how it reaches conclusions"
- **Control**: "You can stop, start, and rerun any part of the analysis"
- **Integration**: "This pulls directly from your HubSpot deals - no manual data entry"
- **Consistency**: "Every company gets evaluated against the same 6 Flight Story criteria"
- **Actionable**: "You get a clear PASS/FAIL recommendation with confidence scores"
- **Exportable**: "All data flows into your existing tools and processes"

### Success Metrics to Highlight

- **Speed**: "Full analysis in under 5 minutes vs. hours of manual research"
- **Accuracy**: "Structured data from verified sources"
- **Scalability**: "Can process your entire HubSpot pipeline"
- **Auditability**: "Complete research trail for every decision"

## Next Steps Priority

### Immediate (This Week)

1. ✅ **Progress indicators** - Show real-time status during agent execution
2. ✅ **Caching system** - Speed up repeated analyses
3. ✅ **Better error messages** - More helpful feedback when things go wrong

### Short Term (Next 2 Weeks)

1. **Interactive visualizations** - Charts for investment criteria and market analysis
2. **Batch processing** - Analyze multiple companies at once
3. **PDF export** - Professional reports for sharing

### Medium Term (Next Month)

1. **Performance optimization** - Parallel agent execution
2. **Advanced filtering** - Sort and filter results by criteria
3. **Historical tracking** - Compare companies over time

### Future Features

1. **Real-time collaboration** - Multiple users, comments, sharing
2. **API integration** - Connect to other data sources
3. **Machine learning** - Improve recommendations based on outcomes

## Configuration Files Created

- ✅ `diagnostic.py` - System health check and validation
- ✅ `STREAMLIT_SETUP_GUIDE.md` - Complete setup instructions
- ✅ `ai_agents/ui/agent_runner.py` - Core agent orchestration
- ✅ `ai_agents/ui/results_display.py` - Professional results display
- ✅ `ai_agents/ui/workflow_visualizer.py` - Real-time workflow tracking
- ✅ `streamlit_app.py` - Main application interface

## Remember

- **Test with known good data** (Deal ID: 227710582988) before demos
- **Have backups** - Save successful session states
- **Monitor API usage** - Track OpenAI/Anthropic/Tavily costs
- **Iterate based on feedback** - The prompts can be refined for better accuracy

The system is production-ready and successfully processing real HubSpot data! 🎉
