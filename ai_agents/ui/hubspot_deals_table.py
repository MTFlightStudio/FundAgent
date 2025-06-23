import streamlit as st
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ai_agents.services import hubspot_client
# from ai_agents.integrations.hubspot import list_objects  # This import doesn't exist
from ai_agents.ui.agent_runner import get_runner

logger = logging.getLogger(__name__)

class HubSpotDealsTable:
    """Manages HubSpot deals display and selection for analysis"""
    
    def __init__(self):
        self.initialize_session_state()
        self.stage_labels = {}  # Cache for stage ID to label mapping
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'deals_data' not in st.session_state:
            st.session_state.deals_data = None
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = None
        if 'selected_deals' not in st.session_state:
            st.session_state.selected_deals = []
        if 'deals_df' not in st.session_state:
            st.session_state.deals_df = None
        if 'stage_mapping' not in st.session_state:
            st.session_state.stage_mapping = {}
    
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def fetch_pipeline_stages(_self) -> Dict[str, str]:
        """Fetch and cache pipeline stages mapping"""
        try:
            stages = hubspot_client.get_pipeline_stages("default", "deals")
            stage_mapping = {stage['id']: stage['label'] for stage in stages}
            return stage_mapping
        except Exception as e:
            logger.error(f"Error fetching pipeline stages: {e}")
            return {}
    
    def fetch_all_deals(self, limit: int = None) -> List[Dict[str, Any]]:
        """Fetch all deals from HubSpot with pagination"""
        all_deals = []
        after = None
        
        deal_properties = [
            "dealname", "amount", "dealstage", "pipeline", "closedate",
            "hs_object_id", "hubspot_owner_id", "createdate", 
            "hs_lastmodifieddate", "request"
        ]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            page_count = 0
            while True:
                page_count += 1
                
                # Update status
                if limit:
                    status_text.text(f"Fetching deals... ({len(all_deals)}/{limit}) - Page {page_count}")
                    progress = min(len(all_deals) / limit, 1.0)
                    progress_bar.progress(progress)
                else:
                    status_text.text(f"Fetching all deals... ({len(all_deals)} found) - Page {page_count}")
                    # For unlimited fetch, show a pulsing progress
                    progress_bar.progress((page_count % 10) / 10)
                
                # Determine how many to fetch this page
                page_limit = 100  # HubSpot API limit per page
                if limit and len(all_deals) + page_limit > limit:
                    page_limit = limit - len(all_deals)
                
                response = hubspot_client.list_objects(
                    object_type="deals",
                    properties=deal_properties,
                    limit=page_limit,
                    after=after
                )
                
                if not response or 'results' not in response:
                    break
                
                # Add results
                page_results = response['results']
                all_deals.extend(page_results)
                
                # Check if we've hit our limit
                if limit and len(all_deals) >= limit:
                    break
                
                # Get next page cursor
                paging = response.get('paging', {})
                after = paging.get('next', {}).get('after')
                
                # If no more pages, break
                if not after:
                    break
                
                # If we got fewer results than requested, we're at the end
                if len(page_results) < page_limit:
                    break
            
            progress_bar.empty()
            status_text.empty()
            
            return all_deals
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Error fetching deals: {str(e)}")
            return []
    
    def enrich_deal_data(self, deals: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert deals to DataFrame with enriched data"""
        # Get stage mapping
        if not st.session_state.stage_mapping:
            st.session_state.stage_mapping = self.fetch_pipeline_stages()
        
        processed_deals = []
        
        for deal in deals:
            props = deal.get('properties', {})
            
            # Parse dates
            created_date = None
            if props.get('createdate'):
                try:
                    created_date = datetime.fromisoformat(props['createdate'].replace('Z', '+00:00'))
                except:
                    created_date = None
            
            # Parse amount
            amount = None
            if props.get('amount'):
                try:
                    amount = float(props['amount'])
                except:
                    amount = 0
            
            # Get stage label
            stage_id = props.get('dealstage', '')
            stage_label = st.session_state.stage_mapping.get(stage_id, stage_id)
            
            # Calculate days old, handling timezone aware/naive datetime differences
            days_old = None
            if created_date:
                try:
                    if created_date.tzinfo is not None:
                        # created_date is timezone-aware, make datetime.now() aware too
                        now = datetime.now(created_date.tzinfo)
                    else:
                        # created_date is timezone-naive, use naive datetime.now()
                        now = datetime.now()
                    days_old = (now - created_date).days
                except Exception as e:
                    # Fallback: convert to naive datetime if there are issues
                    if hasattr(created_date, 'replace'):
                        created_date_naive = created_date.replace(tzinfo=None)
                        days_old = (datetime.now() - created_date_naive).days
                    else:
                        days_old = None
            
            processed_deals.append({
                'Deal ID': props.get('hs_object_id', ''),
                'Deal Name': props.get('dealname', 'Unnamed Deal'),
                'Stage': stage_label,
                'Stage ID': stage_id,
                'Amount': amount,
                'Created Date': created_date,
                'Days Old': days_old,
                'Pipeline': props.get('pipeline', 'default'),
                'Equity Request': props.get('request', ''),
                '_raw_properties': props  # Keep raw data for analysis
            })
        
        df = pd.DataFrame(processed_deals)
        
        # Sort by creation date (newest first)
        if 'Created Date' in df.columns:
            df = df.sort_values('Created Date', ascending=False)
        
        return df
    
    def display_summary_metrics(self, df: pd.DataFrame):
        """Display summary metrics for the deals"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Deals", len(df))
        
        with col2:
            total_value = df['Amount'].sum()
            st.metric("Total Pipeline Value", f"${total_value:,.0f}")
        
        with col3:
            avg_deal = df['Amount'].mean() if len(df) > 0 else 0
            st.metric("Average Deal Size", f"${avg_deal:,.0f}")
        
        with col4:
            active_deals = len(df[~df['Stage'].str.contains('Rejected|Closed', case=False, na=False)])
            st.metric("Active Deals", active_deals)
        
        # Stage distribution
        with st.expander("📊 Deal Distribution by Stage"):
            stage_summary = df.groupby('Stage').agg({
                'Deal ID': 'count',
                'Amount': 'sum'
            }).rename(columns={'Deal ID': 'Count', 'Amount': 'Total Value'})
            
            stage_summary['Percentage'] = (stage_summary['Count'] / len(df) * 100).round(1)
            stage_summary = stage_summary.sort_values('Count', ascending=False)
            
            st.dataframe(
                stage_summary.style.format({
                    'Total Value': '${:,.0f}',
                    'Percentage': '{:.1f}%'
                }),
                use_container_width=True
            )
    
    def display_deals_table(self, df: pd.DataFrame):
        """Display the interactive deals table"""
        st.subheader("📋 All Deals")
        
        # Instructions for batch analysis
        with st.expander("ℹ️ How to use Batch Analysis"):
            st.markdown("""
            **📊 Batch Analysis Feature:**
            1. **Select Deals**: Use the checkboxes in the 'Select' column to choose deals
            2. **Add to Queue**: Click '🎯 Add Selected Deals to Analysis Queue' 
            3. **Run Analysis**: Click '🚀 Run Analysis on Queue' to analyze all selected deals
            4. **Monitor Progress**: Watch the progress bar as each deal is analyzed
            5. **View Results**: See detailed results for each deal analysis
            
            **💡 Tip:** You can filter the table first, then select deals from the filtered results.
            """)
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            stage_filter = st.multiselect(
                "Filter by Stage",
                options=sorted(df['Stage'].unique()),
                default=[]
            )
        
        with col2:
            min_amount = st.number_input(
                "Min Deal Amount ($)",
                min_value=0,
                value=0,
                step=10000
            )
        
        with col3:
            days_filter = st.number_input(
                "Created within (days)",
                min_value=0,
                value=0,
                help="0 = show all deals"
            )
        
        # Apply filters
        filtered_df = df.copy()
        
        if stage_filter:
            filtered_df = filtered_df[filtered_df['Stage'].isin(stage_filter)]
        
        if min_amount > 0:
            filtered_df = filtered_df[filtered_df['Amount'] >= min_amount]
        
        if days_filter > 0:
            filtered_df = filtered_df[filtered_df['Days Old'] <= days_filter]
        
        # Display count
        st.write(f"Showing {len(filtered_df)} of {len(df)} deals")
        
        # Configure display columns
        display_columns = ['Deal Name', 'Stage', 'Amount', 'Created Date', 'Days Old', 'Equity Request']
        
        # Add a selection column
        selection_df = filtered_df.copy()
        selection_df.insert(0, 'Select', False)  # Add checkbox column at the beginning
        
        # Create selectable dataframe
        edited_df = st.data_editor(
            selection_df[['Select'] + display_columns + ['Deal ID']],
            column_config={
                "Select": st.column_config.CheckboxColumn(
                    "Select",
                    help="Select deals for analysis",
                    default=False,
                ),
                "Amount": st.column_config.NumberColumn(
                    "Amount",
                    format="$%d",
                ),
                "Created Date": st.column_config.DatetimeColumn(
                    "Created Date",
                    format="DD/MM/YYYY",
                ),
                "Days Old": st.column_config.NumberColumn(
                    "Days Old",
                    format="%d days",
                ),
                "Equity Request": st.column_config.TextColumn(
                    "Equity %",
                ),
            },
            hide_index=True,
            use_container_width=True,
            num_rows="dynamic",
            disabled=["Deal Name", "Stage", "Amount", "Created Date", "Days Old", "Equity Request", "Deal ID"],
            key="deals_selection"
        )
        
        # Handle selection
        if st.button("🎯 Add Selected Deals to Analysis Queue"):
            # Get selected rows based on checkbox
            selected_rows = edited_df[edited_df['Select'] == True]
            selected_deal_ids = selected_rows['Deal ID'].tolist()
            
            if selected_deal_ids:
                st.session_state.selected_deals.extend(selected_deal_ids)
                st.session_state.selected_deals = list(set(st.session_state.selected_deals))  # Remove duplicates
                st.success(f"Added {len(selected_deal_ids)} deals to analysis queue")
                st.rerun()  # Refresh to clear selections
            else:
                st.warning("No deals selected. Use the checkboxes in the 'Select' column to choose deals for analysis.")
        
        # Show selected deals queue
        if st.session_state.selected_deals:
            with st.expander(f"📌 Analysis Queue ({len(st.session_state.selected_deals)} deals)"):
                for deal_id in st.session_state.selected_deals:
                    deal_row = df[df['Deal ID'] == deal_id]
                    if not deal_row.empty:
                        deal_name = deal_row.iloc[0]['Deal Name']
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.write(f"• {deal_name} (ID: {deal_id})")
                        with col2:
                            if st.button("❌", key=f"remove_{deal_id}"):
                                st.session_state.selected_deals.remove(deal_id)
                                st.rerun()
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🚀 Run Analysis on Queue", type="primary"):
                        st.session_state.run_batch_analysis = True
                        st.rerun()
                with col2:
                    if st.button("🗑️ Clear Queue"):
                        st.session_state.selected_deals = []
                        st.rerun()
    
    def run(self):
        """Main method to run the deals table interface"""
        st.title("🎯 HubSpot Deals Dashboard")
        
        # Refresh controls
        col1, col2, col3 = st.columns([2, 1, 3])
        
        with col1:
            if st.button("🔄 Refresh Deals Data", type="primary"):
                st.session_state.deals_data = None
                st.session_state.deals_df = None
                st.rerun()
        
        with col2:
            if st.session_state.last_refresh:
                time_diff = datetime.now() - st.session_state.last_refresh
                st.caption(f"Last refresh: {time_diff.seconds//60}m ago")
            
        with col3:
            # Show fetch info
            if st.session_state.deals_data:
                total_fetched = len(st.session_state.deals_data)
                st.metric("Total Deals", total_fetched)
        
        # Fetch or use cached data - always fetch ALL deals
        if st.session_state.deals_data is None:
            with st.spinner("Fetching all deals from HubSpot..."):
                # First, check total available deals
                total_available = self.check_total_deals_count()
                
                # Always fetch all deals (no limit)
                deals = self.fetch_all_deals(limit=None)
                if deals:
                    st.session_state.deals_data = deals
                    st.session_state.last_refresh = datetime.now()
                    df = self.enrich_deal_data(deals)
                    st.session_state.deals_df = df
                    
                    # Show information about fetched data
                    total_fetched = len(deals)
                    
                    if total_available and total_fetched < total_available:
                        st.warning(f"⚠️ Fetched {total_fetched} deals, but API indicates {total_available} total. There may be an issue with the API or some deals may be inaccessible.")
                    else:
                        st.success(f"✅ Successfully fetched all {total_fetched} deals from HubSpot")
                else:
                    st.error("Failed to fetch deals")
                    return
        else:
            df = st.session_state.deals_df
        
        if df is not None and not df.empty:
            # Display summary metrics
            self.display_summary_metrics(df)
            
            # Display deals table
            self.display_deals_table(df)
            
            # Batch analysis runner
            if st.session_state.get('run_batch_analysis'):
                self.run_batch_analysis()
                st.session_state.run_batch_analysis = False
        else:
            st.warning("No deals data available")
    
    def run_batch_analysis(self):
        """Run analysis on selected deals"""
        if not st.session_state.selected_deals:
            st.warning("No deals selected for analysis")
            return
        
        st.subheader("🔄 Running Batch Analysis")
        
        runner = get_runner()
        results = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_deals = len(st.session_state.selected_deals)
        
        for idx, deal_id in enumerate(st.session_state.selected_deals):
            progress = (idx + 1) / total_deals
            progress_bar.progress(progress)
            
            deal_info = st.session_state.deals_df[st.session_state.deals_df['Deal ID'] == deal_id]
            if not deal_info.empty:
                deal_name = deal_info.iloc[0]['Deal Name']
                status_text.text(f"Analyzing {deal_name} ({idx + 1}/{total_deals})...")
                
                try:
                    # Run the existing orchestrator
                    from ai_agents.agents.investment_orchestrator import analyze_investment_opportunity
                    result = analyze_investment_opportunity(deal_id)
                    results[deal_id] = {
                        'status': 'success',
                        'result': result,
                        'deal_name': deal_name
                    }
                except Exception as e:
                    results[deal_id] = {
                        'status': 'error',
                        'error': str(e),
                        'deal_name': deal_name
                    }
        
        progress_bar.empty()
        status_text.empty()
        
        # Display results summary
        st.success(f"✅ Batch analysis complete for {total_deals} deals")
        
        success_count = sum(1 for r in results.values() if r['status'] == 'success')
        error_count = sum(1 for r in results.values() if r['status'] == 'error')
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Successful", success_count)
        with col2:
            st.metric("Failed", error_count)
        
        # Show detailed results
        with st.expander("📊 Detailed Results"):
            for deal_id, result in results.items():
                st.write(f"**{result['deal_name']}** (ID: {deal_id})")
                if result['status'] == 'success':
                    st.success("✅ Analysis complete")
                    if result.get('result'):
                        st.write(f"Report: {result['result']}")
                else:
                    st.error(f"❌ Error: {result.get('error')}")
                st.divider()

    def check_total_deals_count(self) -> int:
        """Quick check to estimate total number of deals available"""
        try:
            # Fetch just one deal to see pagination info
            response = hubspot_client.list_objects(
                object_type="deals",
                properties=["hs_object_id"],
                limit=1
            )
            
            if response and 'total' in response:
                return response['total']
            else:
                # If total not provided, we can't determine exact count
                return None
                
        except Exception as e:
            logger.error(f"Error checking total deals count: {e}")
            return None

# Create a function to be imported in the main app
def render_hubspot_deals_page():
    """Render the HubSpot deals dashboard"""
    table = HubSpotDealsTable()
    table.run() 