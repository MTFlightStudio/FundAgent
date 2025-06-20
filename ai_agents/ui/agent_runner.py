import streamlit as st
import json
import logging
import sys
import io
import time
import hashlib
import os
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
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
        if 'execution_times' not in st.session_state:
            st.session_state.execution_times = {}
        
        self.cache_expiry_hours = 24  # Cache expires after 24 hours
        self._setup_cache_directories()

    def _setup_cache_directories(self):
        """Create cache directories if they don't exist"""
        cache_dirs = ['cache', 'cache/company', 'cache/founder', 'cache/market', 'cache/decision']
        for cache_dir in cache_dirs:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)

    def _get_cache_key(self, agent_type: str, **kwargs) -> str:
        """Generate a unique cache key based on agent type and parameters"""
        # Create a string representation of the parameters
        params_str = json.dumps(kwargs, sort_keys=True)
        # Create a hash of the parameters for a clean filename
        cache_key = hashlib.md5(f"{agent_type}_{params_str}".encode()).hexdigest()
        return cache_key

    def _get_cache_file_path(self, agent_type: str, cache_key: str) -> str:
        """Get the cache file path for the given agent type and cache key"""
        return f"cache/{agent_type}/{cache_key}.json"

    def _is_cache_valid(self, cache_file_path: str) -> bool:
        """Check if cache file exists and is not expired"""
        if not os.path.exists(cache_file_path):
            return False
        
        # Check if cache is expired
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_file_path))
        expiry_time = datetime.now() - timedelta(hours=self.cache_expiry_hours)
        
        return file_time > expiry_time

    def _load_from_cache(self, cache_file_path: str) -> Optional[Dict[str, Any]]:
        """Load data from cache file"""
        try:
            with open(cache_file_path, 'r') as f:
                cached_data = json.load(f)
                # Add cache metadata
                cached_data['_cache_info'] = {
                    'cached': True,
                    'cache_time': datetime.fromtimestamp(os.path.getmtime(cache_file_path)).isoformat(),
                    'cache_age_hours': round((datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file_path))).total_seconds() / 3600, 2)
                }
                return cached_data
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None

    def _save_to_cache(self, cache_file_path: str, data: Dict[str, Any]):
        """Save data to cache file"""
        try:
            # Remove cache info if it exists in the data before saving
            data_to_cache = {k: v for k, v in data.items() if k != '_cache_info'}
            with open(cache_file_path, 'w') as f:
                json.dump(data_to_cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")

    def clear_cache(self, agent_type: str = None):
        """Clear cache files for specified agent type or all agents"""
        if agent_type:
            cache_dir = f"cache/{agent_type}"
            if os.path.exists(cache_dir):
                for file in os.listdir(cache_dir):
                    if file.endswith('.json'):
                        os.remove(os.path.join(cache_dir, file))
                st.success(f"🗑️ Cleared {agent_type} cache")
        else:
            # Clear all caches
            for agent_type in ['company', 'founder', 'market', 'decision']:
                self.clear_cache(agent_type)
            st.success("🗑️ Cleared all caches")

    def _extract_company_info_from_hubspot(self, deal_id: str) -> Dict[str, Any]:
        """Extract company information from HubSpot deal data"""
        if not self.hubspot_available:
            raise ValueError("HubSpot client not available")
            
        try:
            deal_data = hubspot_client.get_deal_with_associated_data(deal_id)
            
            # Extract company data
            company_name = None
            company_info = {}
            if deal_data.get('associated_companies'):
                company_props = deal_data['associated_companies'][0].get('properties', {})
                company_name = company_props.get('name')
                
                # Extract comprehensive company information including ALL HubSpot fields
                company_info = {
                    'company_name': company_name,
                    'website': company_props.get('website') or company_props.get('domain'),
                    'description': company_props.get('description'),
                    'industry': company_props.get('what_sector_is_your_business_product_') or company_props.get('industry'),
                    'sub_industry': company_props.get('what_sector_is_your_business_product_'),
                    'location_hq': company_props.get('where_is_your_business_based_'),
                    'business_model': company_props.get('what_is_your_usp__what_makes_you_different_from_your_competitors_'),
                    'target_customer': company_props.get('what_best_describes_your_customer_base_'),
                    'funding_stage': company_props.get('what_best_describes_your_stage_of_business_'),
                    'total_funding_raised': company_props.get('how_much_have_you_raised_prior_to_this_round_'),
                    
                    # Employee count
                    'team_size': company_props.get('how_many_employees_do_you_have__full_time_equivalents_'),
                    'number_of_employees': company_props.get('number_of_employees'),
                    
                    # Impact & Innovation fields
                    'which__if_any__of_the_un_sdg_17_goals_does_your_business_address_': company_props.get('which__if_any__of_the_un_sdg_17_goals_does_your_business_address_'),
                    'does_your_product_contribute_to_a_healthier__happier_whole_human_experience_': company_props.get('does_your_product_contribute_to_a_healthier__happier_whole_human_experience_'),
                    'how_does_your_product_contribute_to_a_healthier__happier_whole_human_experience_': company_props.get('how_does_your_product_contribute_to_a_healthier__happier_whole_human_experience_'),
                    'how_does_your_company_use_innovation__through_technology_or_to_differentiate_the_business_model__': company_props.get('how_does_your_company_use_innovation__through_technology_or_to_differentiate_the_business_model__'),
                    
                    # Investment & Strategy fields
                    'please_expand': company_props.get('please_expand'),  # Investment use & expansion plans
                    'what_is_it_that_you_re_looking_for_with_a_partnership_from_flight_': company_props.get('what_is_it_that_you_re_looking_for_with_a_partnership_from_flight_'),
                    
                    # Documents & Resources
                    'please_attach_your_pitch_deck': company_props.get('please_attach_your_pitch_deck'),
                    
                    # Key metrics for display
                    'key_metrics': {
                        'LTM Revenue': company_props.get('what_is_your_ltm__last_12_months__revenue_'),
                        'Monthly Revenue': company_props.get('what_is_your_current_monthly_revenue_'),
                        'Current Raise Amount': company_props.get('how_much_are_you_raising_at_this_stage_'),
                        'Valuation': company_props.get('what_valuation_are_you_raising_at_'),
                        'Prior Funding': company_props.get('how_much_have_you_raised_prior_to_this_round_'),
                        'Team Equity': company_props.get('how_much_of_the_equity_do_you_your_team_have_'),
                        'Business Stage': company_props.get('what_best_describes_your_stage_of_business_'),
                    }
                }
                
                # Extract additional details
                if company_props.get('describe_the_business_product_in_one_sentence'):
                    company_info['one_sentence_description'] = company_props.get('describe_the_business_product_in_one_sentence')
            
            # Extract industry/sector for fallback
            industry = None
            if deal_data.get('associated_companies'):
                company_props = deal_data['associated_companies'][0].get('properties', {})
                industry = company_props.get('what_sector_is_your_business_product_', 
                                            company_props.get('industry'))
            
            # Extract founder information with LinkedIn URLs
            founders = []
            founder_linkedin_urls = []
            founder_info_list = []  # Store complete founder info

            if deal_data.get('associated_contacts'):
                for contact in deal_data['associated_contacts']:
                    props = contact.get('properties', {})
                    first_name = props.get('firstname', '')
                    last_name = props.get('lastname', '')
                    email = props.get('email', '')
                    
                    # Primary LinkedIn URL for this contact
                    primary_linkedin = props.get('hs_linkedin_url', '')
                    
                    if first_name or last_name:
                        full_name = f"{first_name} {last_name}".strip()
                        
                        founder_info = {
                            'name': full_name,
                            'email': email,
                            'linkedin_url': primary_linkedin,
                            'first_name': first_name,
                            'last_name': last_name
                        }
                        
                        founder_info_list.append(founder_info)
                        founders.append(full_name)
                        
                        if primary_linkedin:
                            founder_linkedin_urls.append(primary_linkedin)

            # Extract additional founder LinkedIn URLs from the multi-line field
            additional_founders_field = None
            if deal_data.get('associated_contacts') and deal_data['associated_contacts']:
                # Check first contact for the additional founders field
                first_contact_props = deal_data['associated_contacts'][0].get('properties', {})
                additional_founders_field = first_contact_props.get('attach_link_to_all_founders_linkedin_profiles', '')

            # Parse additional LinkedIn URLs
            additional_linkedin_urls = []
            if additional_founders_field:
                # Split by newlines and clean up
                urls = additional_founders_field.strip().split('\n')
                for url in urls:
                    url = url.strip()
                    if url and 'linkedin.com' in url:
                        additional_linkedin_urls.append(url)

            # Try to extract names from additional LinkedIn URLs
            for url in additional_linkedin_urls:
                if url not in founder_linkedin_urls:  # Avoid duplicates
                    # Extract username from LinkedIn URL
                    import re
                    match = re.search(r'linkedin\.com/in/([^/]+)', url)
                    if match:
                        username = match.group(1)
                        # Convert username to a readable name (basic heuristic)
                        name_parts = username.replace('-', ' ').title()
                        
                        # Check if this might be an existing founder
                        matched = False
                        for founder_info in founder_info_list:
                            # Simple matching - could be improved
                            if any(part.lower() in founder_info['name'].lower() for part in username.split('-')):
                                # This URL likely belongs to this founder
                                if not founder_info['linkedin_url']:
                                    founder_info['linkedin_url'] = url
                                matched = True
                                break
                        
                        if not matched:
                            # This is likely an additional founder
                            founder_info = {
                                'name': name_parts,
                                'email': None,
                                'linkedin_url': url,
                                'first_name': name_parts.split()[0] if name_parts else '',
                                'last_name': ' '.join(name_parts.split()[1:]) if len(name_parts.split()) > 1 else ''
                            }
                            founder_info_list.append(founder_info)
                            founders.append(name_parts)
                            founder_linkedin_urls.append(url)

            # Update key_metrics with founder info
            if company_info.get('key_metrics'):
                if founders:
                    company_info['key_metrics']['Founders'] = ', '.join(founders)
                    company_info['key_metrics']['Founder Count'] = str(len(founders))

            return {
                'company_name': company_name,
                'industry': industry,
                'founders': founders,
                'founder_info_list': founder_info_list,  # Complete founder information
                'linkedin_urls': founder_linkedin_urls,
                'deal_data': deal_data,
                'company_info': company_info
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

    @st.cache_data(ttl=3600)  # Streamlit cache for 1 hour
    def run_company_research(_self, company_name: str = None, deal_id: str = None) -> Dict[str, Any]:
        """Run company research agent with caching"""
        start_time = time.time()
        
        if 'execution_times' not in st.session_state:
            st.session_state.execution_times = {}
        
        # Generate cache key
        cache_key = _self._get_cache_key('company', company_name=company_name, deal_id=deal_id)
        cache_file_path = _self._get_cache_file_path('company', cache_key)
        
        # Check cache first
        if _self._is_cache_valid(cache_file_path):
            cached_result = _self._load_from_cache(cache_file_path)
            if cached_result:
                execution_time = time.time() - start_time
                st.session_state.execution_times['company_research'] = execution_time
                st.success(f"⚡ Company research loaded from cache in {execution_time:.1f} seconds (Age: {cached_result['_cache_info']['cache_age_hours']:.1f} hours)")
                return cached_result
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("🔍 Initializing company research...")
            progress_bar.progress(10)
            
            hubspot_data = None
            # If deal_id provided, extract company info and full HubSpot data
            if deal_id and not company_name:
                status_text.text("📊 Extracting HubSpot data...")
                progress_bar.progress(20)
                
                hubspot_info = _self._extract_company_info_from_hubspot(deal_id)
                company_name = hubspot_info.get('company_name')
                hubspot_data = hubspot_info.get('deal_data')  # Pass the full HubSpot data
                
                if not company_name:
                    raise ValueError("Could not extract company name from HubSpot deal")
                
                _self.log_execution(f"Extracted company '{company_name}' from HubSpot deal {deal_id}")
            
            if not company_name:
                raise ValueError("Company name is required")
            
            status_text.text("🌐 Researching company with enhanced data...")
            progress_bar.progress(40)
            
            _self.log_execution(f"Running company research for: {company_name}")
            
            # Use enhanced company research with HubSpot data if available
            if hubspot_data:
                _self.log_execution(f"Using HubSpot financial and business data for enhanced research")
                result = run_company_research_cli(company_name, hubspot_data=hubspot_data)
                
                # If we have company_info from HubSpot, use it directly for better data consistency
                if hubspot_info.get('company_info'):
                    _self.log_execution(f"Enhancing result with structured HubSpot company data")
                    # Convert company_info to match the expected structure
                    enhanced_result = {
                        'status': 'success',
                        'company_profile': hubspot_info['company_info'],
                        'timestamp': datetime.now().isoformat()
                    }
                    # Merge with research result if available
                    if result and hasattr(result, 'model_dump'):
                        research_data = result.model_dump(mode='json')
                        # Keep research insights but prioritize HubSpot financial data
                        if research_data.get('company_profile'):
                            for key, value in research_data['company_profile'].items():
                                if key not in enhanced_result['company_profile'] or not enhanced_result['company_profile'].get(key):
                                    enhanced_result['company_profile'][key] = value
                    
                    # Cache and return the enhanced result
                    _self._save_to_cache(cache_file_path, enhanced_result)
                    execution_time = time.time() - start_time
                    st.session_state.execution_times['company_research'] = execution_time
                    progress_bar.empty()
                    status_text.empty()
                    st.success(f"🎉 Enhanced company research completed in {execution_time:.1f} seconds (with HubSpot financial data)")
                    return enhanced_result
            else:
                _self.log_execution(f"Running standard web-only research (no HubSpot data)")
                result = run_company_research_cli(company_name)
            
            status_text.text("✅ Company research completed!")
            progress_bar.progress(100)
            
            if result:
                result_dict = result.model_dump(mode='json')
                
                # Save to cache
                _self._save_to_cache(cache_file_path, result_dict)
                
                # Track execution time
                execution_time = time.time() - start_time
                st.session_state.execution_times['company_research'] = execution_time
                
                # Clean up progress indicators
                progress_bar.empty()
                status_text.empty()
                
                if hubspot_data:
                    st.success(f"🎉 Enhanced company research completed in {execution_time:.1f} seconds (with HubSpot financial data)")
                else:
                    st.success(f"🎉 Company research completed in {execution_time:.1f} seconds (web search only)")
                
                return result_dict
            else:
                raise ValueError("No results returned from company research")
                
        except Exception as e:
            execution_time = time.time() - start_time
            st.session_state.execution_times['company_research'] = execution_time
            
            progress_bar.empty()
            status_text.empty()
            
            error_msg = f"Company research failed: {str(e)}"
            _self.log_execution(error_msg, "ERROR")
            st.error(f"❌ {error_msg}")
            
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    @st.cache_data(ttl=3600)
    def run_founder_research(_self, founder_names: List[str] = None, linkedin_urls: List[str] = None, 
                           deal_id: str = None, company_name: str = None) -> Dict[str, Any]:
        """Run founder research agent with caching"""
        start_time = time.time()
        
        if 'execution_times' not in st.session_state:
            st.session_state.execution_times = {}
        
        # Generate cache key
        cache_key = _self._get_cache_key('founder', founder_names=founder_names, linkedin_urls=linkedin_urls, deal_id=deal_id, company_name=company_name)
        cache_file_path = _self._get_cache_file_path('founder', cache_key)
        
        # Check cache first
        if _self._is_cache_valid(cache_file_path):
            cached_result = _self._load_from_cache(cache_file_path)
            if cached_result:
                execution_time = time.time() - start_time
                st.session_state.execution_times['founder_research'] = execution_time
                st.success(f"⚡ Founder research loaded from cache in {execution_time:.1f} seconds (Age: {cached_result['_cache_info']['cache_age_hours']:.1f} hours)")
                return cached_result
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("👤 Initializing founder research...")
            progress_bar.progress(10)
            
            # Extract founder info from HubSpot if needed
            founder_info_list = []
            current_company = company_name
            
            if deal_id and not founder_names:
                hubspot_data = _self._extract_company_info_from_hubspot(deal_id)
                founder_info_list = hubspot_data.get('founder_info_list', [])
                if not current_company:
                    current_company = hubspot_data.get('company_name')
                if not founder_info_list:
                    raise ValueError("Could not extract founder information from HubSpot deal")
            elif deal_id and not current_company:
                hubspot_data = _self._extract_company_info_from_hubspot(deal_id)
                current_company = hubspot_data.get('company_name')
            
            # If manual entry, create founder info list
            if founder_names and not founder_info_list:
                for i, name in enumerate(founder_names):
                    linkedin_url = linkedin_urls[i] if linkedin_urls and i < len(linkedin_urls) else None
                    founder_info_list.append({
                        'name': name,
                        'linkedin_url': linkedin_url
                    })
            
            if not founder_info_list:
                raise ValueError("No founder information available")
            
            status_text.text("🔗 Searching LinkedIn profiles...")
            progress_bar.progress(25)
            
            founder_profiles = []
            total_founders = len(founder_info_list)
            
            for i, founder_info in enumerate(founder_info_list):
                progress = 25 + (50 * (i + 1) / total_founders)
                founder_name = founder_info['name']
                linkedin_url = founder_info.get('linkedin_url')
                
                status_text.text(f"📊 Analyzing {founder_name}'s background... ({i+1}/{total_founders})")
                progress_bar.progress(int(progress))
                
                _self.log_execution(f"Running founder research for: {founder_name}" + (f" with LinkedIn: {linkedin_url}" if linkedin_url else ""))
                
                try:
                    if linkedin_url:
                        founder_result = run_founder_research_cli_entrypoint(founder_name, linkedin_url, current_company)
                    else:
                        founder_result = run_founder_research_cli_entrypoint(founder_name, current_company=current_company)
                    
                    if founder_result:
                        founder_dict = founder_result.model_dump(mode='json')
                        founder_profiles.append(founder_dict)
                except Exception as e:
                    _self.log_execution(f"Failed to research founder {founder_name}: {str(e)}", "ERROR")
                    # Continue with other founders
                    continue
            
            status_text.text("✅ Founder research completed!")
            progress_bar.progress(100)
            
            founder_data = {
                'status': 'success',
                'founders': founder_profiles,
                'founder_count': len(founder_profiles),
                'total_attempted': total_founders,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save to cache
            _self._save_to_cache(cache_file_path, founder_data)
            
            # Track execution time
            execution_time = time.time() - start_time
            st.session_state.execution_times['founder_research'] = execution_time
            
            # Clean up progress indicators
            progress_bar.empty()
            status_text.empty()
            
            st.success(f"🎉 Founder research completed in {execution_time:.1f} seconds ({len(founder_profiles)}/{total_founders} founders researched successfully)")
            
            return founder_data
                
        except Exception as e:
            execution_time = time.time() - start_time
            st.session_state.execution_times['founder_research'] = execution_time
            
            progress_bar.empty()
            status_text.empty()
            
            error_msg = f"Founder research failed: {str(e)}"
            _self.log_execution(error_msg, "ERROR")
            st.error(f"❌ {error_msg}")
            
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def run_market_research(_self, company_name: str = None, industry: str = None, 
                          deal_id: str = None) -> Dict[str, Any]:
        """Run market research agent with caching"""
        start_time = time.time()
        
        if 'execution_times' not in st.session_state:
            st.session_state.execution_times = {}
        
        # Generate cache key
        cache_key = _self._get_cache_key('market', company_name=company_name, industry=industry, deal_id=deal_id)
        cache_file_path = _self._get_cache_file_path('market', cache_key)
        
        # Check cache first
        if _self._is_cache_valid(cache_file_path):
            cached_result = _self._load_from_cache(cache_file_path)
            if cached_result:
                execution_time = time.time() - start_time
                st.session_state.execution_times['market_research'] = execution_time
                st.success(f"⚡ Market research loaded from cache in {execution_time:.1f} seconds (Age: {cached_result['_cache_info']['cache_age_hours']:.1f} hours)")
                return cached_result
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("📈 Initializing market research...")
            progress_bar.progress(10)
            
            # If deal_id provided, extract industry from HubSpot
            if deal_id and not industry:
                status_text.text("📊 Extracting industry from HubSpot...")
                progress_bar.progress(20)
                
                hubspot_info = _self._extract_company_info_from_hubspot(deal_id)
                industry = hubspot_info.get('industry')
                
                if not industry:
                    # Try to infer from company name if available
                    if not company_name:
                        company_name = hubspot_info.get('company_name')
                    
                    if company_name:
                        industry = f"{company_name} industry"  # Fallback
                    else:
                        raise ValueError(f"Could not extract industry from deal {deal_id}")
                        
                _self.log_execution(f"Extracted industry '{industry}' from HubSpot deal {deal_id}")
            
            if not industry:
                raise ValueError("Industry/sector is required for market research")
            
            status_text.text("🔍 Researching market with enhanced intelligence...")
            progress_bar.progress(40)
            
            # Run the market intelligence agent
            _self.log_execution(f"Running market research for sector: {industry}")
            
            # Enhanced market research: pass HubSpot data if available
            hubspot_data = None
            if deal_id and _self.hubspot_available:
                try:
                    hubspot_data = _self._extract_company_info_from_hubspot(deal_id)
                    # Get full deal data for enhanced targeting
                    if hubspot_data:
                        deal_data = hubspot_client.get_deal_with_associated_data(deal_id)
                        hubspot_data = deal_data
                        _self.log_execution("Using HubSpot data for enhanced market targeting")
                except Exception as e:
                    _self.log_execution(f"Could not fetch HubSpot data for enhanced targeting: {e}")
            
            result = run_market_intelligence_cli(industry, hubspot_data=hubspot_data)
            
            status_text.text("📋 Compiling market analysis...")
            progress_bar.progress(90)
            
            status_text.text("✅ Market research completed!")
            progress_bar.progress(100)
            
            # Track execution time
            execution_time = time.time() - start_time
            st.session_state.execution_times['market_research'] = execution_time
            
            # Clean up progress indicators
            progress_bar.empty()
            status_text.empty()
            
            if result:
                # Convert Pydantic model to dict
                market_data = {
                    'status': 'success',
                    'market_analysis': result.model_dump(mode='json') if hasattr(result, 'model_dump') else result,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Save to cache
                _self._save_to_cache(cache_file_path, market_data)
                
                st.success(f"🎉 Market research completed in {execution_time:.1f} seconds")
                
                return market_data
            else:
                raise ValueError("Market research returned no results")
                
        except Exception as e:
            execution_time = time.time() - start_time
            st.session_state.execution_times['market_research'] = execution_time
            
            progress_bar.empty()
            status_text.empty()
            
            error_msg = f"Market research failed: {str(e)}"
            _self.log_execution(error_msg, "ERROR")
            st.error(f"❌ {error_msg}")
            
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def run_decision_support(self, company_research: Dict[str, Any] = None,
                           founder_research: Dict[str, Any] = None,
                           market_research: Dict[str, Any] = None,
                           deal_id: str = None) -> Dict[str, Any]:
        """Run decision support agent with progress tracking"""
        start_time = time.time()
        
        if 'execution_times' not in st.session_state:
            st.session_state.execution_times = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("🤔 Initializing decision analysis...")
            progress_bar.progress(15)
            
            self.log_execution("Running decision support analysis...")
            
            # Extract the actual data from the research results
            company_data = None
            if company_research:
                # Debug: log what we actually received
                self.log_execution(f"Company research status: {company_research.get('status')}")
                self.log_execution(f"Company research keys: {list(company_research.keys()) if isinstance(company_research, dict) else 'Not a dict'}")
                
                if company_research.get('status') == 'success':
                    # Try different possible locations for company data
                    company_data = company_research.get('company_profile')
                    if not company_data:
                        # Check if it's nested under a different key or is the direct result
                        company_data = company_research
                        # Remove status field if present to avoid confusion
                        if isinstance(company_data, dict) and 'status' in company_data:
                            company_data = {k: v for k, v in company_data.items() if k != 'status'}
                elif company_research.get('status') == 'error':
                    self.log_execution(f"Company research failed with error: {company_research.get('error')}")
                    raise ValueError(f"Company research failed: {company_research.get('error', 'Unknown error')}")
                else:
                    # Try to extract data even if status is not 'success'
                    if isinstance(company_research, dict) and any(key in company_research for key in ['company_name', 'company_profile']):
                        company_data = company_research
                        if 'status' in company_data:
                            company_data = {k: v for k, v in company_data.items() if k != 'status'}
            else:
                self.log_execution("No company research data provided")
            
            founder_data = []
            if founder_research and founder_research.get('status') == 'success':
                founder_data = founder_research.get('founders', [])
            
            market_data = None
            if market_research and market_research.get('status') == 'success':
                # Try different possible locations for market data
                market_data = market_research.get('market_analysis')
                if not market_data:
                    # Check if it's the direct result
                    market_data = market_research
                    # Remove status field if present
                    if isinstance(market_data, dict) and 'status' in market_data:
                        market_data = {k: v for k, v in market_data.items() if k != 'status'}
            
            # Debug logging to see what data we actually have
            self.log_execution(f"Company data available: {company_data is not None}")
            self.log_execution(f"Founder data count: {len(founder_data) if founder_data else 0}")
            self.log_execution(f"Market data available: {market_data is not None}")
            
            # Ensure we have at least company data
            if not company_data:
                raise ValueError("Company research data is required but not available")
            
            # Get additional context from HubSpot if deal_id provided
            additional_context = {}
            if deal_id and self.hubspot_available:
                try:
                    deal_data = hubspot_client.get_deal_with_associated_data(deal_id)
                    additional_context['hubspot_deal_data'] = deal_data
                except Exception as e:
                    logger.warning(f"Could not fetch HubSpot data for context: {e}")
            
            # Run the decision support analysis
            result = run_decision_support_analysis(
                company_data_dict=company_data,
                founder_data_list=founder_data,
                market_data_dict=market_data,
                additional_context=additional_context
            )
            
            status_text.text("🎯 Generating recommendation...")
            progress_bar.progress(90)
            
            status_text.text("✅ Decision analysis completed!")
            progress_bar.progress(100)
            
            # Track execution time
            execution_time = time.time() - start_time
            st.session_state.execution_times['decision_support'] = execution_time
            
            # Clean up progress indicators
            progress_bar.empty()
            status_text.empty()
            
            st.success(f"🎉 Decision support analysis completed in {execution_time:.1f} seconds")
            
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
            execution_time = time.time() - start_time
            st.session_state.execution_times['decision_support'] = execution_time
            
            progress_bar.empty()
            status_text.empty()
            
            error_msg = f"Decision support analysis failed: {str(e)}"
            self.log_execution(error_msg, "ERROR")
            st.error(f"❌ {error_msg}")
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

    # Add cache management UI to sidebar
    def display_cache_management_ui(self):
        """Display cache management controls in the sidebar"""
        with st.sidebar.expander("🗂️ Cache Management"):
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🗑️ Clear All", key="clear_all_cache", help="Clear all cached data"):
                    self.clear_cache()
            
            with col2:
                cache_types = st.selectbox("Cache Type:", ["All", "Company", "Founder", "Market", "Decision"], key="cache_type_select")
                if st.button("🗑️ Clear", key="clear_specific_cache"):
                    if cache_types == "All":
                        self.clear_cache()
                    else:
                        self.clear_cache(cache_types.lower())
            
            # Show cache stats
            cache_stats = self._get_cache_stats()
            if cache_stats:
                st.markdown("**Cache Statistics:**")
                for agent_type, stats in cache_stats.items():
                    st.text(f"{agent_type.title()}: {stats['files']} files, {stats['size_mb']:.1f} MB")

    def _get_cache_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get cache statistics for each agent type"""
        stats = {}
        cache_types = ['company', 'founder', 'market', 'decision']
        
        for agent_type in cache_types:
            cache_dir = f"cache/{agent_type}"
            if os.path.exists(cache_dir):
                files = [f for f in os.listdir(cache_dir) if f.endswith('.json')]
                total_size = sum(os.path.getsize(os.path.join(cache_dir, f)) for f in files)
                stats[agent_type] = {
                    'files': len(files),
                    'size_mb': total_size / (1024 * 1024)
                }
            else:
                stats[agent_type] = {'files': 0, 'size_mb': 0}
        
        return stats

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