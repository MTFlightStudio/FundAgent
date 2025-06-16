import time
import random
import logging
from typing import Callable, Any, Optional, Dict, List
from functools import wraps
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class RetryReason(Enum):
    RATE_LIMIT = "rate_limit"
    NETWORK_ERROR = "network_error"
    MODEL_ERROR = "model_error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"

@dataclass
class RetryConfig:
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    switch_model_on_rate_limit: bool = True

class RetryableError(Exception):
    def __init__(self, message: str, reason: RetryReason, retry_after: Optional[float] = None):
        self.reason = reason
        self.retry_after = retry_after
        super().__init__(message)

def classify_error(error: Exception) -> RetryReason:
    """Classify errors to determine retry strategy"""
    error_str = str(error).lower()
    
    if "rate limit" in error_str or "429" in error_str:
        return RetryReason.RATE_LIMIT
    elif "timeout" in error_str or "connection" in error_str:
        return RetryReason.NETWORK_ERROR
    elif "model" in error_str or "api" in error_str: # More generic for API related issues beyond just "model"
        return RetryReason.MODEL_ERROR
    else:
        return RetryReason.UNKNOWN

def extract_retry_after(error: Exception) -> Optional[float]:
    """Extract retry-after time from error if available"""
    error_str = str(error)
    
    # Look for rate limit specific retry times
    # This part is highly dependent on the actual error message format from APIs
    # For now, a generic check. More specific parsing might be needed.
    if "rate limit" in error_str.lower():
        # Example: "Rate limit exceeded. Try again in 60 seconds."
        import re
        match = re.search(r"try again in (\d+)", error_str, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return 60.0 # Default if specific time not found in rate limit message
    
    return None

class SmartRetryHandler:
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
        # self.model_failures = {} # This was in the original but not used, can be added if per-model failure tracking is needed.
        
    def calculate_delay(self, attempt: int, reason: RetryReason, retry_after: Optional[float] = None) -> float:
        """Calculate delay based on retry reason and attempt number"""
        
        if retry_after:
            # Use the server-suggested retry-after time if available
            delay = retry_after
        elif reason == RetryReason.RATE_LIMIT:
            # For rate limits, use a potentially longer base or respect exponential backoff capped higher
            delay = min(self.config.base_delay * (self.config.exponential_base ** attempt), 120.0) # Cap at 2 mins for rate limits
        else:
            # Standard exponential backoff
            delay = min(self.config.base_delay * (self.config.exponential_base ** attempt), self.config.max_delay)
        
        if self.config.jitter:
            # Add jitter: random percentage of the delay (e.g., 0% to 50% of delay)
            jitter_amount = random.uniform(0, 0.5) * delay
            delay += jitter_amount
            
        return delay

    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if we should retry based on error type and attempt count"""
        if attempt >= self.config.max_retries:
            return False
            
        reason = classify_error(error)
        
        # Always retry rate limits and network errors (up to max_retries)
        if reason in [RetryReason.RATE_LIMIT, RetryReason.NETWORK_ERROR]:
            return True
            
        # Retry model errors up to a certain number of times (e.g., 2 as in original)
        # Could be configured in RetryConfig if needed
        if reason == RetryReason.MODEL_ERROR and attempt < 2: 
            return True
        
        # Optionally, retry UNKNOWN errors for one attempt to catch transient issues
        if reason == RetryReason.UNKNOWN and attempt < 1:
             return True
            
        return False

def with_smart_retry(
    agent_name: str,
    retry_config_override: Optional[RetryConfig] = None, # Renamed to avoid conflict
    model_selector_func: Optional[Callable] = None # Renamed for clarity
):
    """
    Decorator for intelligent retries with optional model switching.
    Args:
        agent_name (str): Name of the agent for logging.
        retry_config_override (Optional[RetryConfig]): Specific retry config for this agent.
        model_selector_func (Optional[Callable]): Function to get a new LLM and its config.
                                                   Expected signature: (agent_name: str, prefer_fast: bool) -> (LLM, ModelConfig)
    """
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_retry_config = retry_config_override or RetryConfig()
            retry_handler = SmartRetryHandler(current_retry_config)
            
            current_llm = kwargs.get('llm')
            current_model_config_details = kwargs.get('model_config')

            # --- Added: Ensure initial LLM if not provided and selector exists ---
            if current_llm is None and model_selector_func:
                try:
                    logger.info(f"Initial LLM not provided for {agent_name}. Attempting to fetch with model_selector_func.")
                    initial_llm, initial_model_config = model_selector_func(agent_name, prefer_fast=False) # Use prefer_fast=False for initial selection
                    current_llm = initial_llm
                    current_model_config_details = initial_model_config
                    # Update kwargs for the first call if the decorated function expects them explicitly
                    # This is important if the function signature includes llm or model_config directly.
                    # For **kwargs, this ensures they are present if the function tries to .get() them.
                    kwargs['llm'] = current_llm
                    kwargs['model_config'] = current_model_config_details
                    logger.info(f"Fetched initial LLM for {agent_name}: {initial_model_config.model_name}")
                except Exception as e_initial_fetch:
                    logger.error(f"Failed to fetch initial LLM for {agent_name} using model_selector_func: {e_initial_fetch}")
                    # Decide if we should raise, or let the function be called without an LLM (it might handle it)
                    # For now, let it proceed, the decorated function will then fail if it strictly needs an LLM.
            # --- End Added --- 

            for attempt in range(current_retry_config.max_retries + 1):
                try:
                    # Logic for attempting model switch
                    if attempt > 0 and model_selector_func and current_retry_config.switch_model_on_rate_limit:
                        # We'd typically switch if the previous error was a rate limit,
                        # or if a model seems persistently problematic.
                        # For simplicity, this example switches on any retry if switch_model_on_rate_limit is true.
                        # More nuanced logic could be added based on `reason` from previous attempt.
                        
                        logger.info(f"Retrying for {agent_name} (attempt {attempt + 1}). Attempting model switch.")
                        try:
                            # prefer_fast=True is a good default when switching due to issues
                            new_llm, new_model_config = model_selector_func(agent_name, prefer_fast=True)
                            
                            # Update the LLM instance and its config in kwargs if the decorated function expects them
                            if 'llm' in kwargs: # Check if 'llm' is an expected kwarg
                                kwargs['llm'] = new_llm
                            if 'model_config' in kwargs: # Check for 'model_config' as well
                                kwargs['model_config'] = new_model_config
                            
                            current_llm = new_llm # Keep track for logging or other uses
                            current_model_config_details = new_model_config

                            logger.info(f"Switched model for {agent_name} to: {new_model_config.model_name}")
                        except Exception as e_switch:
                            logger.warning(f"Failed to switch models for {agent_name} during retry: {e_switch}")
                            # Continue with the current/original LLM if switching fails

                    # Execute the wrapped function
                    # Ensure the most current llm and model_config are used if they were part of kwargs
                    if 'llm' in func.__code__.co_varnames and 'llm' not in kwargs and current_llm:
                        kwargs['llm'] = current_llm
                    if 'model_config' in func.__code__.co_varnames and 'model_config' not in kwargs and current_model_config_details:
                         kwargs['model_config'] = current_model_config_details
                        
                    result = func(*args, **kwargs)
                    
                    if attempt > 0: # Log successful recovery
                        logger.info(f"Successfully recovered {agent_name} after {attempt} retries.")
                    
                    return result
                    
                except Exception as error:
                    reason = classify_error(error)
                    retry_after = extract_retry_after(error)
                    
                    # Check if we should retry this specific error and attempt
                    if not retry_handler.should_retry(error, attempt):
                        logger.error(f"Max retries ({current_retry_config.max_retries}) reached or non-retryable error for {agent_name}. Last error: {error}", exc_info=True)
                        raise # Re-raise the last error
                    
                    # Calculate delay for the next attempt
                    delay = retry_handler.calculate_delay(attempt + 1, reason, retry_after) # Pass attempt + 1 for delay calc
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{current_retry_config.max_retries} failed for {agent_name}. "
                        f"Reason: {reason.value}. Retrying in {delay:.2f}s. Error: {str(error)}"
                    )
                    
                    time.sleep(delay)
            
            # This line should ideally not be reached if max_retries is handled correctly.
            # If it is, it means all retries (0 to max_retries) failed.
            logger.error(f"All {current_retry_config.max_retries + 1} attempts failed for {agent_name}.")
            raise RuntimeError(f"All retry attempts failed for {agent_name}. See logs for details.") # Should have been re-raised above
        
        return wrapper
    return decorator

# Helper to create an agent with retry (simplified for direct function wrapping)
def create_agent_with_retry(
    agent_name: str, 
    agent_function: Callable, 
    retry_config: Optional[RetryConfig] = None,
    pass_llm_and_config: bool = False # If true, will fetch and pass llm and model_config to agent_function
):
    """
    Wraps an agent function with retry logic and optional model selection.
    Args:
        agent_name (str): Name of the agent.
        agent_function (Callable): The function to wrap.
        retry_config (Optional[RetryConfig]): Custom retry configuration.
        pass_llm_and_config (bool): If True, model_selector will be used to fetch
                                    llm & config and pass them as kwargs to agent_function.
    Returns:
        Callable: The enhanced agent function.
    """
    model_selector_instance = None
    try:
        from ai_agents.config.model_config import get_llm_for_agent
        model_selector_instance = get_llm_for_agent
    except ImportError:
        logger.warning("model_config.get_llm_for_agent not found. Model switching in retries will not be available.")

    # Determine the effective retry_config
    effective_retry_config = retry_config or RetryConfig()
    if model_selector_instance is None: # If no model selector, disable model switching
        effective_retry_config.switch_model_on_rate_limit = False

    @with_smart_retry(agent_name, effective_retry_config, model_selector_instance)
    def enhanced_agent_func(*args, **kwargs):
        # If pass_llm_and_config is True, and llm/model_config are not already provided, fetch them.
        # The decorator itself will handle passing these if they are already in kwargs or if switched.
        if pass_llm_and_config and model_selector_instance:
            if 'llm' not in kwargs and 'model_config' not in kwargs:
                try:
                    llm, m_config = model_selector_instance(agent_name)
                    kwargs['llm'] = llm
                    kwargs['model_config'] = m_config
                    logger.info(f"Initial LLM ({m_config.model_name}) provided to {agent_name} by create_agent_with_retry.")
                except Exception as e_fetch:
                    logger.error(f"Failed to fetch initial LLM for {agent_name}: {e_fetch}")
                    # Depending on strictness, could raise here or let it proceed without LLM
                    # For now, it will proceed, and if agent_function requires llm, it will fail there.

        return agent_function(*args, **kwargs)
    
    return enhanced_agent_func

# Example of a Mixin class (less direct usage compared to decorator, but can be useful)
class AgentRetryMixin:
    """
    Mixin class to add retry capabilities to agent classes that manage their own execution calls.
    Assumes the class using this mixin will have `self.agent_name` and can access `get_llm_for_agent`.
    """
    
    def _get_model_selector(self):
        try:
            from ai_agents.config.model_config import get_llm_for_agent
            return get_llm_for_agent
        except ImportError:
            logger.warning(f"Model selector not available for AgentRetryMixin used by {getattr(self, 'agent_name', 'Unknown Agent')}.")
            return None

    def execute_with_retry(self, func_to_execute: Callable, *args, **kwargs) -> Any:
        """
        Executes a given function with smart retry logic.
        The function `func_to_execute` is expected to handle its own LLM if needed,
        or the LLM can be passed in `**kwargs` and the decorator will attempt to update it.
        """
        agent_name_val = getattr(self, 'agent_name', "mixin_agent") # Get agent_name from instance
        
        # Default retry config for the mixin, can be overridden by instance if desired
        mixin_retry_config = getattr(self, 'retry_config', RetryConfig())
        
        model_selector_for_mixin = self._get_model_selector()
        if model_selector_for_mixin is None: # Disable model switching if selector not found
            mixin_retry_config.switch_model_on_rate_limit = False
            
        # Dynamically apply the decorator to the function we want to execute
        # This is a bit more complex than direct decoration but provides flexibility
        
        decorated_func = with_smart_retry(
            agent_name=agent_name_val,
            retry_config_override=mixin_retry_config,
            model_selector_func=model_selector_for_mixin
        )(func_to_execute)
            
        return decorated_func(*args, **kwargs)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    
    # --- Test with_smart_retry decorator ---
    print("\n--- Testing @with_smart_retry decorator ---")
    
    # Mock get_llm_for_agent for testing model switching
    class MockModelConfig:
        def __init__(self, name):
            self.model_name = name
            self.provider = "mock_provider"
            self.temperature = 0.1
            self.max_tokens = 100

    def mock_model_selector(agent_name_arg, prefer_fast=False):
        # Simulate model switching
        if not hasattr(mock_model_selector, 'switch_count'):
            mock_model_selector.switch_count = 0
        
        mock_model_selector.switch_count += 1
        model_name_to_return = f"switched_model_v{mock_model_selector.switch_count}"
        if prefer_fast:
            model_name_to_return += "_fast"
            
        logger.info(f"(Mock Selector) For {agent_name_arg}, returning: {model_name_to_return}")
        # Returns a dummy LLM object and a MockModelConfig
        return f"LLM_for_{model_name_to_return}", MockModelConfig(model_name_to_return)

    current_test_config = RetryConfig(
        max_retries=3, 
        base_delay=0.1, 
        switch_model_on_rate_limit=True # Enable model switching for this test
    )

    @with_smart_retry("flaky_test_agent", current_test_config, mock_model_selector)
    def flaky_function_with_llm(fail_count: int = 2, llm=None, model_config=None):
        """Test function that fails, simulates needing an LLM, and shows model switching."""
        if not hasattr(flaky_function_with_llm, 'attempt'):
            flaky_function_with_llm.attempt = 0
        flaky_function_with_llm.attempt += 1
        
        current_model_name = model_config.model_name if model_config else "N/A"
        logger.info(f"flaky_function_with_llm attempt {flaky_function_with_llm.attempt} with LLM: {llm}, Model: {current_model_name}")
        
        if flaky_function_with_llm.attempt <= fail_count:
            if flaky_function_with_llm.attempt == 1:
                raise Exception("rate limit") # Triggers model switch if configured
            else:
                raise Exception("some other network error")
        return f"Success on attempt {flaky_function_with_llm.attempt} with model {current_model_name}"

    try:
        # Initial call can include llm and model_config if the function expects them
        initial_llm, initial_config = "initial_llm_obj", MockModelConfig("initial_model")
        result = flaky_function_with_llm(fail_count=2, llm=initial_llm, model_config=initial_config)
        print(f"Result (flaky_function_with_llm): {result}")
    except Exception as e:
        print(f"Final failure (flaky_function_with_llm): {e}")
    
    # Reset attempt counter for next test
    if hasattr(flaky_function_with_llm, 'attempt'):
        del flaky_function_with_llm.attempt
    if hasattr(mock_model_selector, 'switch_count'):
         del mock_model_selector.switch_count

    print("\n--- Testing create_agent_with_retry helper ---")
    def simple_callable_task(task_name: str, llm=None, model_config=None):
        logger.info(f"Executing simple_callable_task: {task_name} with LLM: {llm}, Model: {model_config.model_name if model_config else 'N/A'}")
        if not hasattr(simple_callable_task, 'call_count'):
            simple_callable_task.call_count = 0
        simple_callable_task.call_count +=1
        
        if simple_callable_task.call_count == 1:
            raise Exception("API error, please retry") # A model_error type
        return f"Task '{task_name}' completed on call {simple_callable_task.call_count} with {model_config.model_name if model_config else 'N/A'}"

    # Wrap the simple_callable_task
    # pass_llm_and_config=True means get_llm_for_agent will be called by the helper to provide these
    enhanced_task = create_agent_with_retry(
        "created_agent", 
        simple_callable_task,
        retry_config=RetryConfig(base_delay=0.2, max_retries=2),
        pass_llm_and_config=True # This will make create_agent_with_retry call get_llm_for_agent
                                 # (which we mocked with mock_model_selector via monkeypatching below for this test)
    )
    
    # For this test, temporarily replace the real get_llm_for_agent
    # so create_agent_with_retry uses our mock for model provision.
    try:
        import ai_agents.config.model_config as mc_module
        original_get_llm = mc_module.get_llm_for_agent
        mc_module.get_llm_for_agent = mock_model_selector # Monkeypatch

        result_created = enhanced_task(task_name="MyImportantTask")
        print(f"Result (created_agent): {result_created}")
    except Exception as e_created:
        print(f"Final failure (created_agent): {e_created}")
    finally:
        if 'original_get_llm' in locals(): # Restore original if it was patched
             mc_module.get_llm_for_agent = original_get_llm
        if hasattr(simple_callable_task, 'call_count'):
            del simple_callable_task.call_count
        if hasattr(mock_model_selector, 'switch_count'):
            del mock_model_selector.switch_count


    print("\n--- Testing AgentRetryMixin ---")
    class MyTestAgent(AgentRetryMixin): # AgentRetryMixin should be first for MRO if super() is used
        def __init__(self, name):
            # super().__init__() # If AgentRetryMixin had its own __init__ with super()
            self.agent_name = name # Required by the mixin for logging
            self.call_tracker = 0
            # self.retry_config = RetryConfig(max_retries=1) # Optionally override default retry config

        def actual_work(self, data: str, llm=None, model_config=None):
            self.call_tracker += 1
            current_model_name = model_config.model_name if model_config else "N/A"
            logger.info(f"MyTestAgent '{self.agent_name}' doing actual_work (attempt {self.call_tracker}) with data: '{data}', Model: {current_model_name}")
            if self.call_tracker == 1:
                raise Exception("Simulated timeout on first call")
            return f"Work done by {self.agent_name} on attempt {self.call_tracker} with {current_model_name}"

        def do_something_risky(self, data: str):
            # The method to be executed with retry needs to accept llm and model_config
            # if model switching is desired and the mixin/decorator is to provide them.
            
            # If the mixin's execute_with_retry is to provide the LLM,
            # we must ensure it's passed into actual_work.
            # Here, we assume the initial LLM comes from an external source or is set up on the agent.
            # The model_selector in with_smart_retry will try to update it in kwargs.
            
            initial_llm_mixin, initial_config_mixin = "initial_llm_for_mixin", MockModelConfig("initial_mixin_model")

            return self.execute_with_retry(self.actual_work, data, llm=initial_llm_mixin, model_config=initial_config_mixin)

    test_agent_instance = MyTestAgent("DataProcessorAgent")
    try:
        # For the mixin test to demonstrate model switching, its _get_model_selector
        # should return our mock_model_selector.
        # We can achieve this by patching where it looks for get_llm_for_agent.
        import ai_agents.config.model_config as mc_module_mixin
        original_get_llm_mixin = mc_module_mixin.get_llm_for_agent
        mc_module_mixin.get_llm_for_agent = mock_model_selector # Monkeypatch for mixin's selector
        
        mixin_result = test_agent_instance.do_something_risky("important_payload")
        print(f"Result (AgentRetryMixin): {mixin_result}")
    except Exception as e_mixin:
        print(f"Final failure (AgentRetryMixin): {e_mixin}")
    finally:
        if 'original_get_llm_mixin' in locals():
            mc_module_mixin.get_llm_for_agent = original_get_llm_mixin
        if hasattr(mock_model_selector, 'switch_count'):
            del mock_model_selector.switch_count
    
    print("\n--- Retry Handler Tests Complete ---") 