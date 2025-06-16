import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import random
import sys # Ensure sys is imported for stderr logging in selector
from dotenv import load_dotenv

load_dotenv()

class ModelProvider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    # TOGETHER = "together" # Keeping this commented unless we add models for it
    
class TaskComplexity(Enum):
    SIMPLE = "simple"      # Classification, simple extraction, routing
    MEDIUM = "medium"      # Research synthesis, structured output, summarization
    COMPLEX = "complex"    # Deep analysis, complex reasoning, multi-step tasks

@dataclass
class ModelConfig:
    provider: ModelProvider
    model_name: str          # The actual model name string for the API call
    temperature: float
    max_tokens: int          # Max output tokens for generation
    # For cost_per_1k_tokens, we'll use a representative value, often input token cost or a slight average.
    # Actual cost calculation would need to distinguish input/output tokens.
    cost_per_1k_tokens: float # Estimated cost for 1k tokens (primarily based on input, or a blend)
    rate_limit_rpm: int      # Estimated requests per minute
    best_for: List[str]      # Keywords for task suitability
    context_window_tokens: Optional[int] = None # Full context window size

# Model definitions with their characteristics (as of hypothetical June 2025)
# Pricing updated based on user research.
MODEL_CONFIGS = {
    # --- Anthropic Models ---
    # Latest
    "claude-3.7-sonnet": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-3-7-sonnet-20250219",
        temperature=0.1,
        max_tokens=8192, 
        cost_per_1k_tokens=0.003, # Input: $0.003, Output: $0.015
        rate_limit_rpm=20, 
        best_for=["complex_reasoning", "research", "analysis", "long_form_generation"],
        context_window_tokens=200000
    ),
    "claude-3.5-haiku": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-3-5-haiku-20241022", 
        temperature=0.1,
        max_tokens=4096,
        cost_per_1k_tokens=0.0008, # Input: $0.0008, Output: $0.004
        rate_limit_rpm=100, 
        best_for=["simple_extraction", "classification", "fast_tasks"],
        context_window_tokens=200000
    ),
    # Legacy/Still Available Anthropic
    "claude-3.5-sonnet-v2": ModelConfig( # This was user-named, refers to claude-3-5-sonnet
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-3-5-sonnet-20241022", # User data lists this as "Claude 3.5 Sonnet v2"
                                                # and also separately "Claude 3.5 Sonnet (Original)" with a different date.
                                                # Using the date as per user data for v2.
        temperature=0.2,
        max_tokens=8192,
        cost_per_1k_tokens=0.003, # Input: $0.003, Output: $0.015 (Same as 3.7 Sonnet & original 3.5 Sonnet)
        rate_limit_rpm=30, 
        best_for=["research", "analysis", "structured_output", "legacy_3.5_sonnet"],
        context_window_tokens=200000
    ),
    "claude-3-opus": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-3-opus-20240229",
        temperature=0.2,
        max_tokens=8192,
        cost_per_1k_tokens=0.015, # Input: $0.015, Output: $0.075. Using input cost primarily.
        rate_limit_rpm=10, 
        best_for=["highest_stakes_reasoning", "deep_analysis", "legacy_opus"],
        context_window_tokens=200000
    ),
    "claude-3-sonnet": ModelConfig( # Original Sonnet
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-3-sonnet-20240229",
        temperature=0.2,
        max_tokens=4096,
        cost_per_1k_tokens=0.003, # Input: $0.003, Output: $0.015
        rate_limit_rpm=20, 
        best_for=["research", "analysis", "legacy_3.0_sonnet"],
        context_window_tokens=200000
    ),
    "claude-3-haiku": ModelConfig( # Original Haiku
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-3-haiku-20240307",
        temperature=0.1,
        max_tokens=4096,
        cost_per_1k_tokens=0.00025, # Input: $0.00025, Output: $0.00125
        rate_limit_rpm=60,
        best_for=["classification", "simple_extraction", "legacy_3.0_haiku"],
        context_window_tokens=200000
    ),

    # --- OpenAI Models ---
    # Latest GPT-4.1 Series
    "gpt-4.1": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4.1", # User provided name, actual API name might be versioned e.g. gpt-4.1-YYYYMMDD
        temperature=0.1,
        max_tokens=8192,
        cost_per_1k_tokens=0.002, # Input: $0.002, Output: $0.008
        rate_limit_rpm=40, 
        best_for=["flagship_multimodal", "complex_reasoning", "deep_analysis"],
        context_window_tokens=1000000
    ),
    "gpt-4.1-mini": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4.1-mini", # User provided name
        temperature=0.1,
        max_tokens=8192, 
        cost_per_1k_tokens=0.0004, # Input: $0.0004, Output: $0.0016
        rate_limit_rpm=100, 
        best_for=["fast_research", "general_purpose_strong", "structured_output"],
        context_window_tokens=128000 
    ),
    "gpt-4.1-nano": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4.1-nano", # User provided name
        temperature=0.0,
        max_tokens=4096,
        cost_per_1k_tokens=0.0002, # Input: ~$0.0002, Output: ~$0.0008
        rate_limit_rpm=200, 
        best_for=["classification", "simple_tasks", "high_volume", "autocompletion"],
        context_window_tokens=32000 
    ),
    # Reasoning Models (OpenAI 'o' series)
    "o3": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="o3", # User provided name
        temperature=0.0, 
        max_tokens=8192,
        cost_per_1k_tokens=0.015, # Input: ~$0.015, Output: ~$0.060
        rate_limit_rpm=20, 
        best_for=["complex_logical_tasks", "highest_stakes_reasoning"],
        context_window_tokens=128000 
    ),
    "o4-mini": ModelConfig( 
        provider=ModelProvider.OPENAI,
        model_name="o4-mini", # User provided name
        temperature=0.1,
        max_tokens=8192,
        cost_per_1k_tokens=0.005, # Input: ~$0.005, Output: ~$0.020
        rate_limit_rpm=50, 
        best_for=["strong_reasoning", "cost_effective_complex"],
        context_window_tokens=128000
    ),
    "o3-mini": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="o3-mini", # User provided name
        temperature=0.1,
        max_tokens=4096,
        cost_per_1k_tokens=0.005, # Input: ~$0.005, Output: ~$0.020
        rate_limit_rpm=60, 
        best_for=["reasoning_with_search_tool", "general_purpose_reasoning"],
        context_window_tokens=128000
    ),
    # GPT-4o Series (Still Available)
    "gpt-4o": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4o", # Keeping it generic; specific version like 'gpt-4o-2024-05-13' can be used if needed
        temperature=0.1,
        max_tokens=8192,
        cost_per_1k_tokens=0.003, # Input: $0.003, Output: $0.010 (was $0.005 in user data, new data shows $0.003 for input)
        rate_limit_rpm=60, 
        best_for=["multimodal_tasks", "legacy_strong_general"],
        context_window_tokens=128000
    ),
    "gpt-4o-mini": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4o-mini", # Keeping it generic; specific version like 'gpt-4o-mini-2024-07-18'
        temperature=0.1,
        max_tokens=4096,
        cost_per_1k_tokens=0.00015, # Input: $0.00015, Output: $0.0006
        rate_limit_rpm=100, 
        best_for=["legacy_fast_cheap", "simple_tasks", "high_volume"],
        context_window_tokens=128000
    ),
    # Legacy OpenAI Models
    "gpt-4-turbo": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4-turbo", # User data refers to gpt-4-turbo-preview. Often "gpt-4-turbo" points to latest turbo version.
                                  # Using generic "gpt-4-turbo". If specific like "gpt-4-turbo-2024-04-09" is needed, update model_name.
        temperature=0.1,
        max_tokens=8192,
        cost_per_1k_tokens=0.010, # Input: $0.010, Output: $0.030
        rate_limit_rpm=40,
        best_for=["legacy_complex_analysis", "legacy_research"],
        context_window_tokens=128000 
    ),
     "gpt-4": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4", 
        temperature=0.1,
        max_tokens=8192, # Standard gpt-4 has 8k, some versions 32k. Assuming 8k for general.
        cost_per_1k_tokens=0.030, # Input: $0.030, Output: $0.060
        rate_limit_rpm=20, # Lower than turbo
        best_for=["legacy_powerful", "original_gpt4_tasks"],
        context_window_tokens=8192 # Or 32768 depending on exact version used.
    ),
    "gpt-3.5-turbo": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-3.5-turbo-0125", # Specific version from previous config
        temperature=0.1,
        max_tokens=4096,
        cost_per_1k_tokens=0.0005, # Input: $0.0005, Output: $0.0015
        rate_limit_rpm=100,
        best_for=["legacy_general_purpose", "cost_effective_simple"],
        context_window_tokens=16385 
    ),
}

# Agent-specific model preferences - Remained the same as previous step, as they use keys from MODEL_CONFIGS
AGENT_MODEL_PREFERENCES = {
    "founder_research": {
        "primary": ["gpt-4.1", "claude-3.7-sonnet", "o3"], 
        "fallback": ["gpt-4.1-mini", "o4-mini", "claude-3.5-sonnet-v2"],
        "task_complexity": TaskComplexity.COMPLEX
    },
    "company_research": {
        "primary": ["gpt-4.1", "claude-3.7-sonnet", "o3"],
        "fallback": ["gpt-4.1-mini", "o4-mini", "claude-3.5-sonnet-v2"],
        "task_complexity": TaskComplexity.COMPLEX
    },
    "market_intelligence": {
        "primary": ["gpt-4.1", "claude-3.7-sonnet", "o3"],
        "fallback": ["gpt-4.1-mini", "o4-mini", "claude-3.5-sonnet-v2", "gpt-4o"],
        "task_complexity": TaskComplexity.COMPLEX
    },
    "decision_support": {
        "primary": ["gpt-4.1", "o3"],  
        "fallback": ["claude-3.7-sonnet", "gpt-4.1-mini", "o4-mini"],
        "task_complexity": TaskComplexity.COMPLEX
    },
    "email_classification": {
        "primary": ["gpt-4.1-nano", "claude-3.5-haiku", "gpt-4o-mini"], 
        "fallback": ["gpt-4.1-mini", "gpt-3.5-turbo"],
        "task_complexity": TaskComplexity.SIMPLE
    },
    "pdf_extraction": { 
        "primary": ["claude-3.7-sonnet", "gpt-4.1", "claude-3.5-sonnet-v2"],
        "fallback": ["gpt-4.1-mini", "o4-mini", "gpt-4o"],
        "task_complexity": TaskComplexity.MEDIUM
    },
    # Add other agent types here if needed
}

class ModelSelectionError(Exception):
    pass

class SmartModelSelector:
    def __init__(self):
        self.usage_tracking = {} 
        
    def get_available_models(self) -> List[str]:
        """Get list of model keys for models with valid API keys and defined in MODEL_CONFIGS"""
        available_keys = []
        
        # Check for Anthropic API Key and add relevant model keys
        if os.getenv("ANTHROPIC_API_KEY"):
            anthropic_models = [key for key, cfg in MODEL_CONFIGS.items() if cfg.provider == ModelProvider.ANTHROPIC]
            available_keys.extend(anthropic_models)
            
        # Check for OpenAI API Key and add relevant model keys
        if os.getenv("OPENAI_API_KEY"):
            openai_models = [key for key, cfg in MODEL_CONFIGS.items() if cfg.provider == ModelProvider.OPENAI]
            available_keys.extend(openai_models)
            
        # Filter out duplicates if a model key was somehow listed under multiple conditions (should not happen with current logic)
        return sorted(list(set(available_keys)))
    
    def select_model_for_agent(self, agent_name: str, prefer_fast: bool = False) -> ModelConfig:
        available_model_keys = self.get_available_models()
        if not available_model_keys:
            raise ModelSelectionError("No API keys found for any provider, or no models configured.")

        if agent_name not in AGENT_MODEL_PREFERENCES:
            print(f"Warning: No specific preferences found for agent '{agent_name}'. Using default model selection logic.", file=sys.stderr)
            return self._get_default_model(available_model_keys)
        
        preferences = AGENT_MODEL_PREFERENCES[agent_name]
        model_key_list_to_try = preferences.get("fallback", []) if prefer_fast else preferences.get("primary", [])
        
        # Try preferred/fallback models in order
        for model_key in model_key_list_to_try:
            if model_key in available_model_keys and model_key in MODEL_CONFIGS:
                config = MODEL_CONFIGS[model_key]
                if self._check_rate_limit(model_key): # Simplified rate limit check
                    self.log_model_selection(agent_name, config, "fallback" if prefer_fast else "primary_preferred")
                    return config
            elif model_key not in MODEL_CONFIGS:
                 print(f"Warning: Model key '{model_key}' listed for agent '{agent_name}' but not defined in MODEL_CONFIGS.", file=sys.stderr)

        # If primary choices failed (and not already trying fallback), try fallbacks
        if not prefer_fast and preferences.get("fallback"):
            print(f"Primary models for agent '{agent_name}' are unavailable or rate-limited. Trying fallbacks.", file=sys.stderr)
            fallback_key_list = preferences["fallback"]
            for model_key in fallback_key_list:
                if model_key in available_model_keys and model_key in MODEL_CONFIGS:
                    config = MODEL_CONFIGS[model_key]
                    if self._check_rate_limit(model_key):
                        self.log_model_selection(agent_name, config, "fallback_after_primary_fail")
                        return config
        
        # If still no model, try a general default model
        print(f"All specified primary/fallback models for agent '{agent_name}' are unavailable or rate-limited. Trying general default model.", file=sys.stderr)
        try:
            default_config = self._get_default_model(available_model_keys)
            self.log_model_selection(agent_name, default_config, "general_default_fallback")
            return default_config
        except ModelSelectionError as e_def:
            raise ModelSelectionError(f"No available or non-rate-limited models found for agent '{agent_name}' after all fallbacks: {e_def}")

    def _check_rate_limit(self, model_key: str) -> bool:
        return random.random() > 0.05  # 95% chance of being available (less aggressive for testing)
    
    def _get_default_model(self, available_model_keys: List[str]) -> ModelConfig:
        default_order = [
            # Prioritize fast, cheap, and good general purpose latest models
            "gpt-4.1-nano", "claude-3.5-haiku", "gpt-4o-mini", 
            # Then slightly more capable but still cost-effective
            "gpt-4.1-mini", "o3-mini", "claude-3.5-sonnet-v2",
            # Then older cost-effective options
            "gpt-3.5-turbo", "claude-3-haiku", 
            # Then more powerful/expensive options as general fallbacks
            "gpt-4o", "claude-3-sonnet", "o4-mini", 
            "claude-3.7-sonnet", "gpt-4.1", "o3", "claude-3-opus", "gpt-4-turbo"
        ]
        for model_key in default_order:
            if model_key in available_model_keys and model_key in MODEL_CONFIGS:
                config = MODEL_CONFIGS[model_key]
                if self._check_rate_limit(model_key): 
                    return config 
        raise ModelSelectionError("No available, non-rate-limited general default models found from the defined list.")

    def log_model_selection(self, agent_name: str, config: ModelConfig, selection_type: str):
        print(f"Agent '{agent_name}' selected model: {config.model_name} (Key: {next(key for key, val in MODEL_CONFIGS.items() if val == config)}, Type: {selection_type}, Provider: {config.provider.value})", file=sys.stderr)

# Factory function to create LLM instances
def create_llm_from_config(config: ModelConfig):
    """Create LLM instance from model config"""
    if config.provider == ModelProvider.ANTHROPIC:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
    elif config.provider == ModelProvider.OPENAI:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
    else:
        raise ValueError(f"Unsupported provider: {config.provider}")

def get_llm_for_agent(agent_name: str, prefer_fast: bool = False):
    selector = SmartModelSelector()
    try:
        config = selector.select_model_for_agent(agent_name, prefer_fast)
        llm = create_llm_from_config(config)
        return llm, config
    except ModelSelectionError as e:
        print(f"Error selecting model for agent '{agent_name}': {e}. Attempting emergency fallback.", file=sys.stderr)
        available_keys = selector.get_available_models()
        emergency_fallbacks = {
            ModelProvider.OPENAI: ["gpt-4o-mini", "gpt-3.5-turbo"],
            ModelProvider.ANTHROPIC: ["claude-3.5-haiku", "claude-3-haiku"]
        }
        for provider_type in [ModelProvider.OPENAI, ModelProvider.ANTHROPIC]: # Prioritize OpenAI for broader availability often
            if (provider_type == ModelProvider.OPENAI and os.getenv("OPENAI_API_KEY")) or \
               (provider_type == ModelProvider.ANTHROPIC and os.getenv("ANTHROPIC_API_KEY")):
                for model_key_fb in emergency_fallbacks[provider_type]:
                    if model_key_fb in available_keys and model_key_fb in MODEL_CONFIGS:
                        try:
                            emergency_config = MODEL_CONFIGS[model_key_fb]
                            # Check rate limit for emergency fallback too
                            if selector._check_rate_limit(model_key_fb): 
                                print(f"Emergency fallback for '{agent_name}': Using {emergency_config.model_name}. Reason: {e}", file=sys.stderr)
                                llm = create_llm_from_config(emergency_config)
                                selector.log_model_selection(agent_name, emergency_config, "emergency_fallback")
                                return llm, emergency_config
                        except Exception as e_fb_create:
                            print(f"Error creating emergency fallback LLM {model_key_fb} for {agent_name}: {e_fb_create}", file=sys.stderr)
                            continue # Try next emergency fallback model
        print(f"CRITICAL: No API keys available or all emergency fallbacks failed for agent '{agent_name}'. Original error: {e}", file=sys.stderr)
        raise # Re-raise the original ModelSelectionError if no emergency fallback is possible

if __name__ == "__main__":
    import sys # Ensure sys is imported for stderr
    # Test the model selection
    print("\n--- Model Selector Test ---", file=sys.stderr)
    selector = SmartModelSelector()
    print("Available models based on API keys & config:", selector.get_available_models(), file=sys.stderr)
    
    print("\nTesting model selection for each agent type:", file=sys.stderr)
    for agent_name_key in AGENT_MODEL_PREFERENCES.keys():
        try:
            llm_instance, model_config_instance = get_llm_for_agent(agent_name_key)
            if llm_instance:
                print(f"Agent '{agent_name_key}': Successfully got LLM for {model_config_instance.model_name} (Key: {next(key for key, val in MODEL_CONFIGS.items() if val == model_config_instance)} - ${model_config_instance.cost_per_1k_tokens:.5f}/1k tokens)", file=sys.stderr)
            else:
                # This case should ideally be handled by exceptions in get_llm_for_agent
                print(f"Agent '{agent_name_key}': Failed to create LLM instance for {model_config_instance.model_name} (LLM is None but no exception caught)", file=sys.stderr)
        except ModelSelectionError as e_sel:
            print(f"Agent '{agent_name_key}': ModelSelectionError - {e_sel}", file=sys.stderr)
        except Exception as e_gen:
            print(f"Agent '{agent_name_key}': General Exception - {e_gen}", file=sys.stderr)

    print("\nTesting with 'prefer_fast=True':", file=sys.stderr)
    for agent_name_key in AGENT_MODEL_PREFERENCES.keys():
        try:
            llm_instance, model_config_instance = get_llm_for_agent(agent_name_key, prefer_fast=True)
            if llm_instance:
                print(f"Agent '{agent_name_key}' (fast): Successfully got LLM for {model_config_instance.model_name} (Key: {next(key for key, val in MODEL_CONFIGS.items() if val == model_config_instance)} - ${model_config_instance.cost_per_1k_tokens:.5f}/1k tokens)", file=sys.stderr)
            else:
                print(f"Agent '{agent_name_key}' (fast): Failed to create LLM instance for {model_config_instance.model_name} (LLM is None)", file=sys.stderr)
        except ModelSelectionError as e_sel:
            print(f"Agent '{agent_name_key}' (fast): ModelSelectionError - {e_sel}", file=sys.stderr)
        except Exception as e_gen:
            print(f"Agent '{agent_name_key}' (fast): General Exception - {e_gen}", file=sys.stderr)
            
    print("\n--- End Model Selector Test ---\n", file=sys.stderr) 