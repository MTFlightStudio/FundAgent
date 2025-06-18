# ai_agents/tools/__init__.py

# Ensure .env is loaded for any tool that might need it at import time
# (though individual tool modules also call load_dotenv())
from dotenv import load_dotenv
load_dotenv()

from .web_search import search_tool, tavily_search_tool_instance
from .wikipedia import wiki_tool, wiki_tool_instance
from .save import save_tool
from .prospect import research_prospect_tool, relevance_ai_tool_configured
# from .classification import classify_email_tool, EmailClassification  # REMOVED - classification.py deleted

__all__ = [
    "search_tool",
    "tavily_search_tool_instance", # Expose for configuration checks
    "wiki_tool",
    "wiki_tool_instance",          # Expose for configuration checks
    "save_tool",
    "research_prospect_tool",
    "relevance_ai_tool_configured", # Expose for configuration checks
    # "classify_email_tool",          # REMOVED - classification.py deleted
    # "EmailClassification"           # REMOVED - classification.py deleted
]

# You can add a simple test here to ensure imports work
if __name__ == '__main__':
    print("Testing imports from ai_agents.tools...")
    print(f"search_tool available: {callable(search_tool)}")
    print(f"wiki_tool available: {callable(wiki_tool)}")
    print(f"save_tool available: {callable(save_tool)}")
    print(f"research_prospect_tool available: {callable(research_prospect_tool)}")
    # print(f"classify_email_tool available: {callable(classify_email_tool)}")  # REMOVED
    print("Tool imports seem OK.") 