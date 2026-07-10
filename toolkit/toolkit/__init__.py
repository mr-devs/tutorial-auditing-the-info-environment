"""
toolkit — shared utilities for the LLM fact-checking tutorial.

Core modules (metrics, response_structure, text, prompts, io, utils,
string_helpers) are copied from the llm-vs-human-fc-agreement project's
toolkit and lightly adapted; config and playwright_helper are tutorial-local.

Heavy-dependency submodules (providers, playwright_helper) are NOT imported
here so that lightweight consumers don't pay the SDK/browser import cost.
Import those submodules directly as needed:

    from toolkit.providers import openai_provider
    from toolkit.playwright_helper import scrape_article_text
"""

__version__ = "0.1.0"
__author__ = "Matthew DeVerna"

# Light-weight submodules safe to star-import.
from .metrics import *
from .utils import *
