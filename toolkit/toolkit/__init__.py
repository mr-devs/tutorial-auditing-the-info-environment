"""
toolkit — shared utilities for the auditing-the-information-environment
tutorial (Guardian news → MCQ generation → LLM-as-judge → LLM horse race).

Submodules are imported explicitly by consumers so lightweight users don't
pay for heavy dependencies:

    from toolkit import guardian          # Guardian Content API collection
    from toolkit import config            # keys, paths
    from toolkit.utils import setup_logging
    from toolkit.providers import openai_provider
"""

__version__ = "0.2.0"
__author__ = "Matthew DeVerna"
