"""AOE Scientist - AI research idea generator and reviewer."""

__version__ = "0.1.0" 

from .llm import create_client
from .idea_generator import generate_research_idea, ResearchIdea
from .idea_reviewer import review_idea, IdeaReview
from .utils import setup_argparse
