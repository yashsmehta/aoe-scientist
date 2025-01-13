import json
import os
import pandas as pd
from dotenv import load_dotenv
from .llm import create_client
from .idea_generator import generate_research_idea, ResearchIdea
from .idea_reviewer import review_idea
from .utils import setup_config, save_json

if __name__ == "__main__":
    cfg = setup_config()
    load_dotenv()
    
    chat = create_client(cfg['provider'])
    
    if cfg['mode'] == 'generate':
        output = generate_research_idea(chat, cfg.get('topic'), rag=cfg.get('rag', False))
        print(output)
        save_json(output, "data/ideas.json")

    elif cfg['mode'] == 'review':
        with open(cfg['idea_path']) as f:
            idea_data = json.load(f)
        idea = ResearchIdea(idea_data)
        output = review_idea(idea, chat, 
                           num_reflections=cfg.get('num_reflections', 1), 
                           num_reviews_ensemble=cfg.get('num_reviews_ensemble', 1))
        save_json(output, "data/reviews.json")
