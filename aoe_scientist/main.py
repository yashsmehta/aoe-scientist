import json
from aoe_scientist.llm import create_client
from aoe_scientist.idea_generator import generate_research_idea
from aoe_scientist.idea_reviewer import review_idea
from aoe_scientist.utils import setup_config, save_json

if __name__ == "__main__":
    cfg = setup_config()
    
    if cfg['mode'] == 'generate':
        chat = create_client("deepseek", temperature=1)
        output = generate_research_idea(cfg, chat, rag=True)
        print(output)
        save_json(output, "data/ideas.json")

    elif cfg['mode'] == 'review':
        chat = create_client("deepseek", temperature=0.25)
        with open(cfg['idea_path']) as f:
            idea_data = json.load(f)
        output = review_idea(idea_data, chat, 
                           num_reflections=cfg['num_reflections'], 
                           num_reviews_ensemble=cfg['num_reviews_ensemble'])
        save_json(output, "data/reviews.json")
