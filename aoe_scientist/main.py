import json
import pandas as pd
from aoe_scientist.llm import create_client
from aoe_scientist.idea_generator import generate_research_idea
from aoe_scientist.idea_reviewer import review_idea
from aoe_scientist.utils import setup_config, save_json

def main():
    cfg = setup_config()
    
    if cfg['mode'] == 'generate':
        chat = create_client(cfg['generate_llm'], temperature=1)
        ideas = []
        num_ideas = cfg.get('num_ideas', 1)
        
        for _ in range(num_ideas):
            idea = generate_research_idea(cfg, chat, rag=True)
            ideas.append(idea)
            print(f"Generated idea:\n{idea}\n")
            
        save_json(ideas, f"data/ideas/{cfg['researcher']}.json")
    
    elif cfg['mode'] == 'review':
        chat = create_client(cfg['review_llm'], temperature=0.25)
        review, reviews_df = review_idea(chat, cfg)
        print(f"Review completed with overall score: {review.overall_score}")
        save_json(review.model_dump(), "data/reviews.json")
        reviews_df.to_csv('data/reviews.csv', index=False)

if __name__ == "__main__":
    main()
