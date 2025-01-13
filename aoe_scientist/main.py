import json
import os
import pandas as pd
from dotenv import load_dotenv
from llm import create_client
from idea_generator import generate_research_idea, ResearchIdea
from idea_reviewer import review_idea
from utils import setup_argparse

if __name__ == "__main__":
    args = setup_argparse()
    load_dotenv()
    
    model = args.gen_model if args.mode == 'generate' else args.review_model
    client, model = create_client(model)
    print(f"Using model: {model}")
    print(f"Using client: {client}")
    
    if args.mode == 'generate':
        idea = generate_research_idea(pd.read_csv("scholar_paper.csv"), client, model, num_reflections=1)
        print(idea)
        if idea:
            filepath = "data/ideas.json"
            with open(filepath, 'w') as f:
                json.dump(idea.to_dict(), f, indent=2)
            print(f"Idea saved to: {filepath}")

    elif args.mode == 'review':
        with open(args.idea_path) as f:
            idea_data = json.load(f)
        idea = ResearchIdea(idea_data)
        
        result = review_idea(idea, client, model, num_reflections=1, num_reviews_ensemble=1)
        if result:
            review_path = "data/reviews.json"
            with open(review_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"Review saved to: {review_path}")
        else:
            print("Failed to generate review")
