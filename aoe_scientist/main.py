import pandas as pd
from aoe_scientist.llm import create_client
from aoe_scientist.idea_generator import generate_research_idea
from aoe_scientist.idea_reviewer import review_ideas
from aoe_scientist.utils import setup_config, save_df

def main():
    cfg = setup_config()
    
    if cfg['mode'] == 'generate':
        print("Generating ideas using: ", cfg['generate_llm'], "\nRAG: ", cfg['rag'], "\nResearcher: ", cfg['researcher'])
        chat = create_client(cfg['generate_llm'], temperature=1)
        ideas_df = pd.DataFrame()
        
        for _ in range(cfg['num_ideas']):
            idea_df = generate_research_idea(chat, cfg)
            ideas_df = pd.concat([ideas_df, idea_df], ignore_index=True)
            for _, row in idea_df.iterrows():
                print(f"\nGenerated idea:\nName: {row['name']}\nTitle: {row['title']}\nDetails: {row['details']}\n")
            
        save_df(ideas_df, 'data/ideas.csv')
    
    elif cfg['mode'] == 'review':
        print("\nReviewing ideas using: ", cfg['review_llm'])
        chat = create_client(cfg['review_llm'], temperature=0.25)
        reviews_df = review_ideas(chat, cfg)
        print(f"Reviews completed with average score: {reviews_df.overall_score.mean():.2f}")
        save_df(reviews_df, 'data/reviews.csv')

if __name__ == "__main__":
    main()
