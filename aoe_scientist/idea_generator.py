from langchain_core.messages import SystemMessage, HumanMessage
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def select_optimal_papers(df, researcher_name, n_papers=10, alpha=0.5, beta=0.5, penalty_weight=1.0):
    """
    Selects up to n_papers from df in which the given researcher_name
    is the first or last author within the last 5 years. We prioritize
    newer papers and papers with higher citations, while also maximizing
    diversity through a similarity penalty.

    Args:
        df (pd.DataFrame): Must contain columns ['year', 'authors', 'title', 'abstract', 'citations'].
        researcher_name (str): Name of researcher to filter by (as first or last author).
        n_papers (int): Desired number of selected papers. Default is 10.
        alpha (float): Weight for recency score. Default is 0.7.
        beta (float): Weight for citation score. Default is 0.3.
        penalty_weight (float): Multiplier for similarity penalty. Increase for stricter diversity.

    Returns:
        List of dictionaries with keys ['title', 'year', 'abstract', 'citations'].
    """

    current_year = pd.Timestamp.now().year

    # Filter by last 5 years and first/last author
    five_years_ago = current_year - 5
    
    # Convert researcher name to lowercase for case-insensitive matching
    researcher_name_lower = researcher_name.lower()
    
    filtered_df = df[
        (df['year'] >= five_years_ago) &
        (
            (df['first_author'].str.lower().str.contains(researcher_name_lower, na=False)) |
            (df['last_author'].str.lower().str.contains(researcher_name_lower, na=False))
        ) &
        (~pd.isna(df['abstract']))  # Remove papers with NaN abstracts
    ].copy()

    if filtered_df.empty:
        print(f"No papers found for {researcher_name} in the last {current_year - five_years_ago} years")
        return []

    # Compute recency and citation scores
    filtered_df['years_ago'] = current_year - filtered_df['year']
    filtered_df['recency_score'] = np.exp(-0.5 * filtered_df['years_ago'])
    filtered_df['log_citations'] = np.log1p(filtered_df['citations'])
    filtered_df['log_citations'] = np.log1p(filtered_df['citations'])

    # For stability
    max_log_citations = filtered_df['log_citations'].max()
    if pd.isna(max_log_citations) or max_log_citations <= 0:
        max_log_citations = 1.0

    # Normalize citation score
    filtered_df['citation_score'] = filtered_df['log_citations'] / max_log_citations

    # Base score combines recency and citation metrics
    filtered_df['base_score'] = alpha * filtered_df['recency_score'] + beta * filtered_df['citation_score']

    # Compute TF-IDF similarities for the diversity penalty
    documents = (
        filtered_df
        .apply(lambda x: f"{x['title']} {x['abstract']} {x['authors']}", axis=1)
        .fillna("")
    )

    tfidf = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=10000,
        norm='l2'
    )
    tfidf_matrix = tfidf.fit_transform(documents)
    similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()

    # Map from dataframe index to row in the similarity matrix
    idx_to_matrix = {idx: i for i, idx in enumerate(filtered_df.index)}

    # Initialize final scores (start equal to base_score)
    filtered_df['final_score'] = filtered_df['base_score']

    # Iteratively select papers, applying penalty for similarity
    selected_indices = []
    remaining_indices = set(filtered_df.index)
    selected_papers = []

    while len(selected_papers) < n_papers and remaining_indices:
        # Pick the paper with the highest current final_score
        idx_best = filtered_df.loc[list(remaining_indices), 'final_score'].idxmax()
        selected_indices.append(idx_best)
        selected_papers.append(filtered_df.loc[idx_best])
        remaining_indices.remove(idx_best)

        if remaining_indices:
            # Update final_score for the remaining papers based on similarity penalty
            selected_matrix_indices = [idx_to_matrix[idx] for idx in selected_indices]
            for r_idx in remaining_indices:
                matrix_idx = idx_to_matrix[r_idx]
                sims = similarity_matrix[matrix_idx, selected_matrix_indices]

                # Use mean of top-3 similarities
                if len(sims) >= 3:
                    top_k_sim = np.mean(np.sort(sims)[-3:])
                else:
                    top_k_sim = np.mean(sims)

                # Sigmoid-based penalty term in [0, 1]
                penalty = 2.0 / (1.0 + np.exp(-penalty_weight * top_k_sim)) - 1.0

                # Final score is base_score scaled down by (1 - penalty)
                filtered_df.at[r_idx, 'final_score'] = (
                    filtered_df.at[r_idx, 'base_score'] * (1.0 - penalty)
                )

    papers_context = []
    for paper in selected_papers[:n_papers]:
        papers_context.append({
            "title": str(paper["title"]),
            "year": int(paper["year"]),
            "abstract": str(paper["abstract"]),
            "citations": int(paper["citations"]),
        })

    return papers_context

def generate_research_idea(cfg, chat, rag=True):
    """Generate a novel research idea based on existing papers."""
    response_schemas = [
        ResponseSchema(name="Name", description="A shortened descriptor of the idea. Lowercase, no spaces, underscores allowed."),
        ResponseSchema(name="Title", description="A title for the idea, will be used for the report writing."),
        ResponseSchema(name="Details", description="Provide detailed technical specifications and potential implementation details. Expand on the concept in a concise, straightforward manner, using 2-3 sentences."),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    
    if rag:
        # Load RAG prompt and papers
        with open("prompts/idea_gen_rag.json", "r") as f:
            prompt_data = json.load(f)

        df = pd.read_csv("data/scholar_papers.csv")
        researcher_name = "Frank Hutter"
        print(f"Selecting 10 papers of {researcher_name}")
        papers = select_optimal_papers(df, researcher_name)
        papers_str = json.dumps(papers, indent=2)
        print(papers_str)
        exit()
        
        messages = [
            SystemMessage(content=prompt_data[0]["content"].replace("[FIELD]", cfg['topic']).replace("[SPECIFIC_AREA]", cfg.get('research_name', cfg['topic']))),
            HumanMessage(content=prompt_data[1]["content"].replace("[FIELD]", cfg['topic']).replace("[PAPERS]", papers_str) + "\n\nFormat Instructions: " + format_instructions)
        ]
    else:
        # Load and format regular prompt template
        with open("prompts/idea_gen.json", "r") as f:
            prompt_data = json.load(f)
        
        messages = [
            SystemMessage(content=prompt_data["messages"][0]["content"].format(
                topic=cfg['topic']
            )),
            HumanMessage(content=prompt_data["messages"][1]["content"].format(
                topic=cfg['topic']
            ) + "\n\nFormat Instructions: " + format_instructions)
        ]
    
    response = chat.invoke(messages)
    
    try:
        return output_parser.parse(response.content)
    except Exception as e:
        raise ValueError(f"Failed to parse response: {str(e)}\nResponse: {response.content}")