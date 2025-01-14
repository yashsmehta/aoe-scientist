import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from thefuzz import fuzz
import re

def normalize_name(name):
    """Normalize a name by removing special chars, extra spaces, etc."""
    if pd.isna(name):
        return ""
    # Convert to lowercase and remove special characters
    name = re.sub(r'[^\w\s]', ' ', str(name).lower())
    # Remove extra whitespace
    name = ' '.join(name.split())
    return name

def is_name_match(name1, name2, threshold=80):
    """
    Check if two names match using various fuzzy matching techniques.
    Returns True if any matching method succeeds.
    """
    if pd.isna(name1) or pd.isna(name2):
        return False
        
    name1 = normalize_name(name1)
    name2 = normalize_name(name2)
    
    # Direct matches
    if name1 == name2:
        return True
    
    # Check if one name is contained within the other
    if name1 in name2 or name2 in name1:
        return True
    
    # Fuzzy ratio matching
    if fuzz.ratio(name1, name2) >= threshold:
        return True
        
    # Token sort ratio (handles reordered names better)
    if fuzz.token_sort_ratio(name1, name2) >= threshold:
        return True
    
    # Token set ratio (handles partial matches better)
    if fuzz.token_set_ratio(name1, name2) >= threshold:
        return True
    
    # Handle initials
    name1_parts = name1.split()
    name2_parts = name2.split()
    if len(name1_parts) > 1 and len(name2_parts) > 1:
        # Compare last names
        if fuzz.ratio(name1_parts[-1], name2_parts[-1]) >= threshold:
            # Check if initials match
            initials1 = ''.join(part[0] for part in name1_parts[:-1])
            initials2 = ''.join(part[0] for part in name2_parts[:-1])
            if initials1 == initials2:
                return True
    
    return False

def select_optimal_papers(df, researcher_name, n_papers=10, alpha=0.6, beta=0.4, penalty_weight=1.0):
    """
    Selects up to n_papers from df in which the given researcher_name
    is the first or last author within the last 5 years. We prioritize
    newer papers and papers with higher citations, while also maximizing
    diversity through a similarity penalty.

    Args:
        df (pd.DataFrame): Must contain columns ['year', 'authors', 'title', 'abstract', 'citations', 'first_author', 'last_author'].
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
    
    # Create mask for fuzzy name matching
    first_author_mask = df['first_author'].apply(lambda x: is_name_match(x, researcher_name))
    last_author_mask = df['last_author'].apply(lambda x: is_name_match(x, researcher_name))
    
    filtered_df = df[
        (df['year'] >= five_years_ago) &
        (first_author_mask | last_author_mask) &
        (~pd.isna(df['abstract']))  # Remove papers with NaN abstracts
    ].copy()

    if filtered_df.empty:
        print(f"No papers found for {researcher_name} in the last {current_year - five_years_ago} years")
        return []

    # Compute recency and citation scores
    filtered_df['years_ago'] = current_year - filtered_df['year']
    filtered_df['recency_score'] = np.exp(-0.5 * filtered_df['years_ago'])
    filtered_df['log_citations'] = np.log1p(filtered_df['citations'])

    # For stability
    max_log_citations = filtered_df['log_citations'].max()
    if pd.isna(max_log_citations) or max_log_citations <= 0:
        max_log_citations = 1.0

    # Normalize citation score
    filtered_df['citation_score'] = filtered_df['log_citations'] / max_log_citations

    # Base score combines recency and citation metrics
    filtered_df['base_score'] = alpha * filtered_df['recency_score'] + beta * filtered_df['citation_score']

    # Compute semantic embeddings for the diversity penalty
    documents = (
        filtered_df
        .apply(lambda x: f"{x['title']} {x['abstract']} {x['authors']}", axis=1)
        .fillna("")
    )

    # use sentence transformer to generate embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(documents.tolist(), show_progress_bar=False)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
    similarity_matrix = np.dot(embeddings, embeddings.T)

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
            "researcher": researcher_name,
            "title": str(paper["title"]),
            "year": int(paper["year"]),
            "citations": int(paper["citations"]),
            "first_author": str(paper["first_author"]),
            "last_author": str(paper["last_author"]),
            "abstract": str(paper["abstract"]),
        })

    return papers_context

def main():
    # Load the paper dump
    print("Loading paper dump...")
    df = pd.read_csv("data/paper_dump.csv")
    
    # Get unique researchers
    researchers = df['researcher'].dropna().unique()
    print(f"Found {len(researchers)} unique researchers")
    
    # Process papers for each researcher
    all_selected_papers = []
    for i, researcher in enumerate(researchers, 1):
        print(f"Processing researcher {i}/{len(researchers)}: {researcher}")
        selected_papers = select_optimal_papers(df, researcher)
        all_selected_papers.extend(selected_papers)
        
        # Print progress
        if selected_papers:
            print(f"  Selected {len(selected_papers)} papers")
        
    # Convert final results to DataFrame and save
    final_df = pd.DataFrame(all_selected_papers)
    final_df.to_csv("data/scholar_papers.csv", index=False)
    print(f"\nCompleted! Saved {len(all_selected_papers)} papers from {len(researchers)} researchers to data/scholar_papers.csv")

if __name__ == "__main__":
    main()
