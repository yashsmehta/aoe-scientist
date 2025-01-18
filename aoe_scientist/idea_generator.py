from langchain_core.messages import SystemMessage, HumanMessage
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from scripts.select_papers import is_name_match
import pandas as pd
import json

# Prompt templates for idea generation
RAG_SYSTEM_TEMPLATE = """You are the amazing AI researcher, {researcher}, tasked with generating novel and impactful \
research ideas in the field of {topic}. Put yourself in the researcher's mindset, and strictly follow the specified format \
for clarity and precision. It should be a novel idea that is not the same as prior work. 
Don't naively combine ideas from the prior work, use it wisely for your context.
Prior work is provided for context, so you know what the researcher has done.

Prior work: {papers}

{format_instructions}"""

RAG_HUMAN_TEMPLATE = """Using the following recent papers authored by {researcher}, provided to you, generate one \
creative and novel research idea in {topic}. The idea must align with the researcher's \
expertise but should be novel.

### Requirements:
1. Must demonstrate novelty and originality (not same as prior work).
2. Feasible using current technology.
3. Significant potential impact in the field.
4. Specific and actionable, avoiding vague concepts.
5. Technical depth, and should make sense.

### Response Format:
1. First provide your thought process and analysis in the Thought field
2. Give a concise name in lowercase with underscores in the Name field
3. Write a clear title in the Title field
4. Provide exactly 3 technical sentences in the Details field

Respond in the format specified in the system message."""

NON_RAG_SYSTEM_TEMPLATE = """You are an advanced research assistant tasked with generating novel and impactful research \
ideas in the field of {topic}. Your responses must demonstrate deep understanding of the \
field and strictly follow the specified format for clarity and precision.

{format_instructions}"""

NON_RAG_HUMAN_TEMPLATE = """Generate one creative and novel research idea in {topic}. The idea must demonstrate deep \
understanding of the field and push the boundaries of current research.

### Requirements:
1. Must demonstrate novelty and originality.
2. Feasible using current technology.
3. Significant potential impact in the field.
4. Specific and actionable, avoiding vague concepts.

### Response Format:
1. First provide your thought process and analysis in the Thought field
2. Give a concise name in lowercase with underscores in the Name field
3. Write a clear title in the Title field
4. Provide exactly 3 technical sentences in the Details field

Respond in the format specified in the system message."""

REFLECTION_SYSTEM_PROMPT = """You are a critical research idea evaluator and improver. Your task is to analyze research ideas \
and suggest concrete improvements while maintaining their core spirit. Be constructive but critical in your evaluation.

Your goal is to iteratively improve the research idea, particularly focusing on making the title more precise and the technical details, and
the problem being solved more specific and actionable.
Each iteration should meaningfully refine and enhance the idea - avoid superficial changes.

{format_instructions}"""

IDEA_REFLECTION_PROMPT = """Round {current_round}/{num_reflections}.

Current Research Idea:
Title: {title}
Name: {name}
Details: {details}
Previous Thought Process: {thought}

Carefully evaluate this research idea and suggest concrete improvements:

1. Title Improvement:
   - Is it specific enough to convey the unique approach?
   - Does it highlight the key technical innovation?
   - Can it be made more precise while staying concise?

2. Technical Details Enhancement:
   - Are the three sentences sufficiently technical and specific?
   - What is the technical justification of the idea?
   - Can you add what the problem is being solved?
   - Are there vague terms that can be replaced with specific techniques?

3. Critical Analysis:
   - Technical depth and soundness
   - Novelty and originality
   - Feasibility with current technology
   - Potential impact in the field
   - Clarity and specificity

Your task is to provide an improved version with:
1. What is the technical goal, and what is the problem being solved?
2. More detailed and concrete technical specifications
3. Very technical justification of the idea

### Response Format:
1. First provide your thought process and analysis in the Thought field
2. Give a concise name in lowercase with underscores in the Name field
3. Write an improved, more specific title in the Title field
4. Provide exactly 3 technical sentences in the Details field

If you truly believe the idea cannot be meaningfully improved, include "I am done" at the end of your Thought field and repeat the exact same JSON.

Respond in the format specified in the system message."""

def generate_research_idea(chat, cfg, num_reflections=3):
    """Generate a novel research idea based on existing papers.
    
    Args:
        chat: The chat model to use
        cfg: Configuration dictionary
        num_reflections: Number of reflection iterations to perform
    
    Returns:
        pd.DataFrame: DataFrame containing the generated idea
    """
    # Define response schema for structured output parsing
    response_schemas = [
        ResponseSchema(name="Thought", description="Your analysis and reasoning about the research idea, including specific critiques and suggested improvements."),
        ResponseSchema(name="Name", description="A short descriptor (lowercase, no spaces, underscores allowed)."),
        ResponseSchema(name="Title", description="A precise and specific title that clearly conveys the technical innovation."),
        ResponseSchema(name="Details", description="Technical specifications and implementation details in exactly 3 sentences. Each sentence should be specific and actionable, covering methodology, implementation approach, and expected outcomes.")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    
    if cfg['rag']:
        # Load RAG prompt and papers
        papers_df = pd.read_csv("data/scholar_papers.csv")
        papers_df = papers_df[papers_df['researcher'].apply(lambda x: is_name_match(x, cfg['researcher']))]
        papers_df = papers_df[['title', 'year', 'abstract']]
        papers_str = papers_df.to_json(orient='records', indent=2)
        
        messages = [
            SystemMessage(content=RAG_SYSTEM_TEMPLATE.format(
                researcher=cfg['researcher'],
                topic=cfg['topic'],
                papers=papers_str,
                format_instructions=format_instructions
            )),
            HumanMessage(content=RAG_HUMAN_TEMPLATE.format(
                researcher=cfg['researcher'],
                topic=cfg['topic']
            ))
        ]
    else:
        messages = [
            SystemMessage(content=NON_RAG_SYSTEM_TEMPLATE.format(
                topic=cfg['topic'],
                format_instructions=format_instructions
            )),
            HumanMessage(content=NON_RAG_HUMAN_TEMPLATE.format(
                topic=cfg['topic']
            ))
        ]

    def has_meaningful_changes(old_idea, new_idea):
        """Check if the new idea has meaningful changes from the old one."""
        # Check for significant changes in title (more than just minor word changes)
        title_changed = len(set(new_idea['Title'].split()) - set(old_idea['Title'].split())) >= 2
        
        # Check for significant changes in details
        details_changed = (
            len(set(new_idea['Details'].split()) - set(old_idea['Details'].split())) >= 5 or
            len(new_idea['Details']) - len(old_idea['Details']) >= 20
        )
        
        return title_changed or details_changed

    # Initial idea generation
    response = chat.invoke(messages)
    
    try:
        idea = output_parser.parse(response.content)
        print("\nInitial idea:")
        print(json.dumps(idea, indent=2))
        current_idea = idea
        
        # Reflection stage with separate message chain
        consecutive_no_changes = 0
        for i in range(num_reflections - 1):
            reflection_messages = [
                SystemMessage(content=REFLECTION_SYSTEM_PROMPT.format(
                    format_instructions=format_instructions
                )),
                HumanMessage(content=IDEA_REFLECTION_PROMPT.format(
                    current_round=i+2,
                    num_reflections=num_reflections,
                    title=current_idea.get('Title', ''),
                    name=current_idea.get('Name', ''),
                    details=current_idea.get('Details', ''),
                    thought=current_idea.get('Thought', '')
                ))
            ]
            
            reflection_response = chat.invoke(reflection_messages)
            try:
                reflected_idea = output_parser.parse(reflection_response.content)
                print(f"\nIteration {i+2}:")
                print(json.dumps(reflected_idea, indent=2))
                
                if "I am done" in reflected_idea.get('Thought', ''):
                    print(f"Idea generation converged after {i+2} iterations.")
                    break
                    
                # Check for meaningful changes
                if not has_meaningful_changes(current_idea, reflected_idea):
                    consecutive_no_changes += 1
                    print("No meaningful changes made in this iteration.")
                    if consecutive_no_changes >= 2:  # If no changes for 2 consecutive iterations
                        print("No meaningful changes for multiple iterations. Stopping reflection.")
                        break
                    if i < num_reflections - 2:  # If not the last iteration
                        continue
                else:
                    consecutive_no_changes = 0  # Reset counter when we see meaningful changes
                        
                current_idea = reflected_idea
            except Exception as e:
                print(f"Warning: Failed to parse reflection response: {str(e)}")
                print(f"Response content: {reflection_response.content[:200]}...")
                break
        
        idea_dict = {
            'name': current_idea.get('Name', ''),
            'generate_llm': cfg['generate_llm'],
            'researcher': cfg['researcher'],
            'rag': cfg['rag'],
            'title': current_idea.get('Title', ''),
            'details': current_idea.get('Details', ''),
            'thought': current_idea.get('Thought', '')
        }
        return pd.DataFrame([idea_dict])
    except Exception as e:
        print(f"Warning: Failed to parse response: {str(e)}\nResponse: {response.content}")
        print("Skipping this idea generation")
        return pd.DataFrame()