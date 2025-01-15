from langchain_core.messages import SystemMessage, HumanMessage
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from scripts.select_papers import is_name_match
import pandas as pd

# Prompt templates for idea generation
RAG_SYSTEM_TEMPLATE = """You are the amazing AI researcher, {researcher}, tasked with generating novel and impactful \
research ideas in the field of {topic}. Put yourself in the researcher's mindset, and strictly follow the specified format \
for clarity and precision. It should be a novel idea that is not the same as prior work. 
Don't naively combine ideas from the prior work, use it wisely for your context.
Prior work is just provided for \
context, you don't need to base your idea on it. It is provided just so you know what the researcher has done.

Prior work: {papers}

{format_instructions}"""

RAG_HUMAN_TEMPLATE = """Using the following recent papers authored by {researcher}, provided to you, generate one \
creative and novel research idea in {topic}. The idea must align with the researcher's \
expertise but should be novel.

### Requirements:
1. Must demonstrate novelty and originality (not same as prior work).
2. Feasible using currency technology.
3. Significant potential impact in the field.
4. Specific and actionable, avoiding vague concepts.
5. Technical depth, and should make sense.

### Respond in the following format:

#### THOUGHT
In <THOUGHT>, first briefly discuss your intuitions and motivations for the idea. \
Detail your high-level plan, necessary design choices and ideal outcomes of the experiments. \
Justify how the idea is different from the existing ones.

#### NEW IDEA JSON
The idea must be formatted according to the schema provided in the system message."""

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

### Respond in the following format:

#### THOUGHT
In <THOUGHT>, first briefly discuss your intuitions and motivations for the idea. \
Detail your high-level plan, necessary design choices and ideal outcomes of the experiments. \
Justify how the idea is different from the existing ones.

#### NEW IDEA JSON
The idea must be formatted according to the schema provided in the system message."""

def generate_research_idea(cfg, chat, rag=True):
    """Generate a novel research idea based on existing papers."""
    response_schemas = [
        ResponseSchema(name="Thought", description="The intuition, motivation and high-level plan for the research idea."),
        ResponseSchema(name="Name", description="A shortened descriptor of the idea. Lowercase, no spaces, underscores allowed."),
        ResponseSchema(name="Title", description="A title for the idea, will be used for the report writing."),
        ResponseSchema(name="Details", description="Provide detailed technical specifications and potential implementation details. Expand on the concept in a concise, straightforward manner, using 2-3 sentences."),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    
    if rag:
        # Load RAG prompt and papers
        df = pd.read_csv("data/scholar_papers.csv")
        df = df[df['researcher'].apply(lambda x: is_name_match(x, cfg['researcher']))]
        df = df[['title', 'year', 'abstract']]
        papers_str = df.to_json(orient='records', indent=2)
        
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
    
    response = chat.invoke(messages)
    
    try:
        return output_parser.parse(response.content)
    except Exception as e:
        raise ValueError(f"Failed to parse response: {str(e)}\nResponse: {response.content}")