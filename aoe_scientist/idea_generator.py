from langchain_core.messages import SystemMessage, HumanMessage
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
import json
import pandas as pd

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
        papers_str = df.to_json(orient='records', indent=2)
        
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