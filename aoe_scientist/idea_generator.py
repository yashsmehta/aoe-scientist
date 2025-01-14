from langchain_core.messages import SystemMessage, HumanMessage
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
import json


def generate_research_idea(chat, cfg):
    """Generate a novel research idea based on existing papers."""
    response_schemas = [
        ResponseSchema(name="Name", description="A shortened descriptor of the idea. Lowercase, no spaces, underscores allowed."),
        ResponseSchema(name="Title", description="A title for the idea, will be used for the report writing."),
        ResponseSchema(name="Details", description="Provide detailed technical specifications and expand on the concept in a concise, straightforward manner, using 2-3 sentences."),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    
    # Load and format prompt template
    with open("prompts/idea_gen.json", "r") as f:
        prompt_data = json.load(f)
    
    # Create messages with format instructions
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