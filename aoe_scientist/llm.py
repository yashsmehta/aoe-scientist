from typing import List, Dict, Optional, Tuple
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import os
import logging
import json

logger = logging.getLogger(__name__)

class IdeaOutput(BaseModel):
    """Schema for research idea output."""
    Name: str = Field(description="A shortened descriptor of the idea")
    Title: str = Field(description="A title for the idea")
    Experiment: str = Field(description="An outline of the implementation")

class ReviewOutput(BaseModel):
    """Schema for review output."""
    novelty: int = Field(description="Scientific Merit and Innovation score (1-10)")
    feasibility: int = Field(description="Feasibility and Resources score (1-10)")
    impact: int = Field(description="Potential impact score (1-10)")
    ethical: int = Field(description="Ethical considerations score (1-10)")
    overall_score: int = Field(description="Overall score (1-10)")

def create_client(provider: str, temperature: float = 0.75) -> ChatOpenAI:
    provider_configs = {
        "deepseek": {
            "class": ChatOpenAI,
            "api_key_env": "DEEPSEEK_API_KEY",
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-chat"
        },
        "anthropic": {
            "class": ChatAnthropic,
            "api_key_env": "ANTHROPIC_API_KEY",
            "model": "claude-3-5-sonnet-latest"
        },
        "openai": {
            "class": ChatOpenAI,
            "api_key_env": "OPENAI_API_KEY",
            "model": "gpt-4o"
        }
    }
    
    provider = provider.lower()
    if provider not in provider_configs:
        raise ValueError(f"Provider {provider} not supported. Must be one of: {', '.join(provider_configs.keys())}")
    
    config = provider_configs[provider]
    
    # Get API key
    api_key = os.environ.get(config["api_key_env"])
    if not api_key:
        raise ValueError(f"Missing {config['api_key_env']} environment variable")
    
    # Create model instance
    kwargs = {
        "model": config["model"],
        "temperature": temperature,
        "api_key": api_key
    }
    
    if "base_url" in config:
        kwargs["base_url"] = config["base_url"]
        
    chat = config["class"](**kwargs)
    print(f"Using LangChain with {provider} model: {config['model']}")
    return chat

def get_batch_responses_from_llm(
        msg: str,
        client: ChatOpenAI,
        model: str,
        system_message: str,
        print_debug: bool = False,
        msg_history: Optional[List[Dict]] = None,
        temperature: float = 0.75,
        n_responses: int = 1,
) -> Tuple[List[str], List[List[Dict]]]:
    """Get multiple responses using LangChain."""
    if msg_history is None:
        msg_history = []
        
    messages = [SystemMessage(content=system_message)]
    
    for m in msg_history:
        if m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            messages.append(AIMessage(content=m["content"]))
            
    messages.append(HumanMessage(content=msg))
    
    contents = []
    histories = []
    
    parser = JsonOutputParser(pydantic_object=IdeaOutput)
    format_instructions = parser.get_format_instructions()
    
    for _ in range(n_responses):
        try:
            response = client.invoke(messages + [HumanMessage(content=format_instructions)])
            content = response.content
            
            # Extract JSON from markdown-formatted response if needed
            if "```json" in content:
                json_start = content.find('```json') + 7
                json_end = content.find('```', json_start)
                content = content[json_start:json_end].strip()
            
            parsed_content = parser.parse(content)
            contents.append(json.dumps(parsed_content.dict()))
            
            new_history = msg_history + [
                {"role": "user", "content": msg},
                {"role": "assistant", "content": content}
            ]
            histories.append(new_history)
        except Exception as e:
            logger.error(f"Error getting response: {str(e)}")
            continue
        
    if print_debug:
        logger.debug("\n" + "*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(histories[0]):
            logger.debug(f'{j}, {msg["role"]}: {msg["content"]}')
        logger.debug(contents)
        logger.debug("*" * 21 + " LLM END " + "*" * 21 + "\n")
        
    return contents, histories

def get_response_from_llm(
        msg: str,
        client: ChatOpenAI,
        model: str,
        system_message: str,
        print_debug: bool = False,
        msg_history: Optional[List[Dict]] = None,
        temperature: float = 0.75,
) -> Tuple[str, List[Dict]]:
    """Get single response using LangChain."""
    contents, histories = get_batch_responses_from_llm(
        msg=msg,
        client=client,
        model=model,
        system_message=system_message,
        print_debug=print_debug,
        msg_history=msg_history,
        temperature=temperature,
        n_responses=1
    )
    return contents[0] if contents else "", histories[0] if histories else []
