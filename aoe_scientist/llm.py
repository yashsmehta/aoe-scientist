from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import os
import logging

logger = logging.getLogger(__name__)


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
            # "model": "o1-preview"
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
        "api_key": api_key
    }
    
    if config["model"] != "o1-preview":
        kwargs["temperature"] = temperature
    
    if "base_url" in config:
        kwargs["base_url"] = config["base_url"]
        
    chat = config["class"](**kwargs)
    print(f"Using LangChain with {provider} model: {config['model']}")
    return chat
