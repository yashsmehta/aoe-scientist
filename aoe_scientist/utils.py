"""Utility functions for AI Research Idea Generator and Reviewer."""
import os
from omegaconf import OmegaConf
import json
from dotenv import load_dotenv


def setup_config(file_path="config/default.yaml"):
    """Load and merge configuration from multiple sources.
    
    Priority (highest to lowest):
    1. Command line arguments
    2. Environment variables
    3. Config file
    4. Default config
    """
    load_dotenv()
    default_conf = OmegaConf.load(file_path)
    custom_conf = OmegaConf.from_cli()
    config = OmegaConf.merge(default_conf, custom_conf)

    if config.mode not in ['generate', 'review']:
        raise ValueError(f"Invalid mode: {config.mode}. Must be 'generate' or 'review'")
    
    if config.mode == 'review' and not config.idea_path:
        raise ValueError("idea_path must be specified in review mode")
        
    return OmegaConf.to_container(config)

def save_json(data, filepath):
    """Append data to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            existing_data = json.load(f)
        if isinstance(existing_data, list):
            existing_data.append(data)
        elif isinstance(existing_data, dict):
            existing_data.update(data)
        else:
            raise ValueError("Unsupported JSON format for appending data")
    else:
        existing_data = data
    
    with open(filepath, 'w') as f:
        json.dump(existing_data, f, indent=2)
    
    print(f"Results appended to: {filepath}")
