"""Utility functions for AI Research Idea Generator and Reviewer."""
import os
from omegaconf import OmegaConf
import json
import pandas as pd
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
    
    # Load default config
    default_conf = OmegaConf.load(file_path)
    
    # Create CLI config with structured format
    cli_conf = OmegaConf.from_dotlist([
        f"{k}={v}" for k, v in OmegaConf.from_cli().items()
    ])
    
    # Merge configs with CLI taking precedence
    config = OmegaConf.merge(default_conf, cli_conf)
    if config.rag == False:
        config.researcher = None

    if 'mode' not in config or config.mode not in ['generate', 'review']:
        raise ValueError(f"Invalid mode: {getattr(config, 'mode', None)}. Must be 'generate' or 'review'")
    
    if config.mode == 'review' and not hasattr(config, 'idea_path'):
        config.idea_path = "data/ideas.json"  # Set default path
        
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

def save_df(data_df, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if os.path.exists(filepath):
        existing_df = pd.read_csv(filepath)
        data_df = pd.concat([existing_df, data_df], ignore_index=True)
    
    data_df.to_csv(filepath, mode='w', index=False)
    print(f"DataFrame updated and saved to: {filepath}")