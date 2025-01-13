"""Utility functions for AI Research Idea Generator and Reviewer."""
import os
from typing import Dict, Any
from omegaconf import OmegaConf
import json

def setup_config() -> Dict[str, Any]:
    """Load and merge configuration from multiple sources.
    
    Priority (highest to lowest):
    1. Command line arguments
    2. Environment variables
    3. Config file
    4. Default config
    """
    # Load default config
    default_conf = OmegaConf.load("config/default.yaml")
    
    # Load custom config if exists
    custom_conf = OmegaConf.from_cli()
    
    # Merge configs with CLI taking precedence
    config = OmegaConf.merge(default_conf, custom_conf)
    
    # Validate config
    if config.mode not in ['generate', 'review']:
        raise ValueError(f"Invalid mode: {config.mode}. Must be 'generate' or 'review'")
    
    if config.mode == 'review' and not config.idea_path:
        raise ValueError("idea_path must be specified in review mode")
        
    return OmegaConf.to_container(config)

def save_json(data, filepath):
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to: {filepath}")
