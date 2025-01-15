#!/usr/bin/env python3
import subprocess
import yaml
from pathlib import Path
from typing import Optional

def run_with_config(llm_name: str, rag: bool, researcher: Optional[str] = None):
    """Run main.py with specified configuration"""
    config_path = Path("config/default.yaml")
    
    # Load existing config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Update config
    config["generate_llm"] = llm_name
    config["rag"] = rag
    if researcher:
        config["researcher"] = researcher
    
    # Save temporary config
    config_name = f"{llm_name}_{'rag' if rag else 'norag'}"
    if researcher:
        config_name += f"_{researcher}"
    temp_config = config_path.parent / f"temp_{config_name}.yaml"
    
    with open(temp_config, "w") as f:
        yaml.dump(config, f)
    
    # Run main with this config
    print(f"\nRunning with LLM: {llm_name}, RAG: {rag}, Researcher: {researcher or 'N/A'}...")
    subprocess.run(["python", "-m", "aoe_scientist.main", "--config", str(temp_config)])
    
    # Cleanup
    temp_config.unlink()

def main():
    # Configure parameters here
    llms = ["deepseek", "gpt4o", "anthropic"]
    researchers = ["Mehta", "Ha", "Lillicrap", "Hutter", "Funke", "Bonner"]
    
    # Run without RAG
    for llm in llms:
        run_with_config(llm, rag=False)
    
    # Run with RAG for each researcher
    for llm in llms:
        for researcher in researchers:
            run_with_config(llm, rag=True, researcher=researcher)

if __name__ == "__main__":
    main() 