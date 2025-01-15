#!/usr/bin/env python3
import subprocess
from typing import Optional

def run_with_config(llm_name: str):
    """Run main.py with specified configuration"""
    cmd = ["python", "-m", "aoe_scientist.main", 
           "mode=review",
           f"review_llm={llm_name}"]
    
    subprocess.run(cmd)

def main():
    llms = ["deepseek", "openai", "anthropic"]
    
    for llm in llms:
        run_with_config(llm)
    
if __name__ == "__main__":
    main() 