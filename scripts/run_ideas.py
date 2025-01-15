#!/usr/bin/env python3
import subprocess
from typing import Optional

def run_with_config(llm_name: str, rag: bool, researcher: Optional[str] = None):
    """Run main.py with specified configuration"""
    cmd = ["python", "-m", "aoe_scientist.main", 
           "mode=generate",
           f"generate_llm={llm_name}",
           f"rag={str(rag).lower()}"]
    
    if researcher:
        cmd.append(f"researcher={researcher}")
    
    subprocess.run(cmd)

def main():
    # Configure parameters here
    llms = ["deepseek", "openai", "anthropic"]
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