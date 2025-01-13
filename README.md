# AOE Scientist

AI research idea generator and reviewer based on the AI Scientist paper.

## Installation

1. Install UV (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create and activate a new virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

3. Install dependencies using UV:
```bash
uv pip install .
```

For development dependencies:
```bash
uv pip install ".[dev]"
```

## Usage

1. Set your API keys:
```bash
export OPENAI_API_KEY='your-openai-key'
export ANTHROPIC_API_KEY='your-anthropic-key'
```

2. Run the idea generator and reviewer:
```bash
python main.py
```

The system will:
1. Generate a novel research idea using Claude or GPT-4
2. Review the idea using multiple reviewers and reflection rounds
3. Output both the idea and its review in a structured format

## Project Structure

- `main.py`: Main script to run the idea generation and review pipeline
- `idea_generator.py`: Research idea generation with reflections
- `idea_reviewer.py`: Multi-reviewer idea evaluation system
- `llm.py`: LLM client handling and utilities
