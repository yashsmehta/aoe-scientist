# AoE Scientist: Agents-of-Experts for Novel Scientific Idea Generation 🧪

An AI-powered research idea generator and reviewer that emulates the thought processes of distinguished researchers. Inspired by the [AI Scientist work](https://sakana.ai/ai-scientist/), this system goes further by creating AI agents that embody the unique perspectives and methodologies of leading minds in their respective fields. The system employs multiple approaches to idea generation: direct researcher emulation, evolutionary optimization of agent populations for multi-objective idea quality, and dynamic collaboration networks that form research teams based on complementary strengths.

## Installation 🛠️

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

## Usage 🚀

1. Set your API keys:
```bash
export OPENAI_API_KEY='your-openai-key'
export ANTHROPIC_API_KEY='your-anthropic-key'
export DEEPSEEK_API_KEY='your-deepseek-key'  # Optional
```

2. Generate ideas:
```bash
python aoe_scientist/main.py mode=generate topic="neural architecture search" researcher="david ha" rag=true
```

3. Review generated ideas:
```bash
python aoe_scientist/main.py mode=review
```

## Project Structure 📁

```
aoe_scientist/
├── main.py           # Main execution pipeline
├── idea_generator.py # Research idea generation with RAG
├── idea_reviewer.py  # Multi-criteria idea evaluation
├── llm.py           # LLM client handling (OpenAI, Anthropic, DeepSeek)
└── utils.py         # Helper functions and configuration

data/
├── ideas.csv        # Generated research ideas
└── reviews.csv      # Idea evaluations and scores
```

## Roadmap 🗺️

### Multi-Agent Evolution
Implementing evolutionary algorithms to optimize a population of research agents:
- Pareto-based selection across multiple objectives (technical merit, novelty)
- Diversity maintenance through crowding distance
- Clustering-based agent characteristics inheritance
- Population evolution until diversity stabilization

### Adaptive Collaboration Networks
Dynamic research team formation based on past success:
- Weighted collaboration network between researcher agents
- Team formation based on relationship strength
- Adaptive learning of successful partnerships
- Network evolution towards optimal research collaborations

## Contributing 🤝

Contributions are welcome! Feel free to open issues or submit pull requests to help improve the system.

## License 📄

MIT
