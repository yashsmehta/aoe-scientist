from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import Dict, Optional
import json


IDEA_SYSTEM_PROMPT = """You are a world-class AI researcher tasked with generating novel and impactful research ideas."""

IDEA_FIRST_PROMPT = """Based on the following recent papers in the field, generate ONE creative and novel research idea.

Recent papers:
{papers_context}

Requirements for the research idea:
1. Must be novel and not directly addressed in the provided papers
2. Should be feasible with current technology
3. Should have significant potential impact
4. Should be specific and concrete, not vague
5. Should combine insights from multiple papers when possible

Respond in the following format:

THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```

In <THOUGHT>, first briefly discuss your intuitions and motivations for the idea. Detail your high-level plan, necessary design choices and ideal outcomes of the experiments. Justify how the idea is different from the existing ones.

In <JSON>, provide the new idea in JSON format with the following fields:
- "Name": A shortened descriptor of the idea. Lowercase, no spaces, underscores allowed.
- "Title": A title for the idea, will be used for the report writing.
- "Experiment": An outline of the implementation. E.g. which functions need to be added or modified, how results will be obtained, ...

This JSON will be automatically parsed, so ensure the format is precise.
You will have {num_reflections} rounds to iterate on the idea, but do not need to use them all."""

IDEA_REFLECTION_PROMPT = """Round {current_round}/{num_reflections}.
In your thoughts, first carefully consider the quality, novelty, and feasibility of the idea you just created.
Include any other factors that you think are important in evaluating the idea.
Ensure the idea is clear and concise, and the JSON is the correct format.
Do not make things overly complicated.
In the next attempt, try and refine and improve your idea.
Stick to the spirit of the original idea unless there are glaring issues.

Respond in the same format as before:
THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```

If there is nothing to improve, simply repeat the previous JSON EXACTLY after the thought and include "I am done" at the end of the thoughts but before the JSON.
ONLY INCLUDE "I am done" IF YOU ARE MAKING NO MORE CHANGES."""

def generate_research_idea(chat, topic, rag=False):
    """Generate a novel research idea based on existing papers."""
    
    # Initial idea generation
    msg_history = []
    print(f"Generated idea: {json_str}")
    current_idea = ResearchIdea(json.loads(json_str))
    
    return current_idea