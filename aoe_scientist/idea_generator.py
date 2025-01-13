from typing import Dict, Optional
import json
import logging

from llm import get_response_from_llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResearchIdea:
    def __init__(self, data: Dict):
        self.name = data["Name"]
        self.title = data["Title"]
        self.experiment = data["Experiment"]
    
    def to_dict(self):
        return {
            "Name": self.name,
            "Title": self.title,
            "Experiment": self.experiment,
        }

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

def generate_research_idea(papers_df, client, model, num_reflections=1) -> Optional[ResearchIdea]:
    """Generate a novel research idea based on existing papers."""
    try:
        # Create a sample of recent papers for context
        papers_sample = papers_df.sample(n=5)
        papers_context = "\n".join([f"Title: {row['title']}\nAbstract: {row['abstract']}" 
                                   for _, row in papers_sample.iterrows()])
        
        # Initial idea generation
        msg_history = []
        logger.info(f"Iteration 1/{num_reflections}")
        json_str, msg_history = get_response_from_llm(
            IDEA_FIRST_PROMPT.format(
                papers_context=papers_context,
                num_reflections=num_reflections
            ),
            client=client,
            model=model,
            system_message=IDEA_SYSTEM_PROMPT,
            msg_history=msg_history
        )
        
        logger.debug(f"Generated idea: {json_str}")
        current_idea = ResearchIdea(json.loads(json_str))
        
        # Iteratively improve idea
        if num_reflections > 1:
            for j in range(num_reflections - 1):
                logger.info(f"Iteration {j + 2}/{num_reflections}")
                json_str, msg_history = get_response_from_llm(
                    IDEA_REFLECTION_PROMPT.format(
                        current_round=j + 2,
                        num_reflections=num_reflections
                    ),
                    client=client,
                    model=model,
                    system_message=IDEA_SYSTEM_PROMPT,
                    msg_history=msg_history
                )
                
                logger.debug(f"Refined idea: {json_str}")
                
                if "I am done" in json_str:
                    logger.info(f"Idea generation converged after {j + 2} iterations.")
                    break
                
                current_idea = ResearchIdea(json.loads(json_str))
        
        return current_idea
    
    except Exception as e:
        logger.error(f"Error generating research idea: {str(e)}")
        return None 