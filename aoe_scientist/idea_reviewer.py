import json
from typing import Dict, List
from aoe_scientist.idea_generator import ResearchIdea
from llm import get_response_from_llm, get_batch_responses_from_llm
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

REVIEWER_SYSTEM_PROMPT = """You are a senior AI researcher tasked with reviewing research ideas.
Be critical and constructive in your assessment. If an idea has issues or you are unsure, point them out clearly."""

REVIEW_TEMPLATE = """Review the following research idea critically and constructively:

Research Idea:
Name: {name}
Title: {title}
Experiment: {experiment}
Interestingness Score: {interestingness}/10
Feasibility Score: {feasibility}/10
Novelty Score: {novelty}/10

Evaluate this idea on:
1. Technical Merit: Is the approach sound and well-thought-out?
2. Novelty: How original and innovative is the idea?
3. Feasibility: Can it be implemented with current technology?
4. Impact: What is the potential scientific and practical impact?
5. Clarity: How well-defined and specific is the idea?

Respond in the following format:

THOUGHT:
<THOUGHT>

REVIEW JSON:
```json
<JSON>
```

In <THOUGHT>, first briefly discuss your intuitions and reasoning for the evaluation.
Detail your high-level arguments and assessment of the idea's potential.
Be specific and constructive in your criticism.

In <JSON>, provide the review in JSON format with the following fields:
- "Technical_Merit": A rating from 1 to 10 (lowest to highest)
- "Novelty": A rating from 1 to 10 (lowest to highest)
- "Feasibility": A rating from 1 to 10 (lowest to highest)
- "Impact": A rating from 1 to 10 (lowest to highest)
- "Clarity": A rating from 1 to 10 (lowest to highest)
- "Overall_Score": A rating from 1 to 10 (lowest to highest)
- "Strengths": List of main strengths of the idea
- "Weaknesses": List of main weaknesses or concerns
- "Suggestions": List of specific suggestions for improvement
- "Decision": Either "Accept" or "Reject"

This JSON will be automatically parsed, so ensure the format is precise."""

REVIEW_REFLECTION_PROMPT = """Round {current_round}/{num_reflections}.
In your thoughts, first carefully consider the accuracy and fairness of the review you just created.
Include any other factors that you think are important in evaluating the idea.
Ensure the review is clear and constructive, and the JSON is in the correct format.
Do not make things overly complicated.
In the next attempt, try to refine and improve your review.
Stick to your core assessment unless there are clear issues.

Respond in the same format as before:
THOUGHT:
<THOUGHT>

REVIEW JSON:
```json
<JSON>
```

If there is nothing to improve, simply repeat the previous JSON EXACTLY after the thought and include "I am done" at the end of the thoughts but before the JSON.
ONLY INCLUDE "I am done" IF YOU ARE MAKING NO MORE CHANGES."""

class ReviewOutput(BaseModel):
    """Schema for review output."""
    Technical_Merit: int = Field(description="Technical merit rating from 1-10")
    Novelty: int = Field(description="Novelty rating from 1-10")
    Feasibility: int = Field(description="Feasibility rating from 1-10")
    Impact: int = Field(description="Impact rating from 1-10")
    Clarity: int = Field(description="Clarity rating from 1-10")
    Overall_Score: int = Field(description="Overall score from 1-10")
    Strengths: List[str] = Field(description="List of main strengths")
    Weaknesses: List[str] = Field(description="List of main weaknesses")
    Suggestions: List[str] = Field(description="List of suggestions for improvement")
    Decision: str = Field(description="Either 'Accept' or 'Reject'")

class IdeaReview:
    def __init__(self, data: Dict):
        self.technical_merit = data["Technical_Merit"]
        self.novelty = data["Novelty"]
        self.feasibility = data["Feasibility"]
        self.impact = data["Impact"]
        self.clarity = data["Clarity"]
        self.overall_score = data["Overall_Score"]
        self.strengths = data["Strengths"]
        self.weaknesses = data["Weaknesses"]
        self.suggestions = data["Suggestions"]
        self.decision = data["Decision"]
    
    def to_dict(self):
        return {
            "Technical_Merit": self.technical_merit,
            "Novelty": self.novelty,
            "Feasibility": self.feasibility,
            "Impact": self.impact,
            "Clarity": self.clarity,
            "Overall_Score": self.overall_score,
            "Strengths": self.strengths,
            "Weaknesses": self.weaknesses,
            "Suggestions": self.suggestions,
            "Decision": self.decision
        }

def review_idea(idea, client, model, num_reflections=5, num_reviews_ensemble=1, temperature=0.3) -> IdeaReview:
    """Review a research idea using multiple reviewers and reflection rounds."""
    
    base_prompt = REVIEW_TEMPLATE.format(
        name=idea.name,
        title=idea.title,
        experiment=idea.experiment,
        interestingness=idea.interestingness,
        feasibility=idea.feasibility,
        novelty=idea.novelty
    )
    
    parser = JsonOutputParser(pydantic_object=ReviewOutput)
    format_instructions = parser.get_format_instructions()
    
    if num_reviews_ensemble > 1:
        # Get multiple reviews
        llm_reviews, msg_histories = get_batch_responses_from_llm(
            base_prompt,
            client=client,
            model=model,
            system_message=REVIEWER_SYSTEM_PROMPT,
            temperature=temperature,
            n_responses=num_reviews_ensemble
        )
        
        # Parse all reviews
        parsed_reviews = []
        for rev in llm_reviews:
            parsed_content = parser.parse(rev)
            parsed_reviews.append(parsed_content.model_dump())
        
        # Aggregate reviews
        review = aggregate_reviews(parsed_reviews)
        msg_history = msg_histories[0]  # Use first message history
    else:
        # Single review
        llm_review, msg_history = get_response_from_llm(
            base_prompt,
            client=client,
            model=model,
            system_message=REVIEWER_SYSTEM_PROMPT,
            temperature=temperature
        )
        parsed_content = parser.parse(llm_review)
        review = parsed_content.model_dump()
    
    # Reflection rounds
    if num_reflections > 1:
        for j in range(num_reflections - 1):
            print(f"Reflection {j + 2}/{num_reflections}")
            text, msg_history = get_response_from_llm(
                REVIEW_REFLECTION_PROMPT.format(
                    current_round=j + 2,
                    num_reflections=num_reflections
                ),
                client=client,
                model=model,
                system_message=REVIEWER_SYSTEM_PROMPT,
                msg_history=msg_history,
                temperature=temperature
            )
            
            if "I am done" in text:
                print(f"Review converged after {j + 2} iterations.")
                break
                
            parsed_content = parser.parse(text)
            review = parsed_content.model_dump()
    
    return IdeaReview(review)

def aggregate_reviews(reviews: List[Dict]) -> Dict:
    """Aggregate multiple reviews into a single review."""
    # Average numerical scores
    score_fields = ["Technical_Merit", "Novelty", "Feasibility", "Impact", "Clarity", "Overall_Score"]
    aggregated = {}
    for field in score_fields:
        scores = [r[field] for r in reviews if field in r]
        aggregated[field] = round(sum(scores) / len(scores))
    
    # Combine text fields
    text_fields = ["Strengths", "Weaknesses", "Suggestions"]
    for field in text_fields:
        items = []
        for r in reviews:
            if field in r:
                items.extend(r[field])
        aggregated[field] = list(set(items))  # Remove duplicates
    
    # Decision based on majority vote
    decisions = [r["Decision"] for r in reviews if "Decision" in r]
    aggregated["Decision"] = max(set(decisions), key=decisions.count)
    
    return aggregated

if __name__ == "__main__":
    # Initialize the model
    model = ChatOpenAI(model_name="gpt-4", temperature=0.3)  # Lower temperature for more consistent reviews
    
    # Example idea for testing
    test_idea = ResearchIdea(
        title="Example Research Idea",
        hypothesis="Test hypothesis",
        approach="Test approach",
        potential_impact="Test impact",
        novelty_score=8
    )
    
    # Review the idea
    review = review_idea(test_idea, model)
    print(json.dumps(review.dict(), indent=2)) 