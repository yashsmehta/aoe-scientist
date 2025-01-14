from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any
import json


class Review(BaseModel):
    """Schema for a research idea review with integer scores between 1-10."""
    technical_merit: int = Field(..., description="Rating from 1-10 on technical soundness", ge=1, le=10)
    novelty: int = Field(..., description="Rating from 1-10 on originality", ge=1, le=10)
    feasibility: int = Field(..., description="Rating from 1-10 on implementation feasibility", ge=1, le=10)
    impact: int = Field(..., description="Rating from 1-10 on potential impact", ge=1, le=10)
    clarity: int = Field(..., description="Rating from 1-10 on idea clarity", ge=1, le=10)
    reasoning: str = Field(..., description="Detailed reasoning for the scores")

    @field_validator('technical_merit', 'novelty', 'feasibility', 'impact', 'clarity', mode='before')
    @classmethod
    def validate_score(cls, v: int) -> int:
        if not isinstance(v, int):
            raise ValueError("Score must be an integer")
        if v < 1 or v > 10:
            raise ValueError("Score must be between 1 and 10")
        return v

    @property
    def overall_score(self) -> float:
        """Calculate overall score as average of all metrics."""
        scores = [self.technical_merit, self.novelty, self.feasibility, 
                 self.impact, self.clarity]
        return sum(scores) / len(scores)


def create_review_chain(chat, prompt_data: Dict[str, Any]):
    """Create a review chain with proper JSON parsing."""
    parser = JsonOutputParser(pydantic_object=Review)
    
    review_prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_data["messages"][0]["content"]),
        ("human", prompt_data["messages"][1]["content"])
    ]).partial(format_instructions=parser.get_format_instructions())
    
    return review_prompt | chat | parser


def review_idea(chat, cfg):
    """Review a research idea."""
    # Load idea from JSON
    with open("data/ideas.json", "r") as f:
        ideas = json.load(f)
        idea = ideas[cfg.idea_id]
    
    # Load prompt template
    with open("prompts/idea_review.json", "r") as f:
        prompt_data = json.load(f)
    
    # Create and execute review chain
    review_chain = create_review_chain(chat, prompt_data)
    
    try:
        return review_chain.invoke({
            "name": idea["name"],
            "title": idea["title"],
            "experiment": idea["details"],
            "interestingness": idea.get("interestingness", 5),
            "feasibility": idea.get("feasibility", 5),
            "novelty": idea.get("novelty", 5)
        })
    except Exception as e:
        raise ValueError(f"Failed review: {str(e)}")