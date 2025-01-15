from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from pydantic import BaseModel, Field, field_validator
import json
import pandas as pd
from aoe_scientist.utils import save_df

# Prompt templates for idea review
REVIEW_SYSTEM_TEMPLATE = """You are a senior AI research reviewer tasked with critically evaluating research ideas in the field of {topic}. 
Your evaluation must be thorough, unbiased, and critical - your research standards are extremely high. Note, many of the ideas might sound good, but ultimately, 
not make any sense in {topic} - be very critical of such ideas (dont shy away from giving low scores). Be wise, and critically 
evaluate the ideas on ones that actually make sense. Use your knowledge of the field to critically evaluate the ideas.

For each metric, provide a score from 1-10 and justify your reasoning:
- Technical Merit (1-10): Evaluate the technical soundness, methodology, and theoretical foundation
- Novelty (1-10): Assess originality and innovation compared to existing work
- Feasibility (1-10): Judge practical implementability with current technology
- Impact (1-10): Evaluate potential influence on the field and broader applications
- Clarity (1-10): Rate how well-defined and clearly articulated the idea is
- Criticism: What are the critisisms of the idea? and provide a well throught out explanation for your scores

{format_instructions}"""

REVIEW_HUMAN_TEMPLATE = """Please review the following research idea:

Title: {title}
Description: {details}

### Review Guidelines:
1. Analyze each aspect independently and objectively
2. Support scores with specific examples and reasoning
3. Consider both immediate and long-term implications
4. Identify potential challenges and limitations
5. Suggest concrete improvements where applicable

### Required Format:
Provide scores (1-10) and detailed reasoning for each metric in the specified JSON format."""


class Review(BaseModel):
    """Schema for a research idea review with integer scores between 1-10."""
    criticism: str = Field(..., description="Criticism of the idea, detailed reasoning for the scores (single paragraph)")
    technical_merit: int = Field(..., description="Rating from 1-10 on technical soundness", ge=1, le=10)
    novelty: int = Field(..., description="Rating from 1-10 on originality", ge=1, le=10)
    feasibility: int = Field(..., description="Rating from 1-10 on implementation feasibility", ge=1, le=10)
    impact: int = Field(..., description="Rating from 1-10 on potential impact", ge=1, le=10)
    clarity: int = Field(..., description="Rating from 1-10 on idea clarity", ge=1, le=10)

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
    
    def model_dump(self, **kwargs) -> dict:
        """Convert model to dictionary with support for Pydantic v2 arguments."""
        base_dict = super().model_dump(**kwargs)
        # Add overall_score if not explicitly excluded
        if not kwargs.get('exclude') or 'overall_score' not in kwargs['exclude']:
            base_dict['overall_score'] = self.overall_score
        return base_dict


def create_review_chain(chat, topic: str):
    """Create a review chain with proper response schema parsing."""
    response_schemas = [
        ResponseSchema(name="technical_merit", description="Rating from 1-10 on technical soundness (integer)"),
        ResponseSchema(name="novelty", description="Rating from 1-10 on originality (integer)"),
        ResponseSchema(name="feasibility", description="Rating from 1-10 on implementation feasibility (integer)"),
        ResponseSchema(name="impact", description="Rating from 1-10 on potential impact (integer)"),
        ResponseSchema(name="clarity", description="Rating from 1-10 on idea clarity (integer)"),
        ResponseSchema(name="criticism", description="Detailed reasoning for the scores")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    
    review_prompt = ChatPromptTemplate.from_messages([
        ("system", REVIEW_SYSTEM_TEMPLATE),
        ("human", REVIEW_HUMAN_TEMPLATE)
    ])
    
    # Properly bind format instructions
    review_prompt = review_prompt.partial(
        topic=topic,
        format_instructions=format_instructions
    )
    
    def convert_scores_to_int(response_dict):
        """Convert score values to integers."""
        score_fields = ["technical_merit", "novelty", "feasibility", "impact", "clarity"]
        for field in score_fields:
            if field in response_dict:
                try:
                    response_dict[field] = int(float(response_dict[field]))
                except (ValueError, TypeError):
                    raise ValueError(f"Could not convert {field} to integer")
        return response_dict
    
    return review_prompt | chat | output_parser | convert_scores_to_int


def review_idea(chat, cfg):
    """Review a research idea and save results to a dataframe."""
    with open(f"data/ideas/{cfg['researcher']}.json", "r") as f:
        ideas = json.load(f)
        idea = ideas[0] if isinstance(ideas, list) else ideas
    
    review_chain = create_review_chain(chat, cfg['topic'])
    
    max_retries = 3
    last_error = None
    
    for attempt in range(max_retries):
        try:
            response = review_chain.invoke({
                "title": idea["Title"],
                "details": idea["Details"],
            })
            review = Review(**response)
            
            # Create review data for dataframe
            review_data = {
                'name': idea["Name"],
                'title': idea["Title"],
                'researcher': cfg['researcher'],
                'review_llm': cfg.get('review_llm', 'deepseek'),
                'technical_merit': review.technical_merit,
                'novelty': review.novelty,
                'feasibility': review.feasibility,
                'impact': review.impact,
                'clarity': review.clarity,
                'overall_score': review.overall_score,
                'criticism': review.criticism
            }
            
            df = save_df(review_data, 'data/reviews.csv')
            
            return review, df
            
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                continue
    
    raise ValueError(f"Failed review after {max_retries} attempts. Last error: {str(last_error)}")