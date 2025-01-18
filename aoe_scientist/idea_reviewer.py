from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from pydantic import BaseModel, Field
import pandas as pd
import json
from typing import Dict, Any

class ReviewOutput(BaseModel):
    """Schema for the review output."""
    technical_merit: int = Field(description="Rating from 1-10 on technical soundness", ge=1, le=10)
    novelty: int = Field(description="Rating from 1-10 on originality", ge=1, le=10)
    feasibility: int = Field(description="Rating from 1-10 on implementation feasibility", ge=1, le=10)
    impact: int = Field(description="Rating from 1-10 on potential impact", ge=1, le=10)
    clarity: int = Field(description="Rating from 1-10 on idea clarity", ge=1, le=10)
    justification: str = Field(description="Technical justification for the scores in 3-4 concise sentences")

# Prompt templates
REVIEW_SYSTEM_TEMPLATE = """You are a senior AI research reviewer tasked with critically evaluating research ideas in the field of {topic}. 
Your evaluation must be extremely thorough, unbiased, and highly critical - your research standards are extremely high. Many ideas might sound good on the surface, but ultimately 
not make any sense in {topic} - be very critical of such ideas and don't hesitate to give low scores. Use your knowledge of the field to critically evaluate ideas.

For each metric, provide a score from 1-10 and justify your reasoning. Be extremely critical - most ideas should score low unless they are truly exceptional:

- Technical Merit (1-10): Evaluate technical soundness, methodology, and theoretical foundation. Most ideas lack proper theoretical grounding or have methodological flaws.
- Novelty (1-10): Assess originality and innovation compared to existing work. Most ideas are minor variations of existing work.
- Feasibility (1-10): Judge practical implementability with current technology. Many ideas ignore real-world constraints and challenges.
- Impact (1-10): Evaluate potential influence on the field and broader applications. Most ideas have limited impact beyond incremental improvements.
- Clarity (1-10): Rate how well-defined and clearly articulated the idea is. Many ideas are vague or poorly specified.
- Justification: Provide a technical justification for your scores in 3-4 concise sentences. Focus on major technical limitations and methodological flaws."""

REVIEW_HUMAN_TEMPLATE = """Please review the following research idea with extreme rigor and skepticism:

Title: {title}
Description: {details}

### Review Guidelines:
1. Critically analyze each aspect objectively and think deeply about the idea
2. Support scores with specific examples and reasoning
3. Consider both immediate and long-term implications
4. Identify major challenges, limitations and flaws
5. Be extremely critical - most ideas should score low unless truly exceptional"""

REFLECTION_SYSTEM_TEMPLATE = """You are an expert senior research idea evaluator. Your task is to analyze and refine the initial review scores.
Look for additional technical limitations that weren't identified. Don't hesitate to lower scores or increase scores of the initial review based on technical merit.

Here is the current state and context of the field to help evaluate against:
{field_context}"""

REFLECTION_HUMAN_TEMPLATE = """Current Research Idea:
Title: {title}
Description: {details}

Initial Review Scores:
Technical Merit: {technical_merit}/10
- Technical Merit (1-10): Evaluate technical soundness, methodology, and theoretical foundation. Most ideas lack proper theoretical grounding or have methodological flaws.

Novelty: {novelty}/10
- Novelty (1-10): Assess originality and innovation compared to existing work. Most ideas are minor variations of existing work.
Feasibility: {feasibility}/10
- Feasibility (1-10): Judge practical implementability with current technology. Many ideas ignore real-world constraints and challenges.
Impact: {impact}/10
- Impact (1-10): Evaluate potential influence on the field and broader applications. Most ideas have limited impact beyond incremental improvements.
Clarity: {clarity}/10
- Clarity (1-10): Rate how well-defined and clearly articulated the idea is. Many ideas are vague or poorly specified.
Overall Score: {overall_score}/10


As an expert senior evaluator in this research field, use your in-depth knowledge of the domain's current state to critically re-evaluate this idea. Leverage your expertise to provide a rigorous analysis and adjust the scores based on technical merit, novelty, feasibility, impact, and clarity. Ensure your evaluation reflects the highest standards of scientific scrutiny.

1. **Technical Analysis**:
   - Are the theoretical and methodological foundations robust and well-justified?
   - Have any critical technical flaws, gaps, or overlooked elements been identified?
   - Does the idea align with the latest advancements or trends in the field?

2. **Impact and Novelty Assessment**:
   - Is the idea genuinely novel, or does it build incrementally on existing work?
   - Are there competing ideas or solutions that diminish its novelty or impact?
   - Is the proposed impact realistic, or is it overstated?

3. **Feasibility and Clarity**:
   - Are there unaddressed practical challenges or constraints?
   - Does the description clearly and comprehensively outline the implementation pathway?
   - Are there ambiguities or missing details that affect feasibility or clarity?

Your task:
1. Identify additional strengths or limitations based on the current state of the field.
2. Adjust scores (increase or decrease) for each category based on your technical analysis.
3. Provide a concise, 3-4 sentence justification for any score adjustments, explaining your reasoning in technical terms."""

def create_review_chain(chat, topic: str):
    """Create a review chain with proper response schema parsing."""
    # Load context for the topic
    topic = "nas"
    context_path = f"data/surveys/{topic}/context.txt"
    try:
        with open(context_path, 'r') as f:
            field_context = f.read()
    except FileNotFoundError:
        print(f"Warning: No context file found at {context_path}")
        field_context = ""

    # Create structured chat model with function calling
    structured_chat = chat.with_structured_output(ReviewOutput, method="function_calling")

    # Create prompt templates
    review_prompt = ChatPromptTemplate.from_messages([
        ("system", REVIEW_SYSTEM_TEMPLATE),
        ("human", REVIEW_HUMAN_TEMPLATE)
    ]).partial(topic=topic)

    reflection_prompt = ChatPromptTemplate.from_messages([
        ("system", REFLECTION_SYSTEM_TEMPLATE),
        ("human", REFLECTION_HUMAN_TEMPLATE)
    ])

    def get_initial_review(title: str, details: str) -> Dict[str, Any]:
        """Get initial review scores and criticism."""
        messages = review_prompt.format_messages(title=title, details=details)
        review = structured_chat.invoke(messages)
        print("\nInitial review:")
        print(json.dumps(review.dict(), indent=2))
        return review.dict()

    def get_reflection_review(title: str, details: str, initial_review: Dict[str, Any]) -> Dict[str, Any]:
        """Get reflection review scores and criticism."""
        # Calculate overall score using integers
        score_fields = ["technical_merit", "novelty", "feasibility", "impact", "clarity"]
        overall_score = sum(int(initial_review[k]) for k in score_fields) / len(score_fields)
        
        messages = reflection_prompt.format_messages(
            field_context=field_context,
            title=title,
            details=details,
            **initial_review,
            overall_score=overall_score
        )
        review = structured_chat.invoke(messages)
        print("\nFinal review after reflection:")
        print(json.dumps(review.dict(), indent=2))
        return review.dict()

    def review_with_reflection(title: str, details: str) -> Dict[str, Any]:
        """Generate initial review and refine through reflection."""
        try:
            # Get initial review
            initial_review = get_initial_review(title, details)

            # Get reflection review
            try:
                final_review = get_reflection_review(title, details, initial_review)
            except Exception as e:
                print(f"Reflection failed: {str(e)}")
                final_review = initial_review

            # Calculate overall score
            score_fields = ["technical_merit", "novelty", "feasibility", "impact", "clarity"]
            overall_score = sum(int(final_review[k]) for k in score_fields) / len(score_fields)

            # Combine results with overall score
            return {
                **{f"initial_{k}": v for k, v in initial_review.items()},
                **final_review,
                "overall_score": overall_score
            }
        except Exception as e:
            print(f"Review failed: {str(e)}")
            raise e

    return review_with_reflection

def review_ideas(chat, cfg):
    """Review research ideas and save results to a dataframe."""
    ideas = pd.read_csv("data/ideas.csv", index_col=False)
    review_chain = create_review_chain(chat, cfg['topic'])
    reviews_df = pd.DataFrame()

    for idx, idea in ideas.iterrows():
        print(f"\nReviewing idea {idx+1}/{len(ideas)}:")
        for attempt in range(3):
            try:
                review = review_chain(idea['title'], idea['details'])
                review_data = {
                    'name': str(idea['name']),
                    'title': str(idea['title']),
                    'researcher': idea['researcher'],
                    'rag': idea['rag'],
                    'generate_llm': idea['generate_llm'],
                    'review_llm': cfg.get('review_llm'),
                    **review
                }
                reviews_df = pd.concat([reviews_df, pd.DataFrame([review_data])], ignore_index=True)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"Failed to review '{idea['name']}' after 3 attempts: {str(e)}")
                    review_data = {
                        'name': str(idea['name']),
                        'title': str(idea['title']),
                        'researcher': idea['researcher'],
                        'rag': idea['rag'],
                        'review_llm': cfg.get('review_llm'),
                        'justification': f"Failed to review: {str(e)}",
                        'overall_score': 0
                    }
                    reviews_df = pd.concat([reviews_df, pd.DataFrame([review_data])], ignore_index=True)
                else:
                    print(f"Attempt {attempt+1} failed, retrying...")

    return reviews_df