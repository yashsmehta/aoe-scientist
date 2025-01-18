from dotenv import load_dotenv
from aoe_scientist.llm import create_client
import glob
import os

PROMPT = """
You are an expert senior researcher in the field of {field}. You are tasked with producing a comprehensive technical analysis of the following review papers.

You will be provided with text from review papers on a specific topic, surrounded by triple quotes. Your task is to understand the field,
what areas are promising, and what areas are lacking. What is the state of the art? What are the key techniques and methodologies?
These are all review papers, so you should use the provided papers to understand the field. Be very specific and detailed. Your output
should be a very technical and detailed summary of the field.
The output should include:
A detailed summary of the key technical points, contributions, and findings.
An in-depth analysis of the methodologies and techniques discussed.
A synthesis of the main arguments or perspectives presented, highlighting any consensus or debates within the papers.
Insights on the implications of these findings for the field and potential future directions for research.
Please focus on technical accuracy and provide a comprehensive analysis.
Papers:
{papers}

"""

def read_survey_files(directory):
    papers_content = []
    
    for txt_file in glob.glob(os.path.join(directory, "*.txt")):
        with open(txt_file, 'r') as f:
            papers_content.append(f.read())
    
    return "\n\n".join(papers_content)

def call_llm():
    """Process survey papers and save distilled output"""
    load_dotenv()
    llm = create_client(llm_provider="anthropic")
    field = "neuroai"

    directory = f"data/surveys/{field}"
    
    # Read and combine all survey papers
    papers_content = read_survey_files(directory)
    
    # Get distilled content from LLM
    response = llm.invoke(PROMPT.format(papers=papers_content, field=field))
    assert response.content is not None
    
    # Save the distilled output
    output_path = os.path.join(directory, "context.txt")
    with open(output_path, 'w') as f:
        f.write(response.content)
    
    print(f"Distilled content saved to {output_path}")


if __name__ == "__main__":
    call_llm()