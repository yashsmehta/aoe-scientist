from dotenv import load_dotenv
from aoe_scientist.llm import create_client

def test_deepseek_connection():
    """Test basic connection to Deepseek API"""
    load_dotenv()
    llm = create_client(provider="deepseek")
    
    # Simple test prompt
    response = llm.invoke("which model are you using?")
    assert response.content is not None
    print(f"\nRaw response content: {response.content}")


def test_openai_connection():
    """Test basic connection to OpenAI API"""
    load_dotenv()
    llm = create_client(provider="openai")
    
    # Simple test prompt
    response = llm.invoke("which model are you using?")
    assert response.content is not None
    print(f"\nOpenAI response content: {response.content}")

def test_anthropic_connection():
    """Test basic connection to Anthropic API"""
    load_dotenv()
    llm = create_client(provider="anthropic")
    
    # Simple test prompt
    response = llm.invoke("which model are you using?")
    assert response.content is not None
    print(f"\nAnthropic response content: {response.content}")
