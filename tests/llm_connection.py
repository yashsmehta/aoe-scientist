from dotenv import load_dotenv
from aoe_scientist.llm import create_client

def test_deepseek_connection():
    """Test basic connection to Deepseek API"""
    load_dotenv()
    client, model = create_client("deepseek-chat")
    
    # Simple test prompt
    response = client.invoke("Say 'hello world' in JSON format")
    assert response.content is not None
    print(f"\nRaw response content: {response.content}")

def test_openai_connection():
    """Test basic connection to OpenAI API"""
    load_dotenv()
    client, model = create_client("gpt-4o")
    
    # Simple test prompt
    response = client.invoke("Translate 'hello world' to French")
    assert response.content is not None
    print(f"\nOpenAI response content: {response.content}")

def test_anthropic_connection():
    """Test basic connection to Anthropic API"""
    load_dotenv()
    client, model = create_client("claude-3-5-sonnet-v2@20241022")
    
    # Simple test prompt
    response = client.invoke("Summarize the following text: 'LangChain is a framework for developing applications powered by language models.'")
    assert response.content is not None
    print(f"\nAnthropic response content: {response.content}")

def test_json_response():
    """Test JSON response handling"""
    load_dotenv()
    client, model = create_client("deepseek-chat")
    
    # Test with explicit JSON request
    response = client.invoke('''Please respond with only a JSON object in this exact format:
    {
        "Name": "test_idea",
        "Title": "Test Title",
        "Experiment": "Test experiment description"
    }
    No other text, just the JSON.''')
    
    print(f"\nJSON response content: {response.content}")
    assert response.content is not None
    
    # Check if response contains JSON markers
    assert "{" in response.content
    assert "}" in response.content 
