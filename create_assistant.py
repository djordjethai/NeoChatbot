import openai
import os

os.environ.get("OPENAI_API_KEY")
client = openai

tools_list = [
    {
    "type": "function",
    "function": {
        "name": "get_stock_price",
        "description": "Retrieve the latest closing price of a stock using its ticker symbol",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "The ticker symbol of the stock"}
            },
            "required": ["symbol"]}
    }}, 
    {
    "type": "function",
    "function": {
        "name": "web_search_process",
        "description": "This tool uses Google Search to find the most relevant and up-to-date information on the web.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to be searched."}
            },
            "required": ["query"]}
    }},
    {
    "type": "function",
    "function": {
        "name": "hybrid_search_process",
        "description": "This function performs a hybrid search process using Pinecone and BM25Encoder. It initializes Pinecone with the provided API key and environment, creates an index named 'positive', and performs a hybrid query using the provided query and alpha value. The function then formats the results and returns them in a specific format.",
        "parameters": {
            "type": "object",
            "properties": {
                "upit": {
                    "type": "string",
                    "description": "The query to be searched."},
            },
            "required": ["upit"]}
    }}
    ]


our_assistant = client.beta.assistants.create(
    instructions="You are a function calling AI assistant. Use your knowledge base, uploaded files and provided tools to best respond to user queries. \
        Always answer in Serbian. You can use English to search for information on the web.",
    model="gpt-4-1106-preview",
    name="Positive assistant",
    tools=tools_list)

print(our_assistant.id)
