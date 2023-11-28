from openai import OpenAI
from os import getenv

getenv("OPENAI_API_KEY")
client = OpenAI()

our_assistant = client.beta.assistants.create(
    instructions="You are a function calling AI assistant. Use your knowledge base, uploaded files and provided tools to best respond to user queries. \
        Always answer in Serbian. You can use English to search for information on the web.",
    model="gpt-4-1106-preview",
    name="Positive assistant",
    tools=[
        {
        "type": "function",
        "function": {
            "name": "web_search_process",
            "description": """This tool uses Google Search to find the most relevant and up-to-date information on the web. \
                This tool is particularly useful when you need comprehensive information on a specific topic, \
                    want to explore different viewpoints, or are looking for the latest news and data.
                    Please note that the quality and relevance of results may depend on the specificity of your query. \
                        Never use this tool when asked about Positive doo.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                    },
                "required": ["query"]
                },
            }
        }, {
        "type": "function",
        "function" : {
            "name": "hybrid_search_process",
            "description": "This function performs a hybrid search process using Pinecone and BM25Encoder. \
                It initializes Pinecone with the provided API key and environment, creates an index named 'positive', \
                    and performs a hybrid query using the provided query and alpha value. \
                        The function then formats the results and returns them in a specific format.",
            "parameters": {
                "type": "object",
                "properties": {
                    "upit": {
                        "type": "string",
                        "description": "The query to be searched."},
                    "alpha": {
                        "type": "number",
                        "description": "The alpha value used in the hybrid score normalization."}
                    },
                "required": ["upit", "alpha"]
                }
            }
        }
        ]
    )

print(our_assistant.id)