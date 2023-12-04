import openai
import os

os.environ.get("OPENAI_API_KEY")
client = openai

system_prompt = """System: Hello, I am a function calling AI assistant. Use your knowledge base, uploaded files and provided tools to best respond to user queries. 
Always answer in Serbian. You can use English to search for information on the web."""

tool_descriptions = {
   "hybrid_search_process" : """
    This function performs a hybrid search process using Pinecone and BM25Encoder. 
    It initializes Pinecone with the provided API key and environment, reads from an index named 'positive', and performs 
    a hybrid query using the provided query and alpha value. The function then formats the results and returns them in a specific format.
    Call this function each time you are asked about the zapisnik, zapisnici, or sastanak AI Tima and you will get the best possible answer.
    """
}

tools_list = [
    
    {
    "type": "function",
    "function": {
        "name": "hybrid_search_process",
        "description": tool_descriptions["hybrid_search_process"],
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
    instructions=system_prompt,
    model="gpt-4-1106-preview",
    name="Zapisnik assistant",
    tools=tools_list)

print(our_assistant.id)
