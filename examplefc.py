import openai
import time
import yfinance as yf


def get_stock_price(symbol: str) -> float:
    stock = yf.Ticker(symbol)
    price = stock.history(period="1d")['Close'].iloc[-1]
    return price


from langchain.utilities import GoogleSerperAPIWrapper
from os import environ

def web_serach_process(query: str) -> str:
    return GoogleSerperAPIWrapper(environment=environ["SERPER_API_KEY"]).run(query=query)



from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    )
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.llms.openai import OpenAI as OAI

import pinecone
from pinecone_text.sparse import BM25Encoder
from myfunc.mojafunkcija import open_file


client = OAI()
def hybrid_search_process(upit: str) -> str:
    alpha = 0.5
    pinecone.init(
        api_key=environ["PINECONE_API_KEY_POS"],
        environment=environ["PINECONE_ENVIRONMENT_POS"],
    )
    index = pinecone.Index("positive")

    def hybrid_query():
        def get_embedding(text, model="text-embedding-ada-002"):
            text = text.replace("\n", " ")
            return client.embeddings.create(input = [text], model=model).data[0].embedding
        
        hybrid_score_norm = (
            lambda dense, sparse, alpha: (
                [v * alpha for v in dense],
                {
                    "indices": sparse["indices"],
                    "values": [v * (1 - alpha) for v in sparse["values"]],
                },
            )
            if 0 <= alpha <= 1
            else ValueError("Alpha must be between 0 and 1")
        )

        hdense, hsparse = hybrid_score_norm(
            sparse = BM25Encoder().fit([upit]).encode_queries(upit),
            dense=get_embedding(upit),
            alpha=alpha,
        )

        return index.query(
            top_k=3,
            vector=hdense,
            sparse_vector=hsparse,
            include_metadata=True,
            namespace="zapisnici",
            ).to_dict()

    tematika = hybrid_query()

    uk_teme = ""
    for _, item in enumerate(tematika["matches"]):
        if item["score"] > 0.05:    # score
            uk_teme += item["metadata"]["context"] + "\n\n"

    system_message = SystemMessagePromptTemplate.from_template(
        template="You are a helpful assistent. You always answer in the Serbian language."
        ).format()

    human_message = HumanMessagePromptTemplate.from_template(
        template=open_file("prompt_FT.txt")
        ).format(
            zahtev=upit,
            uk_teme=uk_teme,
            ft_model="gpt-4-1106-preview",
            )
    return str(ChatPromptTemplate(messages=[system_message, human_message]))

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

# Initialize the client
client = openai.OpenAI()

# Step 1: Create an Assistant
assistant = client.beta.assistants.create(
    name="Data Analyst Assistant",
    instructions="You are a personal Data Analyst Assistant",
    tools=tools_list,
    model="gpt-4-1106-preview",
)

# Step 2: Create a Thread
thread = client.beta.threads.create()

# Step 3: Add a Message to a Thread
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="Using hybrid search process, tell me the date of the latest meeting held."
)

# Step 4: Run the Assistant
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
    instructions="Please address the user as Mervin Praison."
)

print(run.model_dump_json(indent=4))

while True:
    # Wait for 5 seconds
    time.sleep(5)

    # Retrieve the run status
    run_status = client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id
    )
    print(run_status.model_dump_json(indent=4))

    # If run is completed, get messages
    if run_status.status == 'completed':
        messages = client.beta.threads.messages.list(
            thread_id=thread.id
        )

        # Loop through messages and print content based on role
        for msg in messages.data:
            role = msg.role
            content = msg.content[0].text.value
            print(f"{role.capitalize()}: {content}")

        break
    elif run_status.status == 'requires_action':
        print("Function Calling")
        required_actions = run_status.required_action.submit_tool_outputs.model_dump()
        print(required_actions)
        tool_outputs = []
        import json
        for action in required_actions["tool_calls"]:
            func_name = action['function']['name']
            arguments = json.loads(action['function']['arguments'])
            
            if func_name == "get_stock_price":
                output = get_stock_price(symbol=arguments['symbol'])
                tool_outputs.append({
                    "tool_call_id": action['id'],
                    "output": output
                })
            elif func_name == "web_search_process":
                output = web_serach_process(query=arguments['query'])
                tool_outputs.append({
                    "tool_call_id": action['id'],
                    "output": output
                })
            elif func_name == "hybrid_search_process":
                output = hybrid_search_process(upit=arguments['upit'])
                tool_outputs.append({
                    "tool_call_id": action['id'],
                    "output": output
                })
            else:
                raise ValueError(f"Unknown function: {func_name}")
            
        print("Submitting outputs back to the Assistant...")
        client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread.id,
            run_id=run.id,
            tool_outputs=tool_outputs
        )
    else:
        print("Waiting for the Assistant to process...")
        time.sleep(5)
