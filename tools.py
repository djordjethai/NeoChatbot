from langchain.agents import (
    Tool,
    AgentType,
    AgentExecutor,
    LLMSingleActionAgent,
    load_tools,
    AgentOutputParser,
    create_sql_agent,
)
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    StringPromptTemplate,
)
from langchain.schema import (
    AgentAction,
    AgentFinish,
    OutputParserException,
)
from langchain.sql_database import SQLDatabase
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.llms.openai import OpenAI
from langchain.tools import tool

from os import environ
from re import search, DOTALL
from typing import List, Union
from openai import OpenAI

client = OpenAI()
import pinecone
from pinecone_text.sparse import BM25Encoder
from myfunc.mojafunkcija import open_file

environ.get("OPENAI_API_KEY")


def hybrid_search_process_alpha1(upit):
    """
    The Keyword Search tool is used to find exact matches for the terms in your query. \
    It scans through the data and retrieves all instances where the keywords appear. \
    This makes it particularly useful when you are looking for specific information and know the exact terms to search for.
    However, it may not capture all relevant information if synonyms or related terms are used instead of the exact keywords. \
    Please note that the quality and relevance of results may depend on the specificity of your query. This tool is relevant if the query is about Positive doo.
    """
    return hybrid_search_process(upit, 0.1)


def hybrid_search_process_alpha2(upit):
    """
    The Semantic Search tool is used to understand the intent and contextual meaning of a query. \
    By analyzing the semantics of the query, it can retrieve information that is not just keyword-based but also contextually relevant. \
    This makes it particularly useful when dealing with complex queries or when searching for information in large, unstructured data sets. 
    Please note that the quality and relevance of results may depend on the specificity of your query. 
    This tool is relevant if the query is about Positive doo.
    """
    return hybrid_search_process(upit, 0.9)


def hybrid_search_process(upit, alpha):
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
            top_k=session_state["broj_k"],
            vector=hdense,
            sparse_vector=hsparse,
            include_metadata=True,
            namespace=session_state["namespace"],
            ).to_dict()

    session_state["tematika"] = hybrid_query()

    uk_teme = ""
    for _, item in enumerate(session_state["tematika"]["matches"]):
        if item["score"] > 0.05:    # session_state["score"]
            uk_teme += item["metadata"]["context"] + "\n\n"

    system_message = SystemMessagePromptTemplate.from_template(
        template=session_state["stil"]
        ).format()

    human_message = HumanMessagePromptTemplate.from_template(
        template=open_file("prompt_FT.txt")
        ).format(
            zahtev=question,
            uk_teme=uk_teme,
            ft_model="gpt-4-1106-preview",
            )

    return ChatPromptTemplate(messages=[system_message, human_message])
