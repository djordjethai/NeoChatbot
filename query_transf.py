import logging
from langchain_openai import OpenAIEmbeddings
import os
from pinecone import Pinecone
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder
from langchain_community.utilities import GoogleSerperAPIWrapper
import cohere
from langchain_community.document_transformers import LongContextReorder
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
import json
from openai import OpenAI
from pinecone_text.sparse import BM25Encoder
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate
import pandas as pd
from io import StringIO
from sql_prompt import PromptDatabase

with PromptDatabase() as db:
    result1 = db.query_sql_record("HYDE_RAG")
    result2 = db.query_sql_record("QT_MAIN_PROMPT")
    result3 = db.query_sql_record("CONTEXT_RETRIEVER")

AZ_BLOB_API_KEY = os.getenv("AZ_BLOB_API_KEY")


def load_prompts_from_azure(bsc, inner_dict, key):
    blob_client = bsc.get_container_client("positive-user").get_blob_client("positive_prompts.json")
    prompts = json.loads(blob_client.download_blob().readall().decode('utf-8'))
    
    return prompts["POSITIVE"][inner_dict][key]


def load_data_from_azure(bsc, filename):
    """ Load data from Azure Blob Storage. """
    try:
        blob_service_client = bsc
        container_client = blob_service_client.get_container_client("positive-user")
        blob_client = container_client.get_blob_client(filename)

        streamdownloader = blob_client.download_blob()
        df = pd.read_csv(StringIO(streamdownloader.readall().decode("utf-8")))
        return df.dropna(how="all")
    
    except FileNotFoundError:
        return pd.DataFrame(columns=['Username', 'Thread ID', 'Thread Name', 'Conversation'])
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame(columns=['Username', 'Thread ID', 'Thread Name', 'Conversation'])


def upload_data_to_azure(bsc, filename, new_data):
    """ Upload data to Azure Blob Storage with appending new data. """

    # Convert DataFrame to CSV
    csv_data = new_data.to_csv(index=False)
    
    # Upload combined CSV data to Azure Blob Storage
    blob_service_client = bsc
    blob_client = blob_service_client.get_blob_client("positive-user", filename)
    blob_client.upload_blob(csv_data, overwrite=True)



class MultiQueryDocumentRetriever:
    """
    Retrieves relevant documents for complex queries by utilizing multi-query techniques
    and Pinecone's hybrid search capabilities.
    
    Attributes:
        document_url (str): URL of the document to be queried.
        question (str): Main query for document retrieval.
        namespace (str): Namespace within Pinecone where documents are stored.
        model (str): Model used for generating embeddings for the query and documents.
        temperature (int): Temperature setting for the language model, affecting creativity.
        chunk_size (int): Size of text chunks for generating sub-queries.
        chunk_overlap (int): Overlap size between consecutive text chunks.
        host (str): Host URL for the Pinecone service.
    """

    def __init__(self, document_url, question, namespace="zapisnici", model="text-embedding-3-large", host="https://neo-positive-a9w1e6k.svc.apw5-4e34-81fa.pinecone.io", temperature=0, chunk_size=500, chunk_overlap=0):
        self.document_url = document_url
        self.question = question
        self.namespace = namespace
        self.model = model
        self.host = host
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.logger = self._init_logger()
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=self.temperature)
        self.retriever = self._init_retriever()

    def _init_logger(self):
        """Initializes and configures the logger."""
        log_handler = logging.StreamHandler()
        log_handler.setLevel(logging.INFO)
        logger = logging.getLogger("langchain.retrievers.multi_query")
        logger.setLevel(logging.INFO)
        logger.addHandler(log_handler)
        return logger

    def _init_retriever(self):
        """Initializes the document retriever with Pinecone Hybrid Search."""
        api_key = os.environ.get("PINECONE_API_KEY_S")
        pinecone = Pinecone(api_key=api_key, host=self.host)
        index = pinecone.Index(host=self.host)
        
        embeddings = OpenAIEmbeddings(model=self.model)
        bm25_encoder = BM25Encoder()
        bm25_encoder.fit(self.question)

        pinecone_retriever = PineconeHybridSearchRetriever(
            embeddings=embeddings,
            sparse_encoder=bm25_encoder,
            index=index,
            namespace=self.namespace
        )
        return MultiQueryRetriever.from_llm(retriever=pinecone_retriever, llm=self.llm)

    def get_relevant_documents(self, custom_question=None):
        """Retrieves documents relevant to the provided or default query."""
        if custom_question is None:
            custom_question = self.question
        
        result = self.retriever.get_relevant_documents(query=custom_question)
        
        return "\n\n".join([doc.page_content for doc in result])
        

class PineconeRetriever:
    """Handles document retrieval using Pinecone's hybrid search capabilities.
    
    Attributes:
        query (str): The search query for retrieving documents.
        namespace (str): The namespace within Pinecone where the documents are stored.
        model (str): The model used for generating embeddings for the query and documents.
    """
    def __init__(self, query, namespace="zapisnici", model="text-embedding-3-large", host="https://neo-positive-a9w1e6k.svc.apw5-4e34-81fa.pinecone.io"):
            self.query = query
            self.namespace = namespace
            self.model = model
            self.host = host  # Ensure the host attribute is correctly set here
            self.pinecone = self._init_pinecone(host)  # Pass the host to the Pinecone initialization
            self.index = self.pinecone.Index(host=self.host)  # Use the host attribute here
            self.embeddings = OpenAIEmbeddings(model=self.model)
            self.bm25_encoder = BM25Encoder()
            self.bm25_encoder.fit(self.query)
            self.retriever = PineconeHybridSearchRetriever(
                embeddings=self.embeddings,
                sparse_encoder=self.bm25_encoder,
                index=self.index,
                namespace=self.namespace
            )

    def _init_pinecone(self, host):
        """Initializes the Pinecone service with the API key and host.
        
        Args:
            host (str): The host URL for the Pinecone service.
        """
        api_key = os.environ.get("PINECONE_API_KEY_S")
        return Pinecone(api_key=api_key, host=host)


    def get_relevant_documents(self):
        """Retrieves documents relevant to the query from Pinecone."""
        return self.retriever.get_relevant_documents(self.query)


class CohereReranker:
    """Reranks documents based on relevance to the query using Cohere's rerank model.

    Attributes:
        query (str): The search query for reranking documents.
    """
    def __init__(self, query):
        self.query = query
        self.client = cohere.Client(os.getenv("COHERE_API_KEY"))

    def rerank(self, documents):
        """Reranks the given documents based on their relevance to the query.

        Args:
            documents (list): A list of document texts to be reranked.

        Returns:
            str: A string containing the top reranked documents concatenated together.
        """
        results = self.client.rerank(query=self.query, documents=documents, top_n=3, model="rerank-multilingual-v2.0")
        return "\n\n".join([result.document['text'] for result in results])


class LongContextHandler:
    """Reorders documents to prioritize more relevant content for long contexts.

    This class does not require initialization parameters.
    """
    def reorder(self, documents):
        """Reorders the given documents based on their relevance and context.

        Args:
            documents (list): A list of document texts to be reordered.

        Returns:
            str: A string containing the reordered documents concatenated together.
        """
        reordering = LongContextReorder()
        reordered_docs = reordering.transform_documents(documents)
        return "\n\n".join(reordered_docs)


class ContextRetriever:
    """Retrieves and compresses documents to only include the most relevant parts.
    
    Attributes:
        documents (list): A list of documents to be compressed.
        model (str): The model used for compression.
    """
    
    def __init__(self, documents, model="gpt-4-turbo-preview"):
        self.documents = documents
        self.model = model
        
        response_schemas = [
            ResponseSchema(name="compressed_text", description="The compressed text of the document"),
        ]
        
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()
        
        self.prompt = PromptTemplate(
            template=result3.get('prompt_text', 'You are helpful assistant that always writes in Sebian').format(),
            input_variables=["documents"],
            partial_variables={"format_instructions": format_instructions},
        )
        self.llm = ChatOpenAI(model=self.model, temperature=0)
        # ovo bar je kako je bilo u primeru
        self.chain = self.prompt | self.llm | output_parser

    def get_compressed_context(self):
        """Compresses the provided documents to include only the most relevant parts."""
        # combine documents into a single string
        compressed_output = self.chain.invoke({"documents": "\n\n".join(self.documents)})

        # just in case...
        if isinstance(compressed_output, dict) and 'compressed_text' in compressed_output:
            return compressed_output['compressed_text']
        else:
            return "Error: Unexpected structure of compressed_output"


class StringLogHandler(logging.Handler):
    """A custom logging handler to collect log records in a list.

    This class is primarily used for debugging and development purposes.
    """
    def __init__(self):
        super().__init__()
        self.log_records = []

    def emit(self, record):
        """Collects log records into a list.

        Args:
            record (logging.LogRecord): The log record to be collected.
        """
        log_entry = self.format(record)
        self.log_records.append(log_entry)


def web_search_process(query: str) -> str:
    """
    Executes a web search using the provided query string.

    This function wraps the Google Search API, specifically using a service wrapper
    designed for this purpose (`GoogleSerperAPIWrapper`). It requires an API key, which
    is expected to be provided via environment variables. The function then runs the query
    through the API and returns the search results as a string.

    Parameters:
    - query (str): The search query string to be submitted to the Google Search API.

    Returns:
    - str: The search results returned by the Google Search API as a string.
    """    
    return GoogleSerperAPIWrapper(environment=os.environ["SERPER_API_KEY"]).run(query)

def hyde_rag(prompt):
  
    client = OpenAI()
    response = client.chat.completions.create(
        model= "gpt-4-turbo-preview",
        temperature=0.5,
        messages=[
            {"role": "system", "content": result1.get('prompt_text', 'You are helpful assistant that always writes in Sebian.')},
            {"role": "user", "content": prompt}
            ]
        )
    response = response.choices[0].message.content
    
    return response

def create_structured_prompt(user_query):
    """
    Constructs a structured prompt for use with an AI model, based on a user's query.

    This function generates a structured JSON prompt that outlines various tools the AI
    assistant can use to answer questions. The prompt includes a description of each tool and
    specifies the scenarios in which each should be used, such as queries related to specific
    company data, SQL queries, or general questions that require web search. The user's query
    is incorporated into the prompt to provide context for the AI's response generation.

    Parameters:
    - user_query: The original query from the user that needs to be addressed by the AI model.

    Returns:
    - A list of dictionaries, each representing a part of the structured prompt, including
      the role (system or user) and the content (instructions for the AI or the user query).
    """
    return [
        {"role": "system", "content": result2.get('prompt_text', 'You are helpful assistant that always writes in Sebian.')},
        {"role": "user", "content": user_query}
    ]


def get_structured_decision_from_model(user_query):

    """
    Determines the most appropriate tool to use for a given user query using an AI model.

    This function sends a user query to an AI model and receives a structured decision in the
    form of a JSON object. The decision includes the recommended tool to use for addressing
    the user's query, based on the content and context of the query. The function uses a
    structured prompt, generated by `create_structured_prompt`, to instruct the AI on how
    to process the query. The AI's response is parsed to extract the tool recommendation.

    Parameters:
    - user_query: The user's query for which the tool recommendation is sought.

    Returns:
    - The name of the recommended tool as a string, based on the AI's analysis of the user query.
    """
    
    client = OpenAI()
    response = client.chat.completions.create(
        model= "gpt-4-turbo-preview",
        temperature=0,
        response_format= { "type": "json_object" },
        messages=create_structured_prompt(user_query),
    )
    json_string = response.choices[0].message.content
    # Parse the JSON string into a Python dictionary
    data_dict = json.loads(json_string)
    # Access the 'tool' value
    tool_value = data_dict['tool']
    
    return tool_value  
        