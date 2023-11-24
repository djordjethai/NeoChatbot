import openai
import streamlit as st

from os import getenv
from time import sleep
from requests import get as requests_get
from bs4 import BeautifulSoup
from pdfkit import configuration, from_string
from csv import reader, writer

version = "v0.8"
getenv("OPENAI_API_KEY")
client = openai
assistant_id = "asst_cGNrHE0NDUn8AHcOkBg2sXaq"



agent_prompt = """
Before you start, please remember to always check if a tool is relevant to your query. \
You can do this by reading the descriptions of the tools:
    1. 'The Keyword Search tool is used to find exact matches for the terms in your query. \
    It scans through the data and retrieves all instances where the keywords appear. \
    This makes it particularly useful when you are looking for specific information and know the exact terms to search for.
    However, it may not capture all relevant information if synonyms or related terms are used instead of the exact keywords. \
    Please note that the quality and relevance of results may depend on the specificity of your query. This tool is relevant if the query is about Positive doo.'
    2. 'The Semantic Search tool is used to understand the intent and contextual meaning of a query. \
    By analyzing the semantics of the query, it can retrieve information that is not just keyword-based but also contextually relevant. \
    This makes it particularly useful when dealing with complex queries or when searching for information in large, unstructured data sets. 
    Please note that the quality and relevance of results may depend on the specificity of your query. 
    This tool is relevant if the query is about Positive doo.'
    3. 'The Web Search tool is used to find relevant information on the web.'
Aways go through the above descriptions before you continue with regular chat. In case you think that the tool is relevant to your query, return it's number as the final answer. \
So if you think that the Keyword Search tool is relevant to your query, return '1' as the final answer. Only '1' or '2' or '3' are valid answers. No additional text is allowed.
"""
# za tools
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    )
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.llms.openai import OpenAI

from os import environ
from openai import OpenAI

client = OpenAI()
import pinecone
from pinecone_text.sparse import BM25Encoder
from myfunc.mojafunkcija import open_file

# tools
web_search = GoogleSerperAPIWrapper(environment=environ["SERPER_API_KEY"])

oparam = {"namespace": "positive", "stil": "system", "tematika": {}}

def hybrid_search_process_alpha1(upit, oparam):
    """
    The Keyword Search tool is used to find exact matches for the terms in your query. \
    It scans through the data and retrieves all instances where the keywords appear. \
    This makes it particularly useful when you are looking for specific information and know the exact terms to search for.
    However, it may not capture all relevant information if synonyms or related terms are used instead of the exact keywords. \
    Please note that the quality and relevance of results may depend on the specificity of your query. This tool is relevant if the query is about Positive doo.
    """
    return hybrid_search_process(upit, 0.1, oparam)

def hybrid_search_process_alpha2(upit, oparam):
    """
    The Semantic Search tool is used to understand the intent and contextual meaning of a query. \
    By analyzing the semantics of the query, it can retrieve information that is not just keyword-based but also contextually relevant. \
    This makes it particularly useful when dealing with complex queries or when searching for information in large, unstructured data sets. 
    Please note that the quality and relevance of results may depend on the specificity of your query. 
    This tool is relevant if the query is about Positive doo.
    """
    return hybrid_search_process(upit, 0.9, oparam)

def hybrid_search_process(upit, alpha, oparam):
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
            namespace=oparam["namespace"],
            ).to_dict()

    oparam["tematika"] = hybrid_query()

    uk_teme = ""
    for _, item in enumerate(oparam["tematika"]["matches"]):
        if item["score"] > 0.05:    # session_state["score"]
            uk_teme += item["metadata"]["context"] + "\n\n"

    system_message = SystemMessagePromptTemplate.from_template(
        template=oparam["stil"]
        ).format()

    human_message = HumanMessagePromptTemplate.from_template(
        template=open_file("prompt_FT.txt")
        ).format(
            zahtev=upit,
            uk_teme=uk_teme,
            ft_model="gpt-4-1106-preview",
            )

    return ChatPromptTemplate(messages=[system_message, human_message])


with open(file="threads.csv", mode="r") as f:
    reader = reader(f)
    next(reader)
    saved_threads = dict(reader)

default_session_states = {
    "file_id_list": [],
    "openai_model": "gpt-4-1106-preview",
    "messages": [],
    "thread_id": None,
    "threads": saved_threads,
    }
for key, value in default_session_states.items():
    if key not in st.session_state:
        st.session_state[key] = value

def scrape_website(url):
    return BeautifulSoup(markup=requests_get(url).text, features="html.parser",).get_text()

def text_to_pdf(text, filename):
    from_string(input=text,
                output_path=filename,
                configuration=configuration(wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"))
    return filename

def upload_to_openai(filepath):
    with open(filepath, "rb") as f:
        response = openai.files.create(file=f.read(), purpose="assistants")
    return response.id

st.set_page_config(page_title="MultiTool app", page_icon="🤖")
st.write(saved_threads)

st.sidebar.header(body="MultiTool chatbot; " + version)

st.sidebar.text("")
website_url = st.sidebar.text_input(label="Unesite URL web-stranice za scrape-ovanje", key="website_url")
if st.sidebar.button(label="Scrape and Upload"):
    try:
        st.session_state.file_id_list.append(
            upload_to_openai(filepath=text_to_pdf(text=scrape_website(url=website_url), filename="scraped_content.pdf")))
    except Exception as e:
        st.warning("Opis greške:\n\n" + str(e))

uploaded_file = st.sidebar.file_uploader(label="Upload fajla u OpenAI embeding", key="uploadedfile")
if st.sidebar.button(label="Upload File"):
    try:
        with open(file=f"{uploaded_file.name}", mode="wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.file_id_list.append(
            upload_to_openai(filepath=f"{uploaded_file.name}"))
    except Exception as e:
        st.warning("Opis greške:\n\n" + str(e))

if st.session_state.file_id_list:
    st.sidebar.write("ID-jevi svih upload-ovanih fajlova:")
    for file_id in st.session_state.file_id_list:
        st.sidebar.write(file_id)
        # povezivanje fajla sa asistentom
        client.beta.assistants.files.create(assistant_id=assistant_id, file_id=file_id)

st.session_state.threads = saved_threads

st.sidebar.text("")
new_chat_name = st.sidebar.text_input(label="Unesite ime za novi chat", key="newchatname")
if new_chat_name.strip() != "" and st.sidebar.button(label="Create Chat"):
    thread = client.beta.threads.create()
    st.session_state.thread_id = thread.id
    with open(file="threads.csv", mode="a", newline="") as f:
        writer(f).writerow([new_chat_name, thread.id])
    st.rerun()
    
chosen_chat = st.sidebar.selectbox(label="Izaberite chat", options=["Select..."] + list(saved_threads.keys()))
if chosen_chat.strip() not in ["", "Select..."] and st.sidebar.button(label="Select Chat"):
    thread = client.beta.threads.retrieve(thread_id=st.session_state.threads[chosen_chat])
    st.session_state.thread_id = thread.id
    st.rerun()

st.sidebar.text("")
assistant = client.beta.assistants.retrieve(assistant_id=assistant_id)
thread = client.beta.threads.retrieve(thread_id=st.session_state.thread_id)

if prompt := st.chat_input(placeholder="Postavite pitanje"):
    message = client.beta.threads.messages.create(thread_id=st.session_state.thread_id, role="user", content=prompt) 
    run = client.beta.threads.runs.create(thread_id=st.session_state.thread_id, assistant_id=assistant.id, 
                                          instructions="Answer only in the Serbian language. For answers consult the uploaded files.") 
    while True: 
        sleep(0.1)
        run_status = client.beta.threads.runs.retrieve(thread_id=st.session_state.thread_id, run_id=run.id)
        if run_status.status == "completed": 
            messages = client.beta.threads.messages.list(thread_id=st.session_state.thread_id) 
            for msg in messages.data: 
                role = msg.role 
                content = msg.content[0].text.value 
                st.write(f"{role.capitalize()}: {content}")
            break

_ = """
# Add user message to the state and display it
st.session_state.messages.append({"role": "user", "content": prompt})
with st.chat_message(name="user"):
    st.markdown(body=prompt)

# Add the user's message to the existing thread
client.beta.threads.messages.create(
    thread_id=st.session_state.thread_id,
    role="user",
    content=prompt,)

# Create a run with additional instructions
run = client.beta.threads.runs.create(
    thread_id=st.session_state.thread_id,
    assistant_id=assistant_id,
    instructions="Please answer the queries using the knowledge provided in the files. \
        When adding other information mark it clearly as such with a different color",)

# Poll for the run to complete and retrieve the assistant"s messages
while run.status != "completed":
    sleep(1)
    run = client.beta.threads.runs.retrieve(
        thread_id=st.session_state.thread_id,
        run_id=run.id,)

# Retrieve messages added by the assistant
messages = client.beta.threads.messages.list(
    thread_id=st.session_state.thread_id,)

# Process and display assistant messages
assistant_messages_for_run = [
    message for message in messages if message.run_id == run.id and message.role == "assistant"]
"""
