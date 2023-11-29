import openai
import streamlit as st

from os import getenv
from time import sleep
from requests import get as requests_get
from bs4 import BeautifulSoup
from pdfkit import configuration, from_string
from csv import reader, writer

version = "v0.9"
getenv("OPENAI_API_KEY")
client = openai
assistant_id = "asst_N4RTKXxTqE28KRttY34el3fE"


# za tools
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    )
from langchain.utilities import GoogleSerperAPIWrapper

from os import environ
from openai import OpenAI

import pinecone
from pinecone_text.sparse import BM25Encoder
from myfunc.mojafunkcija import open_file


def web_serach_process(q: str) -> str:
    return GoogleSerperAPIWrapper(environment=environ["SERPER_API_KEY"]).run(q)


def hybrid_search_process(upit: str) -> str:
    alpha = 0.9
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
    
    x = str(ChatPromptTemplate(messages=[system_message, human_message]))
    with open("chat_prompt.txt", "w", encoding="utf-8-sig") as f:
        f.write(x)
    return x


client = OpenAI()
our_assistant = client.beta.assistants.retrieve(assistant_id=assistant_id)

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
    "cancel_run": None,
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
if st.session_state.thread_id:
    thread = client.beta.threads.retrieve(thread_id=st.session_state.thread_id)

instructions = """
Please remember to always check if a tool is relevant to your query. \
Answer only in the Serbian language. For answers consult the uploaded files.
"""

try:
    run = client.beta.threads.runs.cancel(thread_id=st.session_state.thread_id, run_id=st.session_state.cancel_run)
except:
    pass

run = None

if prompt := st.chat_input(placeholder="Postavite pitanje"):
    message = client.beta.threads.messages.create(thread_id=st.session_state.thread_id, role="user", content=prompt) 

    run = client.beta.threads.runs.create(thread_id=st.session_state.thread_id, assistant_id=assistant.id, 
                                          instructions=instructions)
    
if run is not None:
    while True: 
        sleep(1)
        run_status = client.beta.threads.runs.retrieve(thread_id=st.session_state.thread_id, run_id=run.id)
        if run_status.status in ["completed", "requires_action"]:
            break
        else:
            sleep(1)

def write_chat():
    messages = client.beta.threads.messages.list(thread_id=st.session_state.thread_id) 
    for msg in reversed(messages.data): 
        role = msg.role
        content = msg.content[0].text.value 
        if role == 'user':
            st.markdown(f"<div style='background-color:lightblue; padding:10px; margin:5px; border-radius:5px;'><span style='color:blue'>👤 {role.capitalize()}:</span> {content}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background-color:lightgray; padding:10px; margin:5px; border-radius:5px;'><span style='color:red'>🤖 {role.capitalize()}:</span> {content}</div>", unsafe_allow_html=True)


if run is not None and run_status.status == "requires_action":
    st.session_state.cancel_run = run.id
    st.write("Function Calling")
    required_actions = run_status.required_action.submit_tool_outputs.model_dump()
    print(required_actions)
    tool_outputs = []
    import json
    for action in required_actions["tool_calls"]:
        func_name = action['function']['name']
        arguments = json.loads(action['function']['arguments'])
        
        if func_name == "web_search_process":
            try:
                output = web_serach_process(arguments['query'])
            except:
                output = web_serach_process(arguments['q'])

            tool_outputs.append({
                "tool_call_id": action['id'],
                "output": output
            })
        elif func_name == "hybrid_search_process":
            output = hybrid_search_process(arguments['upit'])
            tool_outputs.append({
                "tool_call_id": action['id'],
                "output": output
            })
        else:
            raise ValueError(f"Unknown function: {func_name}")
        
    client.beta.threads.runs.submit_tool_outputs(
        thread_id=thread.id,
        run_id=run.id,
        tool_outputs=tool_outputs
    )
    sleep(4)

try:
    write_chat()
except:
    pass