import openai
import streamlit as st
import os
from os import getenv
from time import sleep
from json import loads as json_loads
from json import dumps as json_dumps
# from pdfkit import configuration, from_string
from myfunc.mojafunkcija import (
    st_style,
    positive_login,
    open_file,)
import nltk     # kasnije ce se paketi importovati u funkcijama

st.set_page_config(page_title="Pravnik 02N-str ispravka", page_icon="🤖")

version = "v1.1"
getenv("OPENAI_API_KEY")
client = openai
assistant_id = "asst_1YAl3U9XJTOnfYUJrStFO1nH"  # printuje se u drugoj skripti, a moze jelte da se vidi i na OpenAI Playground-u
client.beta.assistants.retrieve(assistant_id=assistant_id)

# isprobati da li ovo radi kod Vas
# from custom_theme import custom_streamlit_style

# importi za funkcije
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    )
# ne treba nam vise web search funkcija
# from langchain.utilities import GoogleSerperAPIWrapper

from os import environ
from openai import OpenAI   # !?

import pinecone
from pinecone_text.sparse import BM25Encoder
from myfunc.mojafunkcija import open_file

ovaj_asistent = "pravnik"

from azure.storage.blob import BlobServiceClient
import pandas as pd
from io import StringIO

def load_data_from_azure(bsc):
    try:
        blob_service_client = bsc
        container_client = blob_service_client.get_container_client("positive-user")
        blob_client = container_client.get_blob_client("assistant_data.csv")

        streamdownloader = blob_client.download_blob()
        return pd.read_csv(StringIO(streamdownloader.readall().decode("utf-8")))
    
    except FileNotFoundError:
        return {"Nisam pronasao fajl"}
    except Exception as e:
        return {f"An error occurred: {e}"}



global username
def main():
    if "username" not in st.session_state:
        try:
            st.session_state.username = username
        except:
            st.session_state.username = "positive"
    client = OpenAI()
    if "data" not in st.session_state:
        st.session_state.data = None
    if "blob_service_client" not in st.session_state:
        st.session_state.blob_service_client = BlobServiceClient.from_connection_string(os.environ.get("AZ_BLOB_API_KEY"))

    st.session_state.data = load_data_from_azure(st.session_state.blob_service_client)
    threads_dict = {thread.chat: thread.ID for thread in st.session_state.data.itertuples() if st.session_state.username == thread.user and ovaj_asistent == thread.assistant}
    
    # Inicijalizacija session state-a
    default_session_states = {
        "file_id_list": [],
        "openai_model": "gpt-4-1106-preview",
        "messages": [],
        "thread_id": None,
        "cancel_run": None,
        "namespace": "pravnik",
        }
    for key, value in default_session_states.items():
        if key not in st.session_state:
            st.session_state[key] = value


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
        
            hybrid_score_norm = (lambda dense, sparse, alpha: 
                                 ([v * alpha for v in dense], 
                                  {"indices": sparse["indices"], 
                                   "values": [v * (1 - alpha) for v in sparse["values"]]}
                                   ))
            hdense, hsparse = hybrid_score_norm(
                sparse = BM25Encoder().fit([upit]).encode_queries(upit),
                dense=get_embedding(upit),
                alpha=alpha,
            )
            return index.query(
                top_k=6,
                vector=hdense,
                sparse_vector=hsparse,
                include_metadata=True,
                namespace=st.session_state.namespace,
                ).to_dict()

        tematika = hybrid_query()

        uk_teme = ""
        for _, item in enumerate(tematika["matches"]):
            if item["score"] > 0.05:    # score
                uk_teme += item["metadata"]["context"] + "\n\n"

        system_message = SystemMessagePromptTemplate.from_template(
            template="You are a helpful assistent. You always answer in the Serbian language.").format()

        human_message = HumanMessagePromptTemplate.from_template(
            template=open_file("prompt_FT.txt")).format(
                zahtev=upit,
                uk_teme=uk_teme,
                ft_model="gpt-4-1106-preview",
                )
        return str(ChatPromptTemplate(messages=[system_message, human_message]))


    

    # funkcije za scrape-ovanje i upload-ovanje dokumenata
    def upload_to_openai(filepath):
        with open(filepath, "rb") as f:
            response = openai.files.create(file=f.read(), purpose="assistants")
        return response.id



    # krecemo polako i sa definisanjem UI-a
   
    # st.markdown(custom_streamlit_style, unsafe_allow_html=True)   # ne radi izgleda vise
    st.sidebar.header(body="Pravnik asistent; " + version)
    with st.sidebar.expander(label="Kako koristiti?", expanded= False):
        st.write(""" 
1. Aplikacija vam omogucava da razgovarate o pitanjima vezanim za interna dokumenta, pravilnike i sl. Pomenite sistematizaciju ili pravilnik. 

2. Pamti razgovore koje ste imali do sada i mozete ih nastaviti po zelji. Odaberite iz padajuceg menija raniji razgovor i odaberite select

3. Mozete zapoceti i novi ragovor. unesite ime novog razgovora i pritisnite novi razgovor , a zatim ga iz padajuceg menija odaberite i potvrdite izbor.

4. Mozete uploadovati dokument i razgovarati o njegovom sadrzaju.

5. Ova aplikacija nema pristup internetu, ali poseduje znanja do Marta 2023. godine.

6. Za neke odgovore mozda ce trebati malo vremena, budite strpljivi.
        """)
    # Narednih 50-tak linija su za unosenje raznih vrednosti
    st.sidebar.text("")


    uploaded_file = st.sidebar.file_uploader(label="Upload fajla u OpenAI embeding", key="uploadedfile")
    if st.sidebar.button(label="Upload File", key="uploadfile"):
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

    st.sidebar.text("")
    new_chat_name = st.sidebar.text_input(label="Unesite ime za novi chat", key="newchatname")
    if new_chat_name.strip() != "" and st.sidebar.button(label="Create Chat", key="createchat"):
        thread = client.beta.threads.create()
        st.session_state.thread_id = thread.id

        st.write(type(st.session_state.data))
        new_row = pd.DataFrame({"username": st.session_state.username, 
                        "chat_name": new_chat_name, 
                        "thread_id": st.session_state.thread_id, 
                        "assistant": ovaj_asistent}, index=[0])
        st.session_state.data.append(new_row, ignore_index=True)
        st.write(st.session_state.data)

        csv_data = st.session_state.data.to_csv(index=False)
        blob_client = st.session_state.blob_service_client.get_blob_client("positive-user", "assistant_data.csv")
        blob_client.upload_blob(csv_data, overwrite=True)
        sleep(0.1)
        st.rerun()
    
    chosen_chat = st.sidebar.selectbox(label="Izaberite chat", options=["Select..."] + list(threads_dict.keys()))
    if chosen_chat.strip() not in ["", "Select..."] and st.sidebar.button(label="Select Chat", key="selectchat2"):
        thread = client.beta.threads.retrieve(thread_id=threads_dict.get(chosen_chat))
        st.session_state.thread_id = thread.id
        st.rerun()

    st.sidebar.text("")
    assistant = client.beta.assistants.retrieve(assistant_id=assistant_id)
    if st.session_state.thread_id:
        thread = client.beta.threads.retrieve(thread_id=st.session_state.thread_id)

    instructions = "Please remember to always check each time for every new question if a tool is relevant to your query. \
    Answer only in the Serbian language."

    # ako se desi error run ce po default-u trajati 10 min pre no sto se prekine -- ovo je da ne moramo da cekamo
    try:
        run = client.beta.threads.runs.cancel(thread_id=st.session_state.thread_id, run_id=st.session_state.cancel_run)
    except:
        pass
    run = None


    # pitalica
    if prompt := st.chat_input(placeholder="Postavite pitanje"):
        if st.session_state.thread_id is not None:
            client.beta.threads.messages.create(thread_id=st.session_state.thread_id, role="user", content=prompt) 

            run = client.beta.threads.runs.create(thread_id=st.session_state.thread_id, assistant_id=assistant.id, 
                                                instructions=instructions)
        else:
            st.warning("Molimo Vas da izaberete postojeci ili da kreirate novi chat.")


    # ako se poziva neka funkcija
    if run is not None:
        while True:
            
            sleep(0.3)
            run_status = client.beta.threads.runs.retrieve(thread_id=st.session_state.thread_id, run_id=run.id)

            if run_status.status == 'completed':
                break

            elif run_status.status == 'requires_action':
                tools_outputs = []

                for tool_call in run_status.required_action.submit_tool_outputs.tool_calls:
                    if tool_call.function.name == "web_search_process":
                        arguments = json_loads(tool_call.function.arguments)
                        tool_output = {"tool_call_id":tool_call.id, "output": json_dumps(output)}
                        tools_outputs.append(tool_output)

                    elif tool_call.function.name == "hybrid_search_process":
                        arguments = json_loads(tool_call.function.arguments)
                        output = hybrid_search_process(arguments["upit"])
                        tool_output = {"tool_call_id":tool_call.id, "output": json_dumps(output)}
                        tools_outputs.append(tool_output)

                if run_status.required_action.type == 'submit_tool_outputs':
                    client.beta.threads.runs.submit_tool_outputs(thread_id=st.session_state.thread_id, run_id=run.id, tool_outputs=tools_outputs)

                sleep(0.3)



    try:
        messages = client.beta.threads.messages.list(thread_id=st.session_state.thread_id) 
        for msg in reversed(messages.data): 
            role = msg.role
            content = msg.content[0].text.value 
            if role == 'user':
                st.markdown(f"<div style='background-color:lightblue; padding:10px; margin:5px; border-radius:5px;'><span style='color:blue'>👤 {role.capitalize()}:</span> {content}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='background-color:lightgray; padding:10px; margin:5px; border-radius:5px;'><span style='color:red'>🤖 {role.capitalize()}:</span> {content}</div>", unsafe_allow_html=True)
    except:
        pass

# Deployment on Stremalit Login functionality
deployment_environment = os.environ.get("DEPLOYMENT_ENVIRONMENT")

if deployment_environment == "Streamlit":
    name, authentication_status, username = positive_login(main, " ")
else:
    if __name__ == "__main__":
        main()
