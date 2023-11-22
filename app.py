import openai
import streamlit as st

from os import getenv
from time import sleep
from requests import get as requests_get
from bs4 import BeautifulSoup
from pdfkit import configuration, from_string
from csv import reader, writer

version = "v0.5"
getenv("OPENAI_API_KEY")
client = openai
assistant_id = "asst_cGNrHE0NDUn8AHcOkBg2sXaq"

with open(file="threads.csv", mode="r") as f:
    reader = reader(f)
    next(reader)    # skip 1st row
    saved_threads = dict(reader)

default_session_states = {
    "file_id_list": [],
    "openai_model": "gpt-4-1106-preview",
    "start_chat": False,
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

def process_message_with_citations(message):    # TESTIRATI EVENTUALLY...
    # extract content and annotations from the message and format citations as footnotes
    message_content = message.content[0].text
    annotations = message_content.annotations if hasattr(message_content, "annotations") else []
    citations = []

    for index, annotation in enumerate(annotations):
        message_content.value = message_content.value.replace(annotation.text, f" [{index + 1}]")

        if (file_citation := getattr(annotation, "file_citation", None)):
            # Retrieve the cited file details (dummy response here since we can"t call OpenAI)
            cited_file = {"filename": "cited_document.pdf"}  # This should be replaced with actual file retrieval
            citations.append(f"[{index + 1}] {file_citation.quote} from {cited_file['filename']}")
        elif (file_path := getattr(annotation, "file_path", None)):
            # Placeholder for file download citation
            cited_file = {"filename": "downloaded_document.pdf"}  # This should be replaced with actual file retrieval
            citations.append(f"[{index + 1}] Click [here](#) to download {cited_file['filename']}")  # The download link should be replaced with the actual download path

    # Add footnotes to the end of the message content
    return message_content.value + "\n\n" + "\n".join(citations)


st.set_page_config(page_title="MultiTool chatbot", page_icon="🀄")

st.sidebar.header(body="MultiTool chatbot")
st.sidebar.markdown(version)

website_url = st.sidebar.text_input(label="Unesite URL web-stranice za scrape-ovanje", key="website_url")
if st.sidebar.button(label="Scrape and Upload"):
    try:
        st.session_state.file_id_list.append(
            upload_to_openai(filepath=text_to_pdf(text=scrape_website(url=website_url), filename="scraped_content.pdf")))
    except:
        st.warning("URL nije validan ili je došlo do greške prilikom scrape-ovanja")

st.sidebar.divider()
# for users to upload their own files
uploaded_file = st.sidebar.file_uploader(label="Upload fajla u OpenAI embeding", key="file_uploader")
if st.sidebar.button(label="Upload File"):
    # upload file provided by user
    if uploaded_file:
        with open(file=f"{uploaded_file.name}", mode="wb") as f:
            f.write(uploaded_file.getbuffer())
        additional_file_id = upload_to_openai(filepath=f"{uploaded_file.name}")
        st.session_state.file_id_list.append(additional_file_id)
        st.sidebar.write(f"Additional File ID: {additional_file_id}")
    else:
        st.warning("Nema upload-ovanog fajla")

# display all uploaded files IDs
if st.session_state.file_id_list:
    st.sidebar.write("ID-jevi upload-ovanih fajlova:")
    for file_id in st.session_state.file_id_list:
        st.sidebar.write(file_id)
        # povezivanje fajla sa asistentom
        assistant_file = client.beta.assistants.files.create(
            assistant_id=assistant_id, 
            file_id=file_id,)

st.sidebar.divider()
chat_name = st.sidebar.text_input(label="Unesite ime ako započinjete novi chat", key="chatname")
if chat_name:
    st.session_state.threads[chat_name] = st.session_state.thread_id
curr_chat = st.sidebar.selectbox(label="Izaberite neki od postojecih chat-ova", options=[chat_name] + list(saved_threads.keys()))


if st.sidebar.button("Start Chat"):     # start the chat session
    if curr_chat not in st.session_state.threads:
        with open(file="threads.csv", mode="a", newline="") as f:
            writer(f).writerow([chat_name, st.session_state.threads[curr_chat]])

    assistant = client.beta.assistants.retrieve(assistant_id="asst_25WzWOh32CdYuTeoX38gIJXh")
    thread = client.beta.threads.retrieve(thread_id="thread_IHVzQg3xUg4xboZc3pX3Het6")
    message = client.beta.threads.messages.create( thread_id=thread.id, role="user", content= pitanje ) 
    run = client.beta.threads.runs.create( thread_id=thread.id, assistant_id=assistant.id, instructions="Please answer in the serbian language. For answers consult the file provided. " ) 
    
    st.session_state.start_chat = True

    while True: 
        sleep(0.1)
        run_status = client.beta.threads.runs.retrieve(
            thread_id=thread.id, 
            run_id=run.id)
        # If run is completed, get messages 
        if run_status.status == 'completed': 
            messages = client.beta.threads.messages.list(thread_id=thread.id) 
            # Loop through messages and print content based on role 
            for msg in messages.data: 
                role = msg.role 
                content = msg.content[0].text.value 
                st.write(f"{role.capitalize()}: {content}") 
            break

    thread = client.beta.threads.create()
    st.write("thread id: ", thread.id)
    client.beta.threads.messages.list(thread_id=st.session_state.threads[curr_chat])
    st.session_state.thread_id = st.session_state.threads[curr_chat]
    st.write("thread id: ", st.session_state.threads[curr_chat])


if st.session_state.start_chat:
    # Initialize the model and messages list if not already in session state
    st.session_state.messages = client.beta.threads.messages.list(thread_id=st.session_state.thread_id)
    st.session_state.messages2 = []
    for message in st.session_state.messages2:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(placeholder="What is up?"):
        # Add user message to the state and display it
        st.session_state.messages2.append({"role": "user", "content": prompt})
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
        
        for message in assistant_messages_for_run:
            full_response = process_message_with_citations(message=message)
            st.session_state.messages2.append({"role": "assistant", "content": full_response})
            with st.chat_message(name="assistant"):
                st.markdown(body=full_response, unsafe_allow_html=True)
else:
    st.write("Učitajte fajlove po potrebi i kliknite na 'Start Chat'")



_ = """

import openai 
import time
# Initialize the client 
client = openai.OpenAI() 

# Step 1: Create an Assistant 
# assistant = client.beta.assistants.create( name="Math Tutor 2", instructions="You are a personal math tutor. Write and run code to answer math questions.", tools=[{"type": "code_interpreter"}], model="gpt-4-1106-preview" ) 
# or retireve assistant by id

assistant = client.beta.assistants.retrieve(assistant_id="asst_25WzWOh32CdYuTeoX38gIJXh")

# Step 2: Create a Thread 
# thread = client.beta.threads.create() 
# or retrieve thread by id

thread = client.beta.threads.retrieve(thread_id="thread_IHVzQg3xUg4xboZc3pX3Het6")

# print("krajnji ", client.beta.threads.messages.list( thread_id="thread_IHVzQg3xUg4xboZc3pX3Het6" ) )

pitanje = input("Pitanje: ")

# Step 3: Add a Message to a Thread 

message = client.beta.threads.messages.create( thread_id=thread.id, role="user", content= pitanje ) 

# Step 4: Run the Assistant 

run = client.beta.threads.runs.create( thread_id=thread.id, assistant_id=assistant.id, instructions="Please answer in the serbian language. For answers consult the file provided. " ) 

# print(run.model_dump_json(indent=4)) 

while True: 
    # Wait for 5 seconds 
    time.sleep(0.1) # Retrieve the run status 
    run_status = client.beta.threads.runs.retrieve( thread_id=thread.id, run_id=run.id ) 
    # print(run_status.model_dump_json(indent=4)) 
    # If run is completed, get messages 
    if run_status.status == 'completed': 
        messages = client.beta.threads.messages.list( thread_id=thread.id ) 
        # Loop through messages and print content based on role 
        for msg in messages.data: 
            role = msg.role 
            content = msg.content[0].text.value 
            print(f"{role.capitalize()}: {content}") 
        break

"""
