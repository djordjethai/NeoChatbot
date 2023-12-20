import openai
import streamlit as st
import os
import json
from time import sleep
from myfunc.mojafunkcija import inner_hybrid

import nltk     # kasnije ce se paketi importovati u funkcijama
from langchain.utilities import GoogleSerperAPIWrapper
from streamlit_extras.stylable_container import stylable_container

import clipboard


def on_copy_click(text):
    # st.session_state.copied.append(text)
    clipboard.copy(text)
    

st.set_page_config(page_title="Chatbot", page_icon="🤖")

version = "v1.0.0 Samo chatbot za website"
st.caption(f"Ver. {version}")
os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()
assistant_id = "asst_cLf9awhvTT1zxY23K3ebpXbs"  # printuje se u drugoj skripti, a moze jelte da se vidi i na OpenAI Playground-u
client.beta.assistants.retrieve(assistant_id=assistant_id)


def main():
    count = 0   
    # Inicijalizacija session state-a
    default_session_states = {
        "file_id_list": [],
        "openai_model": "gpt-4-1106-preview",
        "messages": [],
        "thread_id": None,
        "is_deleted": False,
        "cancel_run": None,
        "namespace": "positive",
        }
    
    for key, value in default_session_states.items():
        if key not in st.session_state:
            st.session_state[key] = value


    def web_serach_process(q: str) -> str:
        return GoogleSerperAPIWrapper(environment=os.environ["SERPER_API_KEY"]).run(q)


    def hybrid_search_process(upit: str) -> str:
        stringic = inner_hybrid(upit)
        return stringic
        
    
    if st.session_state.thread_id is None:
        thread = client.beta.threads.create()
        st.session_state.thread_id = thread.id

    
    assistant = client.beta.assistants.retrieve(assistant_id=assistant_id)
    if st.session_state.thread_id:
        thread = client.beta.threads.retrieve(thread_id=st.session_state.thread_id)

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

            run = client.beta.threads.runs.create(thread_id=st.session_state.thread_id, assistant_id=assistant.id)
                                                
        else:
            st.warning("Molimo Vas da izaberete postojeci ili da kreirate novi chat.")


    # ako se poziva neka funkcija
    with stylable_container(
                    key="bottom_content",
                    css_styles="""
                        {
                            position: fixed;
                            bottom: 150px;
                        }
                        """,
                    ):
                
        with st.spinner("🤖 Chatbot razmislja..."):
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
                                arguments = json.loads(tool_call.function.arguments)
                                try:
                                    output = web_serach_process(arguments["query"])
                                except:
                                    output = web_serach_process(arguments["q"])

                                tool_output = {"tool_call_id":tool_call.id, "output": json.dumps(output)}
                                tools_outputs.append(tool_output)
                    
                            elif tool_call.function.name == "web_search_process":
                                arguments = json.loads(tool_call.function.arguments)
                                tool_output = {"tool_call_id":tool_call.id, "output": json.dumps(output)}
                                tools_outputs.append(tool_output)

                            elif tool_call.function.name == "hybrid_search_process":
                                arguments = json.loads(tool_call.function.arguments)
                                output = hybrid_search_process(arguments["upit"])
                                tool_output = {"tool_call_id":tool_call.id, "output": json.dumps(output)}
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
                
                st.button("📋", on_click=on_copy_click, args=([content]), key=count)
               
                count += 1                        
    except:     
        pass
    
    

    
    
if __name__ == "__main__":
    main()


# # Display sources
#     for thread_message in st.session_state.messages.data:
#         for message_content in thread_message.content:
#             # Access the actual text content
#             message_content = message_content.text
#             annotations = message_content.annotations
#             citations = []
            
#             # Iterate over the annotations and add footnotes
#             for index, annotation in enumerate(annotations):
#                 # Replace the text with a footnote
#                 message_content.value = message_content.value.replace(annotation.text, f' [{index}]')
            
#                 # Gather citations based on annotation attributes
#                 if (file_citation := getattr(annotation, 'file_citation', None)):
#                     cited_file = client.files.retrieve(file_citation.file_id)
#                     citations.append(f'[{index}] {file_citation.quote} from {cited_file.filename}')
#                 elif (file_path := getattr(annotation, 'file_path', None)):
#                     cited_file = client.files.retrieve(file_path.file_id)
#                     citations.append(f'[{index}] Click <here> to download {cited_file.filename}')
#                     # Note: File download functionality not implemented above for brevity

#             # Add footnotes to the end of the message before displaying to user
#             message_content.value += '\n' + '\n'.join(citations)