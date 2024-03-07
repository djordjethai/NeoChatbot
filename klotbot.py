import streamlit as st
from openai import OpenAI
from myfunc.retrievers import HybridQueryProcessor
from sql_prompt import PromptDatabase

client=OpenAI()
processor = HybridQueryProcessor(alpha=0.5, score=0.0, namespace="zapisnici") # namespace moze i iz env

with PromptDatabase() as db:
    result2 = db.query_sql_record("CHAT_TOOLS_PROMPT")
    result3 = db.query_sql_record("ALT_ASISTENT")
    
def main():
    
    if "messages" not in st.session_state:
        st.session_state.messages = {}
    if "thread_id" not in st.session_state:
        new_thread_id = "skroznovi"
        st.session_state["thread_id"] = new_thread_id
        st.session_state.messages[new_thread_id] = []
    
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = result3.get('prompt_text', 'You are helpful assistant that always writes in Sebian.')
        
    current_thread_id = st.session_state["thread_id"]
    # Main conversation UI
    if prompt := st.chat_input("Kako vam mogu pomoci?"):
            
            context, scores = processor.process_query_results(prompt)
            complete_prompt = result2.get('prompt_text', 'You are helpful assistant that always writes in Sebian.').format(prompt=prompt, context=context)
            
            # Append user prompt to the conversation
            st.session_state.messages[current_thread_id].append({"role": "user", "content": complete_prompt})
        
            # Display user prompt in the chat
            with st.chat_message("user"):
                st.markdown(prompt)
                
            # Generate and display the assistant's response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                for response in client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    temperature=0,
                    messages=st.session_state.messages[current_thread_id],
                    stream=True,
            ):
                    full_response += (response.choices[0].delta.content or "")
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            # Append assistant's response to the conversation
            st.session_state.messages[current_thread_id].append({"role": "assistant", "content": full_response})
        

    

if __name__ == "__main__":
        main()