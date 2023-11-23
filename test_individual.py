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
    run_status = client.beta.threads.runs.retrieve(
        thread_id=thread.id, 
        run_id=run.id) 
    # print(run_status.model_dump_json(indent=4)) 
    # If run is completed, get messages 
    if run_status.status == 'completed': 
        messages = client.beta.threads.messages.list(thread_id=thread.id) 
        # Loop through messages and print content based on role 
        for msg in messages.data: 
            role = msg.role 
            content = msg.content[0].text.value 
            print(f"{role.capitalize()}: {content}") 
        break


_ = """
def process_message_with_citations(message):
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
"""

_ = """
for message in assistant_messages_for_run:
    # full_response = process_message_with_citations(message=message)
    st.session_state.messages2.append({"role": "assistant", "content": full_response})
    with st.chat_message(name="assistant"):
        st.markdown(body=full_response, unsafe_allow_html=True)
"""