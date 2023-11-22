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
c.	Novi Vision model koji je u stanju da „prepozna“ tekst sa slike, čak i na srpskom jeziku. Ovo je korisno kod prikupljanja i čuvanja dokumenata u iz ne struktuiranog u struktuiranom obliku. Model prepoznaje pojmove, a kao i ostali novi modeli može da da odgovor i u JSON formatu, što potpuno olakšava prenos podataka u bilo koji drugi struktuirani oblik

3.	Šta to nama omogućava
a.	Svi naši asistenti će od sada biti konverzacioni, sa pamćenjem svih prethodnih razgovora koji će se moći pozvati po želji.
b.	Upotrebu Hybrid searcha i SQL search za polu struktuirane i struktuirane dokumente
c.	Upotrebu semantic searcha koji je ugrađen u nove asistente
d.	Mogućnost upotrebe na samo indeksiranih dokumenata, već i ad hoc dokumenata pa čak i web stranica
Ovaj ceo sistem je sada dosta robusniji i brži kao a i omogućava nam da se sada potpuno posvetimo klijentima.
"""