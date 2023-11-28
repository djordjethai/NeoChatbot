from langchain.utilities import GoogleSerperAPIWrapper

from os import environ

def web_serach_process(query: str) -> str:
    return GoogleSerperAPIWrapper(environment=environ["SERPER_API_KEY"]).run(query=query)

x = "Koji je danas datum?"

y = web_serach_process(x)
print(y)