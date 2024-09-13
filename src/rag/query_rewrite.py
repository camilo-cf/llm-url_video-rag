from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from os import getenv

base_url = getenv("OLLAMA_HOST")
if base_url == None:
    base_url = "localhost:11434"

base_url = "http://"+base_url

def query_rewrite(question):
    model = Ollama(
        base_url = base_url,
        model="gemma2",
        temperature=0.0,
        )

    template = """
    You are a helpful assistant expert in improve vector search queries.\
    Your task is to work with the query I will provide, where you will refine the given a search query,\
    with grammar, punctuation and clarity needed for vector search engine input.\
    In the improved query you will only answer the improved query, nothing else,\
    do not share any thoughts, explanations or additional text. If you don't understand\
    or you do not feel confident in your knowledge, please just answer with the Query as it is.
    
    Query: {question}
    Improved Query:
    """

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    query = chain.invoke({"question": question})
    return query