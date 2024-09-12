from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

def query_rewrite(question):
    model = Ollama(
        model="phi3.5",
        temperature=0.1
        )

    template = """
    You are a helpful assistant expert in improve vector search queries.\
    Your task is to work with the query I will provide, where you will refine the given a search query,\
    with grammar, punctuation and clarity needed for vector search engine input.\
    In the improved query you will only answer the improved query, nothing else,\
    do not share any thoughts, explanations or additional text.
    
    Query: {question}
    Improved Query:
    """

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    query = chain.invoke({"question": question})
    return query