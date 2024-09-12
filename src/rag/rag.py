from fs.memoryfs import MemoryFS
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.schema import Document


from .in_memory_loader import InMemoryLoader

RAG_TEMPLATE = """
        <|system|>
        You are a supportive AI Assistant in a Retrieval Augmentation Generator (RAG). You should follow instructions extremely well and help the user to clarify doubts and provide insightful, truthful and direct answers.

        Please tell "I don't know, this is not clearly in the context" if user query is not in CONTEXT.
        </section>

        <section>
        CONTEXT:
        <context>
        {context}
        </context>
        </section>

        <section>
        <|user|>
        {query}
        </section>

        <|assistant|>
        """

class RAG:
    def __init__(self, web_urls, yt_transcript, mem_fs):
        self.docs = self.data_loaders(web_urls, yt_transcript, mem_fs)
        self.vector_store, self.split_documents = self.vector_db(self.docs)
        self.ensemble_retriever = self.hybrid_search_ensemble_retrieval(self.vector_store, self.split_documents)
        self.compression_retriever = self.re_ranking(self.ensemble_retriever)

        self.llm = Ollama(model="llama3.1",temperature=0.3)

    def data_loaders(self, web_urls, yt_transcript, mem_fs):
        # Loaders
        web_loader = WebBaseLoader(web_urls)
        web_loader.requests_per_second = 1
        yt_loader = InMemoryLoader(mem_fs, yt_transcript)
        loader_all = MergedDataLoader(loaders=[web_loader, yt_loader])
        docs_all = loader_all.load()

        return docs_all

    def vector_db(self, docs):
        text_splitter = RecursiveCharacterTextSplitter()
        split_documents = text_splitter.split_documents(docs)

        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vector_store = FAISS.from_documents(split_documents, embeddings)
        return vector_store, split_documents

    def hybrid_search_ensemble_retrieval(self, vector_store, split_documents):
        # Hybrid search + ensemble retrieval
        K = 5
        retriever_vectordb = vector_store.as_retriever(search_kwargs={"k": K})

        keyword_retriever = BM25Retriever.from_documents(split_documents)
        keyword_retriever.k =  K

        ensemble_retriever = EnsembleRetriever(retrievers=
                                            [retriever_vectordb,keyword_retriever],
                                            weights=[0.5, 0.5])
        
        return ensemble_retriever
        
    def re_ranking(self, ensemble_retriever):
        # Implementing Re-ranking
        compressor = FlashrankRerank()
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble_retriever
        )
        return compression_retriever
        
    def augment_generate(self, query):
        # Augmentation and Generation
        prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
        output_parser = StrOutputParser()

        chain = (
            {"context": self.compression_retriever, "query": RunnablePassthrough()}
            | prompt
            | self.llm
            | output_parser
        )

        response = chain.invoke(query)
        return response
